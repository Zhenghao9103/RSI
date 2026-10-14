import pytest

from backend.app.skills.policy import Policy, SkillTrigger, validate_draft


def trace(**kwargs):
    value = {
        "task_id": "task-1",
        "user_query": "Repair indexing",
        "trace_events": [{"tool": "repair", "args": {"mode": "rebuild"}, "status": "success"}],
        "final_answer": "Index repaired",
        "completion_status": "success",
    }
    value.update(kwargs)
    return value


def draft(**kwargs):
    value = {
        "name": "repair-index",
        "description": "Repair stale indexes",
        "confidence": 0.9,
        "body": "## Applicability\nStale index\n## Steps\nRebuild\n## Constraints\nPreserve files",
    }
    value.update(kwargs)
    return value


def policy(result):
    return Policy(llm=lambda messages, *, max_tokens: result)


def test_generate_requires_usable_trace_and_validated_draft():
    assert policy(draft()).generate(trace())["name"] == "repair-index"
    with pytest.raises(ValueError):
        policy(draft()).generate(trace(trace_events=[]))


@pytest.mark.parametrize(
    "bad",
    [
        draft(name="../escape"),
        draft(name="two--hyphens"),
        draft(description=""),
        draft(body="Generic advice"),
    ],
)
def test_generation_rejects_bad_drafts(bad):
    with pytest.raises(ValueError):
        policy(bad).generate(trace())


def test_generation_requires_explicit_confidence():
    result = draft()
    del result["confidence"]
    with pytest.raises(ValueError, match="confidence"):
        policy(result).generate(trace())


def test_new_cluster_scope_rejection_does_not_require_master_draft():
    result = {
        "action": "NEW_CLUSTER",
        "reason": "Different responsibility",
        "has_information": True,
    }
    assert policy(result).consolidate(draft(), draft(), []) == result
    with pytest.raises(ValueError):
        policy(result).consolidate(draft(), None, [])


def test_structured_generation_renders_applicability_steps_constraints():
    result = draft()
    del result["body"]
    result.update(
        applicability="Stale index", steps=["Inspect", "Rebuild"], constraints=["No data deletion"]
    )
    generated = policy(result).generate(trace())
    assert "## Applicability" in generated["body"]
    assert "1. Inspect" in generated["body"]


def test_consolidation_validates_relation_and_target():
    result = {
        "action": "UPDATE_REFERENCE",
        "reason": "Adds recovery",
        "has_information": True,
        "reference_id": "ref-1",
        "reference": draft(),
    }
    assert (
        policy(result).consolidate(draft(), draft(), [{"id": "ref-1"}])["action"]
        == "UPDATE_REFERENCE"
    )
    for malformed in [
        dict(result, action="UNKNOWN"),
        dict(result, has_information=False),
        dict(result, reference_id=None),
        dict(result, reason=""),
    ]:
        with pytest.raises(ValueError):
            policy(malformed).consolidate(draft(), draft(), [{"id": "ref-1"}])


@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"independent": False, "complete_workflow": True},
        {"independent": True, "complete_workflow": False},
    ],
)
def test_promote_requires_explicit_independence_and_complete_workflow(flags):
    result = {
        "action": "PROMOTE",
        "reason": "Independent recovery",
        "reference_id": "ref-1",
        "draft": draft(),
        **flags,
    }
    with pytest.raises(ValueError):
        policy(result).maintenance({}, draft(), [], [], {})


def test_split_requires_two_groups_with_explicit_assignments():
    result = {
        "action": "SPLIT",
        "reason": "Separate concerns",
        "independent": True,
        "groups": [
            {"master": draft(), "reference_ids": ["r1"], "member_ids": ["m1"]},
            {"master": draft(name="other-index"), "reference_ids": ["r2"], "member_ids": ["m2"]},
        ],
    }
    assert len(policy(result).maintenance({}, draft(), [], [], {})["groups"]) == 2
    result["groups"] = result["groups"][:1]
    with pytest.raises(ValueError):
        policy(result).maintenance({}, draft(), [], [], {})


def test_model_failures_propagate_without_fabricated_drafts():
    def fail(messages, *, max_tokens):
        raise RuntimeError("Provider offline")

    with pytest.raises(RuntimeError, match="Provider offline"):
        Policy(llm=fail).generate(trace())


@pytest.mark.parametrize("result", [None, [], "not-json"])
def test_non_object_model_responses_fail_explicitly(result):
    with pytest.raises(ValueError, match="JSON object"):
        policy(result).generate(trace())


@pytest.mark.parametrize("action", [[], {}, None, 7])
def test_malformed_action_types_raise_value_error(action):
    result = {"action": action, "reason": "Malformed"}
    with pytest.raises(ValueError):
        policy(result).consolidate(draft(), draft(), [])
    with pytest.raises(ValueError):
        policy(result).maintenance({}, draft(), [], [], {})


@pytest.mark.parametrize(
    "result",
    [
        {"action": "KEEP", "reason": "Stable"},
        {
            "action": "PROMOTE",
            "reason": "Independent",
            "reference_id": "r1",
            "draft": draft(),
            "independent": True,
            "complete_workflow": True,
        },
        {"action": "MERGE", "reason": "Duplicate", "target_cluster_id": "c2"},
        {
            "action": "MERGE",
            "reason": "Duplicate",
            "source_reference_ids": ["r1"],
            "target_reference_id": "r2",
            "reference": draft(),
        },
        {"action": "REFACTOR", "reason": "Shorten", "reference_id": "r1", "reference": draft()},
        {"action": "ARCHIVE", "reason": "Obsolete", "reference_id": "r1"},
    ],
)
def test_valid_maintenance_shapes(result):
    assert policy(result).maintenance({}, draft(), [], [], {})["action"] == result["action"]


def test_plain_answers_and_generic_search_sequences_never_trigger():
    trigger = SkillTrigger()
    assert not trigger.inspect(trace(trace_events=[], user_query="保存这个答案"))["triggered"]
    search = trace(
        trace_events=[{"tool": "search", "args": {"query": "paper"}, "status": "success"}]
    )
    previous = [dict(search, task_id="p1"), dict(search, task_id="p2")]
    assert not trigger.inspect(search, previous)["triggered"]


@pytest.mark.parametrize("query", ["Please save this method", "保存这个方法和经验"])
def test_explicit_save_method_with_observed_inspection(query):
    current = trace(
        user_query=query,
        trace_events=[{"type": "tool_result", "status": "ok", "tool": "inspect_document"}],
    )
    assert "explicit_save" in SkillTrigger().inspect(current)["signals"]


@pytest.mark.parametrize("tool", ["hybrid_search", "inspect_document", "get_chunk_neighbors"])
def test_repeated_routine_document_retrieval_is_not_strategy(tool):
    current = trace(trace_events=[{"tool": tool, "status": "ok"}])
    assert not SkillTrigger().inspect(
        current, [dict(current, task_id="p1"), dict(current, task_id="p2")]
    )["triggered"]


def test_generate_accepts_when_to_use_heading():
    result = draft(
        body="## When to use\nStale index\n## Steps\nInspect\n## Constraints\nPreserve files"
    )
    assert policy(result).generate(trace())["name"] == "repair-index"


def test_reusable_solution_annotation_requires_substantive_observed_steps():
    trigger = SkillTrigger()
    assert "reusable_solution" in trigger.inspect(trace(reusable_solution=True))["signals"]
    assert not trigger.inspect(trace(reusable_solution=True, trace_events=[]))["triggered"]


def test_import_validation_does_not_require_generation_headings():
    assert validate_draft(draft(body="Existing workflow"))["body"] == "Existing workflow"


def test_explicit_save_repair_and_uncovered_type_trigger():
    trigger = SkillTrigger()
    assert "explicit_save" in trigger.inspect(trace(user_query="将这套流程保存为技能"))["signals"]
    repair = trace(
        trace_events=[
            {"tool": "repair", "status": "error"},
            {"tool": "repair", "status": "success"},
        ]
    )
    assert "error_correction" in trigger.inspect(repair)["signals"]
    assert "uncovered_task_type" in trigger.inspect(trace(new_task_type="index-repair"))["signals"]
    assert not trigger.inspect(trace(new_task_type="index-repair"), covered=True)["triggered"]


def test_repetition_requires_distinct_tasks_and_semantic_strategy():
    trigger = SkillTrigger()
    current = trace()
    previous = [trace(task_id="p1"), trace(task_id="p2")]
    assert "repeated_strategy" in trigger.inspect(current, previous)["signals"]
    assert not trigger.inspect(current, [trace(task_id="p1"), trace(task_id="p1")])["triggered"]
    changed = trace(
        trace_events=[{"tool": "repair", "args": {"mode": "delete"}, "status": "success"}]
    )
    assert not trigger.inspect(changed, previous)["triggered"]


def test_failed_trace_does_not_trigger():
    assert not SkillTrigger().inspect(trace(user_query="保存技能", completion_status="failed"))[
        "triggered"
    ]
