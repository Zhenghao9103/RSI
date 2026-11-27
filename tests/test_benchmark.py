import json

from rsi import answering, get_settings, procedural_retrieval, procedural_writes
from rsi.benchmark import run


def test_benchmark_uses_same_learned_skill_for_flat_and_rsi(
    db_session, tmp_path, monkeypatch
):
    monkeypatch.setattr(get_settings(), "agent_model", "gpt-4o")
    monkeypatch.setattr(
        procedural_retrieval, "embed_text", lambda *_a, **_k: [1.0, 0.0]
    )
    monkeypatch.setattr(
        procedural_writes,
        "reflect",
        lambda _: {
            "condition": "teaching math",
            "steps": ["use Example before Definition"],
        },
    )
    monkeypatch.setattr(procedural_writes, "judge_advice", lambda *_: {"action": "add"})
    monkeypatch.setattr(
        answering,
        "complex_text_completion",
        lambda messages: (
            "Example: average 2 and 4. Definition: sum divided by count."
            if len(messages) == 3
            else "The average is the sum divided by the count."
        ),
    )
    cases = {
        "learning": [
            {
                "key": "math",
                "previous_answer": "A definition.",
                "feedback": "From now on, give an example before the definition.",
            }
        ],
        "evaluation": [
            {
                "id": "math-1",
                "key": "math",
                "question": "What is the mean?",
                "markers": ["Example", "Definition"],
                "topic": ["sum"],
            }
        ],
    }
    target = tmp_path / "results.json"
    run(cases, db_session, target)
    data = json.loads(target.read_text(encoding="utf8"))
    assert len(data["snapshot"]) == 1
    assert [row["mode"] for row in data["evaluation"]] == [
        "off",
        "write",
        "flat",
        "rsi",
    ]
    assert [row["hit"] for row in data["evaluation"]] == [False, False, True, True]
    assert [row["success"] for row in data["evaluation"]] == [False, False, True, True]
    assert data["evaluation"][2]["used"] == data["evaluation"][3]["used"]
