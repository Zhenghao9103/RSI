import importlib.util
import json

import pytest


def test_cli_module_exists():
    try:
        found = importlib.util.find_spec("backend.app.skills.__main__")
    except ModuleNotFoundError:
        found = None
    assert found is not None, "Skill pool CLI has not been implemented"


def cli():
    from backend.app.skills.__main__ import main

    return main


def test_help_documents_commands_without_loading_models(capsys):
    with pytest.raises(SystemExit) as result:
        cli()(["--help"])
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "integrate" in output and "recover" in output and "smoke" in output


def test_cli_read_invalid_json_has_clear_error_without_manager(tmp_path, capsys):
    bad = tmp_path / "trace.json"
    bad.write_text("{broken", encoding="utf-8")
    assert (
        cli()(
            ["generate", str(bad)],
            manager_factory=lambda **kw: pytest.fail("Manager should not start"),
        )
        == 1
    )
    assert "JSON" in capsys.readouterr().err


def test_cli_unicode_output_and_resource_closure(tmp_path, capsys):
    class Manager:
        def __init__(self, **kwargs):
            self.closed = False
            assert kwargs["root"] == str(tmp_path)

        def status(self):
            return {"说明": "独立模块"}

        def close(self):
            self.closed = True

    managers = []

    def factory(**kwargs):
        manager = Manager(**kwargs)
        managers.append(manager)
        return manager

    assert cli()(["--root", str(tmp_path), "status"], manager_factory=factory) == 0
    assert json.loads(capsys.readouterr().out) == {"说明": "独立模块"}
    assert managers[0].closed


def test_cli_missing_candidate_reports_error_and_closes(capsys):
    class Manager:
        closed = False

        def integrate(self, candidate_id):
            raise KeyError(candidate_id)

        def close(self):
            self.closed = True

    manager = Manager()
    assert cli()(["integrate", "missing"], manager_factory=lambda **kw: manager) == 1
    assert manager.closed
    assert "missing" in capsys.readouterr().err


def test_offline_demo_verifies_promotion_split_and_persistence(tmp_path):
    from backend.app.skills.demo import run_demo

    result = run_demo(tmp_path)
    assert result["mode"] == "offline-demo"
    assert all(result["checks"].values())
    assert result["checks"]["similar_candidates_same_cluster"]
    assert result["checks"]["third_candidate_updates_reference"]
    assert result["checks"]["promotion_migrates_members"]
    assert result["checks"]["promoted_reference_is_collaboration_pointer"]
    assert result["checks"]["split_preserves_vectors_and_centroids"]
    assert result["checks"]["reopen_discovery_consistent"]


def test_temporary_demo_releases_chroma_files_before_cleanup():
    from pathlib import Path

    from backend.app.skills.demo import run_demo

    result = run_demo()
    assert result["temporary_storage_removed"]
    assert not Path(result["root"]).exists()


def test_cli_maintain_due_reports_local_review_results(capsys):
    class Manager:
        def maintain_due(self):
            return [{"cluster_id": "c1", "action": "KEEP"}]

        def close(self):
            pass

    assert cli()(["maintain-due"], manager_factory=lambda **kw: Manager()) == 0
    assert json.loads(capsys.readouterr().out)[0]["action"] == "KEEP"


@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
def test_worker_accepts_mixed_native_stderr_encoding(monkeypatch):
    from backend.app.skills import demo

    original_run = demo.subprocess.run

    def native_log_run(argv, **kwargs):
        argv = list(argv)
        argv[2] = "import sys; sys.stderr.buffer.write(bytes([207])); " + argv[2]
        return original_run(argv, **kwargs)

    monkeypatch.setattr(demo.subprocess, "run", native_log_run)
    assert demo.run_demo()["temporary_storage_removed"]


def test_worker_failure_exposes_stage_category_without_provider_text(tmp_path, monkeypatch, capsys):
    from backend.app.skills import demo
    from backend.app.services.model_clients import ModelClientError

    def fail(root, with_llm=False):
        raise ModelClientError("private-api-key-and-request", category="connection_error")

    monkeypatch.setattr(demo, "_run_smoke", fail)
    assert demo._worker_entry(tmp_path, "smoke", True) == 1
    output = capsys.readouterr()
    assert "connection_error" in output.out and "smoke" in output.out
    assert "private-api-key-and-request" not in output.out + output.err


def test_llm_smoke_integrates_discovers_and_loads_without_remote_models(tmp_path, monkeypatch):
    from backend.app.skills import demo
    import backend.app.skills as skills
    from backend.app.rag import vector_store

    monkeypatch.setattr(vector_store, "embed_text", lambda text: [1.0, 0.0, 0.0])

    class Manager:
        def __init__(self, **kwargs):
            pass

        def generate_candidate(self, trace):
            return {
                "id": "candidate",
                "name": "index-repair",
                "description": "Repair indexes",
            }

        def integrate(self, candidate_id):
            return {
                "action": "NEW_CLUSTER",
                "skill_id": "skill",
                "reference_id": "reference",
                "cluster_id": "cluster",
            }

        def discover(self, query):
            return [{"id": "skill"}]

        def load_master(self, skill_id):
            return "Master body"

        def load_reference(self, reference_id):
            return "Reference body"

        def close(self):
            pass

    monkeypatch.setattr(skills, "SkillManager", Manager)
    result = demo._run_smoke(tmp_path, with_llm=True)
    assert result["llm"]["integration_action"] == "NEW_CLUSTER"
    assert result["llm"]["discover_load_round_trip"]
