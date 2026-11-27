from rsi import Memory, answering, get_settings, procedural_retrieval, procedural_writes


def test_learn_ask_reopen(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "procedural_memory_mode", "rsi")
    monkeypatch.setattr(procedural_retrieval, "embed_text", lambda *a, **k: [1.0, 0.0])
    monkeypatch.setattr(
        procedural_writes,
        "reflect",
        lambda _: {
            "condition": "math explanations",
            "steps": ["start with an example"],
        },
    )
    monkeypatch.setattr(procedural_writes, "judge_advice", lambda *_: {"action": "add"})
    monkeypatch.setattr(
        answering,
        "complex_text_completion",
        lambda messages: (
            "Example first."
            if "start with an example" in str(messages)
            else "Definition first."
        ),
    )
    database = tmp_path / "memory.db"
    with Memory(database) as memory:
        assert (
            memory.learn("From now on, explain mathematics with examples first.")
            == "applied"
        )
    with Memory(database) as memory:
        result = memory.ask("Explain the mean")
        assert result.answer == "Example first."
        assert len(result.used_skills) == 1
