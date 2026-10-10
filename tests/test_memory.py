from agentforge.memory import ConversationMemory, LongTermMemory, estimate_tokens
from fakes import ScriptedLLM, text


def _convo(mem: ConversationMemory, turns: int) -> None:
    for i in range(turns):
        mem.add_user(f"question {i} " + "pad " * 200)
        mem.add_assistant([{"type": "tool_use", "id": f"t{i}", "name": "x", "input": {"i": i}}])
        mem.add_tool_results(
            [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "r" * 100}]
        )
        mem.add_assistant([{"type": "text", "text": f"answer {i}"}])


def test_compaction_never_splits_tool_pairs():
    mem = ConversationMemory(max_context_tokens=500, keep_recent_tokens=600)
    _convo(mem, 6)
    assert mem.needs_compaction()
    llm = ScriptedLLM([text("SUMMARY", tokens=(7, 3))])
    usage = mem.compact(llm)

    assert usage and usage.total == 10
    first = mem.messages[0]
    assert first["role"] == "user" and isinstance(first["content"], str)
    assert "SUMMARY" in first["content"]
    assert len(mem.messages) < 24
    # every tool_result must be preceded by its tool_use
    for i, m in enumerate(mem.messages):
        if m["role"] == "user" and isinstance(m["content"], list):
            assert mem.messages[i - 1]["content"][0]["type"] == "tool_use"
    assert estimate_tokens(mem.messages) < estimate_tokens(
        [{"role": "user", "content": "pad " * 1500}]
    )


def test_compaction_noop_without_user_turn_boundary():
    mem = ConversationMemory()
    mem.add_user("only one")
    assert mem.compact(ScriptedLLM([])) is None


def test_long_term_memory_search_ranking_and_tags():
    ltm = LongTermMemory()
    ltm.add("The deploy pipeline runs on GitHub Actions", ["ci"])
    ltm.add("User prefers answers in Bahasa Indonesia", ["preference"])
    ltm.add("Staging database is PostgreSQL 16", ["db"])
    top = ltm.search("which database do we use in staging?")
    assert top and "PostgreSQL" in top[0].text
    assert ltm.search("preference language")[0].tags == ["preference"]
    assert ltm.search("zzzz") == []
