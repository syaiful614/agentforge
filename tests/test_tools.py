from typing import Annotated

import pytest
from pydantic import Field

from agentforge.builtin_tools import calculator, make_file_tools
from agentforge.tools import MAX_OUTPUT_CHARS, ToolRegistry, tool


@tool
def add(a: Annotated[int, Field(description="first")], b: int = 1) -> int:
    """Add two integers."""
    return a + b


def test_schema_generated_from_type_hints():
    s = add.to_api()
    assert s["name"] == "add"
    assert s["description"] == "Add two integers."
    props = s["input_schema"]["properties"]
    assert props["a"]["type"] == "integer" and props["a"]["description"] == "first"
    assert s["input_schema"]["required"] == ["a"]
    assert "title" not in s["input_schema"]


def test_run_ok_and_non_string_serialised():
    assert add.run({"a": 2, "b": 3}).content == "5"


def test_invalid_args_become_error_result_not_exception():
    r = add.run({"a": "not-int"})
    assert r.is_error and "Invalid arguments" in r.content


def test_tool_exception_captured():
    @tool
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("kaput")

    r = boom.run({})
    assert r.is_error and "RuntimeError: kaput" in r.content


def test_output_truncated():
    @tool
    def big() -> str:
        """Huge output."""
        return "x" * (MAX_OUTPUT_CHARS + 500)

    assert "truncated 500 chars" in big.run({}).content


def test_missing_docstring_rejected():
    def nodoc(x: int) -> int:
        return x

    with pytest.raises(ValueError):
        tool(nodoc)


def test_registry_duplicate_and_subset():
    reg = ToolRegistry([add, calculator])
    with pytest.raises(ValueError):
        reg.register(add)
    sub = reg.subset(["calculator", "nope"])
    assert sub.names == ["calculator"]


@pytest.mark.parametrize(
    "expr,expected",
    [("(17*23)+sqrt(144)", "403"), ("2**10", "1024"), ("10/4", "2.5"), ("-pi*0", "-0")],
)
def test_calculator(expr, expected):
    assert calculator.run({"expression": expr}).content == expected


@pytest.mark.parametrize(
    "expr", ["__import__('os').system('id')", "open('/etc/passwd')", "2**100000", "'a'*5", "x+1"]
)
def test_calculator_rejects_code_and_abuse(expr):
    assert calculator.run({"expression": expr}).is_error


def test_file_tools_sandbox(tmp_path):
    (tmp_path / "a.txt").write_text("hello\nworld")
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("nope")
    tools = {t.name: t for t in make_file_tools(tmp_path)}

    assert tools["read_file"].run({"path": "a.txt"}).content == "hello\nworld"
    assert tools["read_file"].run({"path": "../secret.txt"}).is_error
    assert tools["read_file"].run({"path": "/etc/passwd"}).is_error
    assert "a.txt:2: world" in tools["grep"].run({"pattern": "wor"}).content
    assert tools["write_file"].requires_approval and not tools["write_file"].read_only
    assert "write_file" not in {t.name for t in make_file_tools(tmp_path, allow_write=False)}
