import json
import sys
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "tool_calling"))


def test_build_tool_schemas_basic():
    from tool_loop import build_tool_schemas

    def my_tool(name: str, count: int) -> str:
        """Do something with name and count."""
        pass

    schemas = build_tool_schemas([my_tool])
    assert len(schemas) == 1
    s = schemas[0]
    assert s["name"] == "my_tool"
    assert s["description"] == "Do something with name and count."
    assert s["input_schema"]["properties"]["name"] == {"type": "string"}
    assert s["input_schema"]["properties"]["count"] == {"type": "integer"}
    assert "name" in s["input_schema"]["required"]
    assert "count" in s["input_schema"]["required"]


def test_build_tool_schemas_literal():
    from tool_loop import build_tool_schemas

    def probe(probe_type: Literal["mha", "mlp", "residual"]) -> str:
        """Train a probe."""
        pass

    schemas = build_tool_schemas([probe])
    prop = schemas[0]["input_schema"]["properties"]["probe_type"]
    assert prop == {"type": "string", "enum": ["mha", "mlp", "residual"]}


def test_build_tool_schemas_optional_not_required():
    from tool_loop import build_tool_schemas

    def tool(required: str, optional: int = 5) -> str:
        """A tool."""
        pass

    schemas = build_tool_schemas([tool])
    assert "required" in schemas[0]["input_schema"]["required"]
    assert "optional" not in schemas[0]["input_schema"]["required"]


def test_load_skill_found(tmp_path):
    import tool_loop
    tool_loop.set_skills_dir(tmp_path)
    (tmp_path / "generate_data.md").write_text("# Generate Data\nStep 1...")
    assert tool_loop.load_skill("generate_data") == "# Generate Data\nStep 1..."


def test_load_skill_not_found(tmp_path):
    import tool_loop
    tool_loop.set_skills_dir(tmp_path)
    result = tool_loop.load_skill("nonexistent")
    assert "error" in result and "nonexistent" in result


def test_run_tool_error_serialized():
    from tool_loop import _run_tool

    def bad(x: str) -> str:
        raise ValueError("something went wrong")

    result = _run_tool("bad", {"x": "hi"}, {"bad": bad})
    assert "error" in result and "something went wrong" in result


def test_run_tool_unknown_name():
    from tool_loop import _run_tool
    result = _run_tool("ghost", {}, {})
    assert "error" in result and "ghost" in result


def test_truncation():
    from tool_loop import _truncate
    assert len(_truncate("x" * 9000)) < 9000
    assert "truncated" in _truncate("x" * 9000)


def test_anthropic_loop_end_turn():
    import tool_loop

    mock_text = MagicMock()
    mock_text.type = "text"
    mock_text.text = "Analysis complete."

    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [mock_text]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        result = tool_loop.run_tool_loop(
            system_prompt="sys", user_prompt="go", tools={},
            model="claude-3-5-sonnet-20241022", max_iterations=5,
        )
    assert result == "Analysis complete."


def test_anthropic_loop_tool_call():
    import tool_loop

    call_log = []

    def my_tool(x: str) -> str:
        """Records calls."""
        call_log.append(x)
        return f"result:{x}"

    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.id = "tu_1"
    tool_use_block.name = "my_tool"
    tool_use_block.input = {"x": "hello"}

    resp1 = MagicMock()
    resp1.stop_reason = "tool_use"
    resp1.content = [tool_use_block]

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Done."

    resp2 = MagicMock()
    resp2.stop_reason = "end_turn"
    resp2.content = [text_block]

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [resp1, resp2]

    with patch("anthropic.Anthropic", return_value=mock_client):
        result = tool_loop.run_tool_loop(
            system_prompt="sys", user_prompt="go",
            tools={"my_tool": my_tool},
            model="claude-3-5-sonnet-20241022",
        )

    assert call_log == ["hello"]
    assert result == "Done."


def test_build_system_prompt():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "tool_calling"))
    from run_agent import build_system_prompt

    prompt = build_system_prompt(
        task="sycophancy",
        model_path="google/gemma-3-12b-it",
        skill_names=["generate_data", "inspect_model"],
        tool_names=["load_model", "inspect_model"],
    )
    assert "sycophancy" in prompt
    assert "google/gemma-3-12b-it" in prompt
    assert "generate_data" in prompt
    assert "load_model" in prompt
    assert "Do not ask for user input" in prompt
