"""Fixed text-format comparisons and future model-quality requests.

Parser examples are authored test data, never model-generated answers.
The quality requests are complete tasks; they are not shortened to fit a probe.
"""

from copy import deepcopy
import json


def _user(text): return {"role": "user", "content": text}


def _tool(name="add", namespace=None):
    tool = {"type": "function", "function": {"name": name, "description": "Add two integers.",
        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"], "additionalProperties": False}}}
    if namespace is not None: tool["namespace"] = namespace
    return tool


def _call(identifier="call1", name="add", arguments=None, **extra):
    return {"id": identifier, "type": "function", "function": {"name": name,
        "arguments": '{"a": 2, "b": 3}' if arguments is None else arguments}, **extra}


def encoding_cases():
    cases = []
    def add(name, messages, **options):
        cases.append({"id": name, "messages": deepcopy(messages), "options": {"thinking_mode": "chat", **options}})
    add("chat-basic", [_user("2+2?")])
    add("chat-unicode", [_user("日本語とcafé、🙂をそのまま返して。\nA\tB")])
    add("chat-structured-text", [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
    add("chat-adjacent-users", [_user("one"), _user("two"), _user("")])
    add("chat-system", [{"role": "system", "content": "Be precise."}, _user("OK?")])
    add("chat-mid-system", [_user("x"), {"role": "assistant", "content": "y"}, {"role": "system", "content": "Continue."}])
    add("chat-reminder", [_user("x"), {"role": "latest_reminder", "content": "Use seconds."}])
    add("chat-response-format", [{"role": "system", "content": "JSON", "response_format": {"type": "object", "properties": {"n": {"type": "integer"}}}}, _user("four")])
    add("assistant-prefix", [_user("x"), {"role": "assistant", "content": "The answer is ", "wo_eos": True}])
    add("empty-content", [{"role": "system", "content": None}, _user("")])
    for effort in (None, 1, 50, 100, "low", "high", "max"):
        add("thinking-effort-" + str(effort), [_user("2+2?")], thinking_mode="thinking", reasoning_effort=effort)
    history = [_user("first"), {"role": "assistant", "reasoning_content": "Earlier analysis.", "content": "one"}, _user("next")]
    add("thinking-drop-history", history, thinking_mode="thinking")
    add("thinking-keep-history", history, thinking_mode="thinking", drop_thinking=False)
    add("thinking-mid-system", history[:-1] + [{"role": "system", "content": "next"}], thinking_mode="thinking")
    add("chat-tools", [{"role": "system", "content": "", "tools": [_tool()]}, _user("Add 2 and 3.")])
    add("thinking-tools", [{"role": "system", "content": "", "tools": [_tool()]}, *history], thinking_mode="thinking")
    add("tools-namespace", [{"role": "system", "content": "", "tools": [_tool(namespace={"name": "math", "description": "Exact math."})]}, _user("2+3?")])
    roundtrip = [_user("2+3?"), {"role": "assistant", "reasoning_content": "Use add.", "content": "", "tool_calls": [_call()]},
        {"role": "tool", "tool_call_id": "call1", "content": "5"}]
    add("tool-result", roundtrip)
    add("thinking-tool-result", roundtrip, thinking_mode="thinking")
    add("tool-result-blocks", [*roundtrip[:-1], {"role": "tool", "tool_call_id": "call1", "content": [{"type": "text", "text": "5"}, {"type": "text", "text": "exact"}]}])
    add("tools-call-order", [_user("two calls"), {"role": "assistant", "content": "", "tool_calls": [_call("one"), _call("two")]},
        {"role": "tool", "tool_call_id": "two", "content": "SECOND"}, _user("between"), {"role": "tool", "tool_call_id": "one", "content": "FIRST"}])
    add("tools-call-types", [_user("test"), {"role": "assistant", "content": "", "tool_calls": [_call(name="math::add",
        arguments={"text": "雪\n\"", "n": 3, "ok": True, "empty": None, "array": [1, "x"], "object": {"v": 1}})]}])
    add("tools-double-json", [_user("test"), {"role": "assistant", "content": "", "tool_calls": [_call(arguments=json.dumps(json.dumps({"a": 2})))]}])
    add("tools-empty-arguments", [_user("test"), {"role": "assistant", "content": None, "tool_calls": [_call(arguments={})]}])
    return cases


def parsing_cases():
    eos = "<｜end▁of▁sentence｜>"
    tool = '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="math::add">\n<｜DSML｜ parameter name="a" string="false">2</｜DSML｜ parameter>\n<｜DSML｜ parameter name="text" string="true">雪\nline</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
    empty = '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="ping">\n\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
    return [{"id": "plain", "text": "4" + eos, "thinking_mode": "chat"},
        {"id": "empty", "text": eos, "thinking_mode": "chat"},
        {"id": "unicode", "text": "日本語 café 🙂\n " + eos, "thinking_mode": "chat"},
        {"id": "reasoning", "text": "2+2 is four.</think>4" + eos, "thinking_mode": "thinking"},
        {"id": "empty-reasoning", "text": "</think>4" + eos, "thinking_mode": "thinking"},
        {"id": "tool", "text": tool + eos, "thinking_mode": "chat"},
        {"id": "tool-with-content", "text": "Computing." + tool + eos, "thinking_mode": "chat"},
        {"id": "tool-no-eos", "text": tool, "thinking_mode": "chat"},
        {"id": "thinking-tool", "text": "Use a tool.</think>" + tool + eos, "thinking_mode": "thinking"},
        {"id": "empty-arguments", "text": empty + eos, "thinking_mode": "chat"},
        {"id": "multiple-tools", "text": empty[:-len('\n</｜DSML｜ calls>')] + '\n<｜DSML｜ invoke name="ping2">\n\n</｜DSML｜ invoke>\n</｜DSML｜ calls>' + eos, "thinking_mode": "chat"}]


def rejection_cases():
    cases = []
    for effort in (0, 101, True, 1.5, "unknown"):
        cases.append({"id": "effort-" + str(effort), "operation": "encode", "messages": [_user("x")],
            "options": {"thinking_mode": "thinking", "reasoning_effort": effort}})
    cases.append({"id": "image-token", "operation": "encode", "messages": [_user("<｜deepseek_image｜>")], "options": {"thinking_mode": "chat"}})
    cases.append({"id": "namespace-conflict", "operation": "encode", "messages": [_user("x"), {"role": "assistant", "content": "",
        "tool_calls": [_call(name="one::add", namespace="two")]}], "options": {"thinking_mode": "chat"}})
    for name, text, mode in [("missing-eos", "4", "chat"), ("missing-think-end", "analysis", "thinking"),
            ("trailing-content", "4<｜end▁of▁sentence｜>x", "chat"), ("unexpected-thinking", "<think>x<｜end▁of▁sentence｜>", "chat"),
            ("legacy-dsml", '\n\n<｜DSML｜tool_calls>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>', "chat")]:
        cases.append({"id": name, "operation": "parse", "text": text, "thinking_mode": mode})
    duplicate = parsing_cases()[5]["text"].replace('<｜DSML｜ parameter name="text"', '<｜DSML｜ parameter name="a"')
    cases.append({"id": "duplicate-parameter", "operation": "parse", "text": duplicate, "thinking_mode": "chat"})
    return cases


def quality_cases():
    """Fixed small tasks with explicit scores; no external tools/code execute."""
    return [{"id": "reasoning-integer", "category": "reasoning", "thinking_mode": "thinking", "reasoning_effort": None,
        "messages": [_user("A box has 3 red balls and 2 blue balls. Two red balls are removed. How many balls remain? Give only the integer as your final answer.")],
        "max_new_tokens": 128, "scoring": {"kind": "final_text_exact", "expected": "3"}},
        {"id": "coding-python-expression", "category": "coding", "thinking_mode": "chat", "reasoning_effort": None,
        "messages": [_user("Write one Python expression that returns the last element of a nonempty list named xs. Return only the expression, without Markdown.")],
        "max_new_tokens": 32, "scoring": {"kind": "python_expression_cases", "variable": "xs", "cases": [
            {"input": [2, 5, 9], "expected": 9}, {"input": [7], "expected": 7}, {"input": [-3, 0], "expected": 0}]}},
        {"id": "tool-add", "category": "tool_use", "thinking_mode": "chat", "reasoning_effort": None,
        "messages": [{"role": "system", "content": "Use the add tool for arithmetic.", "tools": [_tool()]}, _user("Use add to calculate 2 + 3.")],
        "max_new_tokens": 96, "scoring": {"kind": "one_tool_call_exact", "name": "add", "arguments": {"a": 2, "b": 3}}}]
