"""Bounded text-only DeepSeek V4.1 messages and completion parsing.

Adapted from the MIT-licensed pinned DeepSeek message format; see
third_party/DeepSeek-MESSAGES-NOTICE.md. This is an adapter, not model execution.
No context-cache, vision, classification task or tool execution is provided.
Unsupported fields fail explicitly instead of silently changing a prompt.
"""

from copy import deepcopy
import json
import re

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
SYSTEM = "<｜System｜>"
THINK, END_THINK = "<think>", "</think>"
DSML = "｜DSML｜"
CALLS = f"<{DSML} calls>"
END_CALLS = f"</{DSML} calls>"
MAX_MESSAGES, MAX_JSON_BYTES, MAX_PROMPT_CHARS = 32, 65536, 4096

TOOLS_TEMPLATE = '''## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}{tc_block_name}>" block like the following:

<{dsml_token}{tc_block_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME">
<{dsml_token}{tool_parameter_tag_name} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}{tool_parameter_tag_name}>
...
</{dsml_token}{tool_call_tag_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME2">
...
</{dsml_token}{tool_call_tag_name}>
</{dsml_token}{tc_block_name}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
'''


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _text(value):
    if value is None: return ""
    if type(value) is not str or "<｜deepseek_image｜>" in value:
        raise ValueError("Expected text without image placeholders")
    return value


def _name(name, namespace=None):
    if isinstance(namespace, dict): namespace = namespace.get("name")
    if type(name) is not str or not name or (namespace is not None and (type(namespace) is not str or not namespace)):
        raise ValueError("Expected a tool name and optional namespace")
    prefix, sep, bare = name.partition("::")
    if sep:
        if namespace not in (None, prefix) or not bare: raise ValueError("Conflicting tool namespace")
        name, namespace = bare, prefix
    if "::" in name or (namespace is not None and "::" in namespace): raise ValueError("Invalid tool namespace")
    return name if namespace is None else namespace + "::" + name


def _arguments(value):
    for _ in range(2):
        if type(value) is str:
            try: value = json.loads(value)
            except (ValueError, TypeError): break
    if type(value) is not dict: raise ValueError("Expected an object for tool arguments")
    return value


def _tools(tools):
    if type(tools) is not list or len(tools) > 16: raise ValueError("Expected bounded function tools")
    functions = []
    for tool in tools:
        if (type(tool) is not dict or set(tool) - {"type", "function", "namespace"}
                or tool.get("type", "function") != "function" or type(tool.get("function")) is not dict):
            raise ValueError("Unsupported tool definition")
        f = dict(tool["function"])
        if tool.get("namespace") is not None: f["namespace"] = tool["namespace"]
        f["name"] = _name(f["name"], f.get("namespace"))
        namespace = f.pop("namespace", None)
        if isinstance(namespace, dict) and namespace.get("description"):
            f["description"] = namespace["description"] + "\n" + (f.get("description") or "")
        functions.append(_json(f))
    return TOOLS_TEMPLATE.format(tool_schemas="\n".join(functions), dsml_token=DSML,
        tc_block_name=" calls", tool_call_tag_name=" invoke", tool_parameter_tag_name=" parameter",
        thinking_start_token=THINK, thinking_end_token=END_THINK)


def _calls(calls):
    if type(calls) is not list or not 1 <= len(calls) <= 16: raise ValueError("Expected bounded function calls")
    out = []
    for call in calls:
        if (type(call) is not dict or set(call) - {"type", "id", "function", "namespace"}
                or call.get("type", "function") != "function" or type(call.get("function")) is not dict):
            raise ValueError("Unsupported function call")
        f = call["function"]
        if set(f) - {"name", "namespace", "arguments", "id"}: raise ValueError("Unsupported function call field")
        if any("id" in obj and type(obj["id"]) is not str for obj in (call, f)):
            raise ValueError("Expected string tool call identifiers")
        namespace = call.get("namespace") or f.get("namespace")
        if namespace is not None and type(namespace) is not str: raise ValueError("Expected a string call namespace")
        name = _name(f["name"], namespace)
        args = []
        for key, value in _arguments(f["arguments"]).items():
            if type(key) is not str: raise ValueError("Expected string parameter names")
            string = "true" if type(value) is str else "false"
            text = value if type(value) is str else _json(value)
            args.append(f'<{DSML} parameter name="{key}" string="{string}">{text}</{DSML} parameter>')
        out.append(f'<{DSML} invoke name="{name}">\n' + "\n".join(args) + f"\n</{DSML} invoke>")
    return "\n\n" + CALLS + "\n" + "\n".join(out) + "\n" + END_CALLS


def _blocks(blocks):
    if type(blocks) is not list or len(blocks) > 64: raise ValueError("Expected bounded text blocks")
    out = []
    for b in blocks:
        if type(b) is not dict: raise ValueError("Expected a text block")
        if b.get("type") == "text" and not set(b) - {"type", "text"}:
            if type(b.get("text", "")) is not str: raise ValueError("Expected string block text")
            out.append({"type": "text", "text": _text(b.get("text", ""))})
        elif b.get("type") == "tool_result" and not set(b) - {"type", "content", "tool_use_id"}:
            if "tool_use_id" in b and type(b["tool_use_id"]) is not str: raise ValueError("Expected a string result identifier")
            content = b.get("content", "")
            if type(content) not in (str, list): raise ValueError("Expected text tool-result content")
            if isinstance(content, list):
                text_blocks = _blocks(content)
                if any(x["type"] != "text" for x in text_blocks): raise ValueError("Nested tool results are unsupported")
                content = "\n\n".join(x["text"] for x in text_blocks)
            out.append({"type": "tool_result", "content": _text(content), "tool_use_id": b.get("tool_use_id", "")})
        else: raise ValueError("Only text and text tool results are supported")
    return out


def _merge(messages):
    merged = []
    for msg in messages:
        role = msg["role"]
        if role == "tool":
            blocks = _blocks([{"type": "tool_result", "content": msg.get("content", ""), "tool_use_id": msg.get("tool_call_id", "")}])
            msg = {"role": "user", "content_blocks": blocks}
        elif role == "user":
            blocks = msg.get("content_blocks")
            if blocks is None:
                content = msg.get("content", "")
                blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            msg["content_blocks"] = _blocks(blocks)
        if msg["role"] == "user" and merged and merged[-1]["role"] == "user":
            merged[-1]["content_blocks"].extend(msg["content_blocks"])
        else: merged.append(msg)
    order = {}
    for msg in merged:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            order = {c.get("id") or c.get("function", {}).get("id", ""): i for i, c in enumerate(msg["tool_calls"])
                if c.get("id") or c.get("function", {}).get("id", "")}
        elif msg["role"] == "user" and order:
            results = iter(sorted((b for b in msg["content_blocks"] if b["type"] == "tool_result"),
                key=lambda b: order.get(b.get("tool_use_id", ""), 0)))
            msg["content_blocks"] = [next(results) if b["type"] == "tool_result" else b for b in msg["content_blocks"]]
    return merged


def encode_messages(messages, *, thinking_mode="chat", reasoning_effort=None, drop_thinking=True):
    """Return the original-format prompt for the explicit text-only subset.

    Always renders a whole conversation with its original BOS. Existing cache
    context, extra fields and unsupported modalities are rejected. Input objects
    are not mutated. This function neither tokenizes nor starts generation.
    """
    aliases = {"low": 50, "high": 75, "max": 100}
    effort = 75 if reasoning_effort is None else aliases.get(reasoning_effort, reasoning_effort) if type(reasoning_effort) is str else reasoning_effort
    if thinking_mode not in ("chat", "thinking") or type(drop_thinking) is not bool or type(effort) is not int or not 1 <= effort <= 100:
        raise ValueError("Expected chat/thinking mode and reasoning effort in [1,100]")
    if type(messages) is not list or not 1 <= len(messages) <= MAX_MESSAGES:
        raise ValueError("Expected a bounded nonempty conversation")
    try:
        if len(_json(messages).encode()) > MAX_JSON_BYTES: raise ValueError("Conversation exceeds the metadata bound")
    except (TypeError, RecursionError, UnicodeError) as exc: raise ValueError("Invalid conversation metadata") from exc
    allowed = {"system": {"content", "tools", "response_format"}, "user": {"content", "content_blocks"},
        "assistant": {"content", "reasoning_content", "tool_calls", "wo_eos"},
        "tool": {"content", "tool_call_id"}, "latest_reminder": {"content"}}
    for msg in messages:
        if (type(msg) is not dict or type(msg.get("role")) is not str or msg["role"] not in allowed
                or set(msg) - (allowed[msg["role"]] | {"role"})):
            raise ValueError("Unsupported message role or field")
        for name in ("content", "reasoning_content"):
            if name in msg and not isinstance(msg[name], list): _text(msg[name])
        if "wo_eos" in msg and type(msg["wo_eos"]) is not bool: raise ValueError("Expected a boolean wo_eos")
        if "reasoning_content" in msg: _text(msg["reasoning_content"])
        if "tool_call_id" in msg and type(msg["tool_call_id"]) is not str: raise ValueError("Expected a string tool result identifier")
        if msg.get("tool_calls"): _calls(msg["tool_calls"])
        if msg.get("tools"): _tools(msg["tools"])
        if isinstance(msg.get("content"), list) and msg["role"] not in ("user", "tool"):
            raise ValueError("Text content blocks are supported only for users and tools")
        if msg["role"] in ("user", "tool", "latest_reminder") and "content" in msg and type(msg["content"]) not in (str, list):
            raise ValueError("Expected explicit user/tool/reminder text")
        if msg["role"] == "latest_reminder" and type(msg.get("content")) is not str:
            raise ValueError("Expected explicit reminder text")
    # The caller may reuse the same dictionary in multiple positions. Copy
    # each message separately so merging cannot alias a later message.
    full = _merge([deepcopy(msg) for msg in messages])
    last_user = max((i for i, m in enumerate(full) if (m["role"] == "system" and i > 0) or
        (m["role"] == "user" and (not m["content_blocks"] or any(b["type"] == "text" for b in m["content_blocks"])))), default=-1)
    drop = drop_thinking and not any(m.get("tools") for m in full)
    chunks = [BOS]
    for i, msg in enumerate(full):
        role, content = msg["role"], msg.get("content") or ""
        prefix = ""
        if i == 0 and (thinking_mode == "thinking" or role == "system"): prefix += SYSTEM
        if i == 0 and thinking_mode == "thinking":
            prefix += f"Reasoning Effort: {effort} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
        if role == "system":
            prefix += (SYSTEM if i > 0 else "") + content
            if msg.get("tools"): prefix += "\n\n" + _tools(msg["tools"])
            if msg.get("response_format"):
                prefix += "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n" + _json(msg["response_format"])
        elif role == "user":
            parts = [b["text"] if b["type"] == "text" else "<tool_result>" + b["content"] + "</tool_result>" for b in msg["content_blocks"]]
            prefix += USER + ("\n\n".join(parts) if parts else content)
        elif role == "assistant":
            if thinking_mode == "thinking" and (not drop or i > last_user): prefix += (msg.get("reasoning_content") or "") + END_THINK
            prefix += content
            if msg.get("tool_calls"): prefix += _calls(msg["tool_calls"])
            if not msg.get("wo_eos", False): prefix += EOS
        else: prefix += "<｜latest_reminder｜>" + content
        next_role = full[i + 1]["role"] if i + 1 < len(full) else None
        if next_role in (None, "assistant", "latest_reminder") and (role == "user" or (role == "system" and i > 0)):
            prefix += ASSISTANT + (THINK if thinking_mode == "thinking" and (not drop or i >= last_user) else END_THINK)
        chunks.append(prefix)
    prompt = "".join(chunks)
    if len(prompt) > MAX_PROMPT_CHARS: raise ValueError("Rendered message prompt exceeds the qualified tokenizer text bound")
    return prompt


def parse_completion(text, *, thinking_mode="chat"):
    """Parse a complete text/DSML response; never execute or repair a tool call.

    As upstream, plain text requires EOS. A closed tool-call block may end
    without EOS. Call arguments retain the original ordered JSON text.
    """
    if type(text) is not str or len(text) > 16384 or thinking_mode not in ("chat", "thinking"):
        raise ValueError("Expected a bounded complete text response")
    reasoning = ""
    if thinking_mode == "thinking":
        reasoning, sep, text = text.partition(END_THINK)
        if not sep: raise ValueError("Missing original reasoning terminator")
    content, marker, tail = text.partition("\n\n" + CALLS)
    calls = []
    if not marker:
        content, end, extra = content.partition(EOS)
        if not end or extra: raise ValueError("Missing EOS or trailing response content")
    else:
        if tail.endswith(EOS): tail = tail[:-len(EOS)]
        if not tail.startswith("\n") or not tail.endswith("\n" + END_CALLS): raise ValueError("Incomplete tool-call block")
        remaining = tail[1:-len("\n" + END_CALLS)]
        invoke = re.compile(r'<｜DSML｜ invoke name="([^"]+)">\n(.*?)\n</｜DSML｜ invoke>', re.DOTALL)
        param = re.compile(r'<｜DSML｜ parameter name="(.*?)" string="(true|false)">(.*?)</｜DSML｜ parameter>', re.DOTALL)
        while remaining:
            m = invoke.match(remaining)
            if not m: raise ValueError("Invalid original invoke syntax")
            name, args = m.groups(); _name(name)
            keys, values = set(), []
            while args:
                a = param.match(args)
                if not a: raise ValueError("Invalid original parameter syntax")
                key, string, value = a.groups()
                if key in keys: raise ValueError("Duplicate tool parameter")
                keys.add(key); values.append(_json(key) + ": " + (_json(value) if string == "true" else value))
                args = args[a.end():]
                if args:
                    if not args.startswith("\n"): raise ValueError("Invalid parameter separator")
                    args = args[1:]
                    if not args: raise ValueError("Trailing parameter separator")
            namespace, sep, bare = name.partition("::")
            call = {"type": "function", "function": {"name": bare if sep else name, "arguments": "{" + ", ".join(values) + "}"}}
            if sep: call["namespace"] = namespace
            calls.append(call)
            if len(calls) > 16: raise ValueError("Too many response tool calls")
            remaining = remaining[m.end():]
            if remaining:
                if not remaining.startswith("\n"): raise ValueError("Invalid invoke separator")
                remaining = remaining[1:]
                if not remaining: raise ValueError("Trailing invoke separator")
        if not calls: raise ValueError("Empty tool-call response")
    if any(token in content or token in reasoning for token in (BOS, EOS, THINK, END_THINK, DSML)):
        raise ValueError("Unexpected special token in response content")
    return {"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": calls}
