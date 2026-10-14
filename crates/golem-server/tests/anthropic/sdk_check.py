"""Official Anthropic SDK against `golem serve` on tiny-moe: text, stream, tools."""
import sys

import anthropic

client = anthropic.Anthropic(base_url=sys.argv[1], api_key="unused")
# SDK 1.x dropped sampling kwargs; greedy keeps tiny-moe deterministic.
GREEDY = {"extra_body": {"temperature": 0}}

m = client.messages.create(
    model="tiny-moe", max_tokens=4, **GREEDY,
    messages=[{"role": "user", "content": "hi"}],
)
assert m.type == "message" and m.stop_reason == "max_tokens", m
assert m.usage.output_tokens == 4, m.usage
print("text ok")

with client.messages.stream(
    model="tiny-moe", max_tokens=4, **GREEDY,
    messages=[{"role": "user", "content": "hi"}],
) as s:
    text = "".join(s.text_stream)
    final = s.get_final_message()
assert final.stop_reason == "max_tokens", final
assert "".join(b.text for b in final.content if b.type == "text") == text
print("stream ok")

tools = [{"name": "f", "input_schema": {
    "type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}]
with client.messages.stream(
    model="tiny-moe", max_tokens=256, **GREEDY, tools=tools,
    tool_choice={"type": "tool", "name": "f"},
    messages=[{"role": "user", "content": "call f"}],
) as s:
    final = s.get_final_message()
assert final.stop_reason == "tool_use", final
call = next(b for b in final.content if b.type == "tool_use")
assert call.name == "f" and isinstance(call.input.get("x"), int), call
m = client.messages.create(
    model="tiny-moe", max_tokens=4, **GREEDY, tools=tools,
    messages=[
        {"role": "user", "content": "call f"},
        {"role": "assistant", "content": final.content},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call.id, "content": "7"}]},
    ],
)
assert m.type == "message", m
print("tools ok")
