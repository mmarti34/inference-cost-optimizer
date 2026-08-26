# optiml

Python SDK for [OptiML](https://optiml.one).

## Install

> **Not on PyPI yet.** There is no publish pipeline for this package in either
> repo, so `pip install optiml` does **not** install this SDK — do not rely on
> it. Install from source until a release job exists:

```bash
pip install "git+https://github.com/<your-org>/inference-cost-optimizer#subdirectory=sdk/python"
# or, from a checkout:
pip install -e sdk/python
```

## Quick start — direct inference (no workflow required)

The fastest way to get your existing application onto OptiML is to change one
line. Point your OpenAI client at OptiML and keep everything else:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.optiml.one/v1",
    api_key="<your_optiml_service_key>",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Your call runs against OpenAI with **your own provider key**. Streaming, tool
calling, `response_format` and `usage` all work as before. OptiML observes cost,
latency, workload and strategy — no OptiML-specific code, no workflow, no
migration.

You do not need this SDK for that. It is here when you want typed helpers:

```python
from optiml import OptiMLClient

client = OptiMLClient(api_key="sk-...")           # no org needed for direct inference

reply = client.chat("gpt-4o", [{"role": "user", "content": "Hello"}])
print(reply["choices"][0]["message"]["content"])
print(reply["optiml"]["cost_usd"], reply["optiml"]["workload"])
```

### Which mode am I in?

`model` decides, and nothing else does:

| `model` | mode |
|---|---|
| `gpt-4o`, `claude-sonnet-4-5`, `meta-llama/Llama-3.3-70B-Instruct-Turbo` | direct inference to that provider |
| `optiml/<endpoint_slug>` | an OptiML deployed workflow |

A bare model id is **always** direct inference. It never resolves to a workflow,
so a production request cannot silently become someone's Studio workflow because
a string happened to collide. The `optiml/` namespace is reserved.

### Telling OptiML about a request (all optional)

```python
client.chat(
    "gpt-4o",
    messages,
    workload="support-refund",      # your own name for this workload
    user_id="user_123",             # your end user
    conversation_id="conv_456",
    experiment_tags=["prompt-v3"],
    temperature=0.2,
)
```

With the raw OpenAI client, send the same thing as `metadata={"optiml": {...}}`
or as `X-OptiML-Workload` / `X-OptiML-User-Id` / `X-OptiML-Conversation-Id` /
`X-OptiML-Experiment-Tags` headers (some SDKs reject unknown body fields).

**None of it is required.** Without it OptiML identifies the workload
structurally, from the model, system prompt, tool signature and response format.

### Streaming

```python
for chunk in client.chat_stream("gpt-4o", [{"role": "user", "content": "hi"}]):
    if "error" in chunk:
        raise RuntimeError(chunk["error"]["message"])
    print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)
```

### Tool calling

Tools are forwarded to the provider and `tool_calls` come back to **you** —
OptiML does not execute your tools:

```python
reply = client.chat("gpt-4o", messages, tools=my_tools, tool_choice="auto")
for call in reply["choices"][0]["message"].get("tool_calls", []):
    ...  # your app runs the tool, as it always did
```

## Deployed workflows

```python
from optiml import OptiMLClient

client = OptiMLClient(api_key="sk-...", org="my-org")

result = client.run("summarize", input_text="Long article text here...")
print(result.final_output)
print(f"Cost: ${result.total_cost:.4f}")
```

You can also call a deployment through the OpenAI-compatible surface:

```python
client.chat("optiml/summarize", [{"role": "user", "content": "Long article..."}])
```

`temperature`, `max_tokens`, `response_format`, `stop`, `seed` and `tools` are
**refused** on `optiml/<slug>`, not silently ignored: a workflow's execution
parameters come from its graph. Set them on the workflow's model node, or use a
provider model id where they pass through to the provider.

## Run with variables

```python
result = client.run("extract", variables={
    "document": "Invoice #1234 for $500 from Acme Corp...",
    "schema": '{"invoice_number": "string", "amount": "number", "vendor": "string"}',
})
print(result.final_output)  # JSON string
```

## Streaming

```python
for event in client.stream("chat", input_text="Explain quantum computing"):
    if event.event == "token":
        print(event.data.get("delta", ""), end="", flush=True)
print()  # newline after stream completes
```

## Multi-turn conversations

```python
from uuid import uuid4

conv_id = str(uuid4())

r1 = client.run("chat", input_text="Hi! What's the weather?", conversation_id=conv_id)
print(r1.final_output)

r2 = client.run("chat", input_text="What about tomorrow?", conversation_id=conv_id)
print(r2.final_output)
```

## Feedback

Submit custom metrics for a previous request to power auto-grading and A/B tests:

```python
result = client.run("chat", input_text="Explain photosynthesis")
client.feedback("chat", result.request_id, {"helpfulness": 5, "accuracy": 4})
```

## Pin a version

```python
# Always use deployment version 3
result = client.run("summarize", input_text="...", version=3)
```

## Error handling

```python
from optiml import OptiMLClient, AuthenticationError, RateLimitError

client = OptiMLClient(api_key="sk-...", org="my-org")

try:
    result = client.run("chat", input_text="Hello")
except AuthenticationError:
    print("Bad API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
```

## Listing what you can call

```python
for model in client.models():
    print(model["id"], model["optiml_mode"])   # "direct" or "workflow"
```

## License

MIT
