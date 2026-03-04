# optiml

Python SDK for the [OptiML](https://optiml.one) AI workflow operations platform.

## Install

```bash
pip install optiml
```

## Quick start

```python
from optiml import OptiMLClient

client = OptiMLClient(api_key="sk-...", org="my-org")

# Run a workflow
result = client.run("summarize", input_text="Long article text here...")
print(result.final_output)
print(f"Cost: ${result.total_cost:.4f}")
```

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

## License

MIT
