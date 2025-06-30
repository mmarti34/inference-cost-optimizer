# utils/pricing.py

PRICING = {
    "openai": {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-32k": {"input": 0.06, "output": 0.12},
        "gpt-4.5": {"input": 0.075, "output": 0.15},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        # Add legacy if needed:
        "text-davinci-003": {"input": 0.02, "output": 0.02},
    },
    "anthropic": {
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-2": {"input": 0.008, "output": 0.024},
    },
    "mistral": {
        "mistral-small": {"input": 0.20, "output": 0.60},  # per 1M tokens :contentReference[oaicite:1]{index=1}
        "mistral-medium": {"input": 2.70, "output": 8.10},  # per 1M tokens :contentReference[oaicite:2]{index=2}
        "mistral-large": {"input": 4.00, "output": 12.00}, # per 1M tokens :contentReference[oaicite:3]{index=3}
        "open-mistral-7b": {"input": 0.25, "output": 0.25}, # per 1M tokens :contentReference[oaicite:4]{index=4}
        "open-mixtral-8x7b": {"input": 0.70, "output": 0.70}, # per 1M tokens :contentReference[oaicite:5]{index=5}
        "mistral-3.1-small": {"input": 0.10, "output": 0.30}, # per 1M tokens :contentReference[oaicite:6]{index=6}
    },
    "cohere": {
        "command-r": {"input": 0.0005, "output": 0.0015},
        "command-r+": {"input": 0.002, "output": 0.006},
    },
    "gemini": {
        "gemini-1.0-pro": {"input": 0.000125, "output": 0.000375},
        "gemini-1.5-pro": {"input": 0.000125, "output": 0.000375},
        "gemini-1.5-flash": {"input": 0.0000625, "output": 0.000125},
    }
}

def get_pricing(provider: str, model: str) -> dict[str, float]:
    p = provider.lower()
    m = model.lower()
    return PRICING.get(p, {}).get(m, {"input": 0.001, "output": 0.002})

def suggest_model(prompt: str) -> dict:
    length = len(prompt.split())
    # Define tiers for OpenAI as an example
    if length <= 50:
        model = 'gpt-3.5-turbo'
        tier = 'cheapest (suitable for short/simple prompts)'
    elif length <= 200:
        model = 'gpt-4-turbo'
        tier = 'mid-tier (for moderate prompts)'
    else:
        model = 'gpt-4o'
        tier = 'high-tier (for long/complex prompts)'
    pricing = get_pricing('openai', model)
    return {
        'provider': 'openai',
        'model': model,
        'reason': tier,
        'pricing': pricing
    }
