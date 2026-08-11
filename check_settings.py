from semantic_cache.config import get_settings
s = get_settings()
print(repr(s.openai_base_url))
print(s.openai_api_key[:10])
