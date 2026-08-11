from semantic_cache.api.auth import admin_keys_match


def test_admin_key_comparison() -> None:
    assert admin_keys_match("secret", "secret") is True
    assert admin_keys_match("secret", "other") is False
    assert admin_keys_match("", "secret") is False
