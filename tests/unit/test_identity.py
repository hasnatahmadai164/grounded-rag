from grounded_rag.identity import generate_api_key, hash_api_key


def test_generated_keys_are_unique() -> None:
    assert generate_api_key().secret != generate_api_key().secret


def test_stored_hash_matches_the_hash_of_the_secret() -> None:
    key = generate_api_key()
    assert key.key_hash == hash_api_key(key.secret)


def test_hash_does_not_contain_the_secret() -> None:
    key = generate_api_key()
    assert key.secret not in key.key_hash


def test_prefix_identifies_a_key_without_revealing_it() -> None:
    key = generate_api_key()
    assert key.key_prefix.startswith("grk_")
    assert len(key.key_prefix) < len(key.secret) // 4
