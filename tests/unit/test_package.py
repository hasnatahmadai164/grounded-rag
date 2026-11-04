import grounded_rag


def test_package_exposes_version() -> None:
    assert grounded_rag.__version__.count(".") == 2
