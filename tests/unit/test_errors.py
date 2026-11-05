import inspect

from grounded_rag import errors


def all_error_classes() -> list[type[errors.GroundedRagError]]:
    return [
        obj
        for _, obj in inspect.getmembers(errors, inspect.isclass)
        if issubclass(obj, errors.GroundedRagError)
    ]


def test_every_error_has_a_distinct_code() -> None:
    codes = [cls.code for cls in all_error_classes()]
    assert len(codes) == len(set(codes))


def test_error_keeps_message_and_details() -> None:
    error = errors.InvalidRequestError("bad top_k", details={"field": "top_k"})
    assert str(error) == "bad top_k"
    assert error.details == {"field": "top_k"}


def test_authentication_errors_share_a_parent() -> None:
    assert issubclass(errors.MissingApiKeyError, errors.AuthenticationError)
    assert issubclass(errors.InvalidApiKeyError, errors.AuthenticationError)
