import pytest

import bilt.errors


def test_bilt_error_exists_and_is_exception() -> None:
    assert hasattr(bilt.errors, "BiltError")
    assert issubclass(bilt.errors.BiltError, Exception)


def test_all_names_are_subclasses_of_bilt_error() -> None:
    for name in bilt.errors.__all__:
        cls = getattr(bilt.errors, name)
        assert isinstance(cls, type), f"{name} is not a class"
        assert issubclass(cls, bilt.errors.BiltError), f"{name} does not subclass BiltError"


def test_direct_subclasses_of_bilt_error() -> None:
    expected_direct = {
        "BiltDefinitionError",
        "BiltAnalysisError",
        "BiltEvaluationError",
        "BiltSymbolicError",
        "BiltUnitsError",
        "BiltBackendError",
    }
    actual_direct = {
        name
        for name in dir(bilt.errors)
        if isinstance(getattr(bilt.errors, name), type)
        and bilt.errors.BiltError in getattr(bilt.errors, name).__bases__
    }
    assert actual_direct == expected_direct


@pytest.mark.parametrize(
    ("class_name", "builtin_mixin"),
    (
        ("BiltEquationReturnError", TypeError),
        ("BiltEquationSignatureError", TypeError),
        ("BiltNoActiveEvaluationError", RuntimeError),
        ("BiltTensorDomainError", ValueError),
        ("BiltSymbolicLoweringError", ValueError),
        ("BiltDomainBindingError", ValueError),
        ("BiltDomainRegistryLockedError", RuntimeError),
        ("BiltUnknownSystemAttributeError", AttributeError),
        ("BiltAtomMutationError", AttributeError),
        ("BiltTreeMutationError", AttributeError),
        ("BiltUnknownPortError", AttributeError),
        ("BiltSymbolicConversionError", TypeError),
        ("BiltInvalidDomainError", ValueError),
    ),
)
def test_builtin_mixins(class_name: str, builtin_mixin: type) -> None:
    cls = getattr(bilt.errors, class_name)
    assert builtin_mixin in cls.__bases__, (
        f"{class_name} should have {builtin_mixin.__name__} in its bases, got {cls.__bases__}"
    )
