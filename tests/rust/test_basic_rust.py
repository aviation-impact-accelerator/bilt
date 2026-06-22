from bilt._native import hello_from_rust


def test_hello_from_rust() -> None:
    assert hello_from_rust() == "hello from Rust"
