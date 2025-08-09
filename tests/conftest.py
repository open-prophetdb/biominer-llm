import pytest


@pytest.fixture
def messages():
    messages = [
        (
            "system",
            "You are a helpful calculator. Calculate the result of the user's expression.",
        ),
        ("human", "1 + 1 = ?"),
    ]
    return messages
