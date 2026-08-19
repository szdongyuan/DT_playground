import sys

import pytest

from src.utils.runtime import get_training_verbose


@pytest.mark.parametrize(
    ("is_frozen", "expected"),
    [
        (False, 1),
        (True, 0),
    ],
    ids=["source", "packaged"],
)
def test_get_training_verbose_matches_runtime(monkeypatch, is_frozen, expected):
    monkeypatch.setattr(sys, "frozen", is_frozen, raising=False)

    assert get_training_verbose() == expected


def test_get_training_verbose_defaults_to_source_mode(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)

    assert get_training_verbose() == 1
