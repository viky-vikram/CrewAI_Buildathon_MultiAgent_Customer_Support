"""UI helper functions that carry logic worth guarding."""

import time

from app import NAV_KEYS, completed_label


def test_completed_label_just_now():
    assert completed_label(time.time()) == "Completed just now"
    assert completed_label(time.time() - 59) == "Completed just now"


def test_completed_label_one_minute():
    assert completed_label(time.time() - 61) == "Completed 1 min ago"


def test_completed_label_many_minutes():
    assert completed_label(time.time() - 60 * 42 - 1) == "Completed 42 min ago"


def test_nav_keys_cover_all_views():
    assert set(NAV_KEYS) == {"new_query", "history", "about"}
