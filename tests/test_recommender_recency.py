from datetime import datetime, timedelta, timezone

import pytest

from src.recommender import RECENCY_FLOOR, _recency_weight


def test_recent_rating_gets_full_weight():
    now = datetime(2026, 11, 2, tzinfo=timezone.utc)
    assert _recency_weight("2026-11-02T00:00:00+00:00", now=now) == 1.0


def test_old_rating_approaches_floor_without_disappearing():
    now = datetime(2026, 11, 2, tzinfo=timezone.utc)
    old = now - timedelta(days=730 * 20)
    assert _recency_weight(old.isoformat(), now=now) == pytest.approx(RECENCY_FLOOR)


def test_future_timestamp_is_capped_at_full_weight():
    now = datetime(2026, 11, 2, tzinfo=timezone.utc)
    future = now + timedelta(days=30)
    assert _recency_weight(future.isoformat(), now=now) == 1.0


def test_missing_or_invalid_timestamp_uses_floor():
    assert _recency_weight("") == RECENCY_FLOOR
    assert _recency_weight("not-a-date") == RECENCY_FLOOR
