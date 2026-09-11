import pandas as pd
import pytest

from leakprobe import advance, delay, use_column


@pytest.fixture
def facts():
    ends = pd.to_datetime(["2024-03-31", "2024-06-30"])
    return pd.DataFrame({"period_end": ends, "available_at": ends + pd.Timedelta(days=34)})


def test_delay_moves_availability_only(facts):
    out = delay(facts, "available_at", pd.Timedelta(days=30))
    assert (out["available_at"] - facts["available_at"] == pd.Timedelta(days=30)).all()
    pd.testing.assert_series_equal(out["period_end"], facts["period_end"])


def test_advance_is_the_inverse(facts):
    by = pd.Timedelta(days=7)
    pd.testing.assert_frame_equal(advance(delay(facts, "available_at", by), "available_at", by), facts)


def test_use_column_models_the_period_end_join(facts):
    out = use_column(facts, "available_at", "period_end")
    pd.testing.assert_series_equal(out["available_at"], facts["period_end"], check_names=False)


def test_input_is_never_mutated(facts):
    before = facts.copy()
    delay(facts, "available_at", pd.Timedelta(days=1))
    pd.testing.assert_frame_equal(facts, before)


def test_non_datetime_column_is_refused(facts):
    bad = facts.assign(available_at=[1, 2])
    with pytest.raises(TypeError, match="not a datetime"):
        delay(bad, "available_at", pd.Timedelta(days=1))
