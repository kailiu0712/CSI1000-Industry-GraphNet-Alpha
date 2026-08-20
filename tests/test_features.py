from iagnn.features import (
    FUNDAMENTAL_FEATURE_COLS,
    IC4NET_FEATURE_COLS,
    TECHNICAL_FEATURE_COLS,
    verify_feature_parity,
)


def test_feature_set_is_the_documented_22():
    assert len(IC4NET_FEATURE_COLS) == 22
    assert len(set(IC4NET_FEATURE_COLS)) == 22


def test_technical_and_fundamental_partition_the_set():
    assert set(TECHNICAL_FEATURE_COLS) | set(FUNDAMENTAL_FEATURE_COLS) == set(IC4NET_FEATURE_COLS)
    assert not set(TECHNICAL_FEATURE_COLS) & set(FUNDAMENTAL_FEATURE_COLS)
    assert len(FUNDAMENTAL_FEATURE_COLS) == 4


def test_copy_has_not_drifted_from_the_research_framework():
    # None means the parent framework is not importable (a standalone
    # checkout), which is not a failure -- only an explicit False is.
    assert verify_feature_parity() is not False
