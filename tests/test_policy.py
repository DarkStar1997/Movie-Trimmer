import pytest

from scanner.detector import Detection
from scanner.policy import Policy

W, H = 640, 360


def det(label, score, w=80, h=80, x=100, y=100):
    return Detection(label=label, score=score, box=(x, y, w, h))


def test_trigger_above_threshold():
    p = Policy()
    v = p.evaluate([det("FEMALE_BREAST_EXPOSED", 0.5)], W, H)
    assert v.triggered
    assert "FEMALE_BREAST_EXPOSED" in v.trigger_labels
    assert v.max_trigger_score == pytest.approx(0.5)


def test_trigger_below_threshold():
    p = Policy()
    v = p.evaluate([det("FEMALE_BREAST_EXPOSED", 0.44)], W, H)
    assert not v.triggered


def test_area_gate_ignores_tiny_boxes():
    p = Policy()
    tiny = det("FEMALE_BREAST_EXPOSED", 0.9, w=20, h=20)  # ~0.17% < 0.5%
    v = p.evaluate([tiny], W, H)
    assert not v.triggered


def test_review_label_not_triggering():
    p = Policy()
    v = p.evaluate([det("BELLY_EXPOSED", 0.7)], W, H)
    assert not v.triggered
    assert v.review


def test_trigger_wins_over_review():
    p = Policy()
    v = p.evaluate([det("BELLY_EXPOSED", 0.7),
                    det("BUTTOCKS_EXPOSED", 0.6)], W, H)
    assert v.triggered
    assert not v.review


def test_unrelated_labels_ignored():
    p = Policy()
    v = p.evaluate([det("FACE_MALE", 0.95), det("FEMALE_BREAST_COVERED", 0.9)],
                   W, H)
    assert not v.triggered and not v.review


def test_custom_overrides():
    p = Policy.from_overrides({
        "triggers": {"BUTTOCKS_EXPOSED": 0.3},
        "review": {},
        "min_box_area_frac": 0.0,
    })
    v = p.evaluate([det("BUTTOCKS_EXPOSED", 0.35, w=5, h=5)], W, H)
    assert v.triggered          # low threshold honored
    # default trigger labels are replaced, not merged:
    v2 = p.evaluate([det("FEMALE_BREAST_EXPOSED", 0.99)], W, H)
    assert not v2.triggered


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        Policy.from_overrides({"triggers": {"NOT_A_LABEL": 0.5}})


def test_unknown_override_key_rejected():
    with pytest.raises(ValueError):
        Policy.from_overrides({"thresholds": {}})


def test_bad_threshold_range_rejected():
    with pytest.raises(ValueError):
        Policy(triggers={"ANUS_EXPOSED": 1.5})


def test_weak_triggers_band_and_area():
    """weak_triggers: score in [min_score, trigger threshold), area-qualified,
    trigger labels only."""
    p = Policy()
    dets = [
        det("FEMALE_BREAST_EXPOSED", 0.30),           # in band -> weak
        det("FEMALE_BREAST_EXPOSED", 0.50),           # >= threshold -> not weak
        det("FEMALE_GENITALIA_EXPOSED", 0.20),        # below band -> not weak
        det("BUTTOCKS_EXPOSED", 0.40, w=20, h=20),    # in band, tiny box -> out
        det("BELLY_EXPOSED", 0.80),                   # review label -> out
    ]
    assert p.weak_triggers(dets, W, H, 0.25) == {"FEMALE_BREAST_EXPOSED": 0.30}
    # boundary: exactly min_score is in, exactly threshold is not
    dets = [det("ANUS_EXPOSED", 0.25), det("ANUS_EXPOSED", 0.45)]
    assert p.weak_triggers(dets, W, H, 0.25) == {"ANUS_EXPOSED": 0.25}
