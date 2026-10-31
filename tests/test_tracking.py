"""Stage A tracking tests: clustering, bridging semantics, hold lifecycle.

All synthetic, numpy-only (host venv has no torch by design).
"""

from scanner.detector import Detection
from scanner.tracking import (EXPLICIT_COUNTERPARTS, TrackBridge,
                              cluster_persons)

EXPL = {"FEMALE_GENITALIA_EXPOSED", "FEMALE_BREAST_EXPOSED",
        "MALE_GENITALIA_EXPOSED", "ANUS_EXPOSED", "BUTTOCKS_EXPOSED"}


def det(label, score, box):
    return Detection(label=label, score=score, box=box)


def person(centre_x, centre_y, labels_scores, w=60, h=40):
    """A vertical stack of part boxes = one person."""
    parts = []
    y = centre_y
    for label, score in labels_scores:
        parts.append(det(label, score, (centre_x - w // 2, y, w, h)))
        y += h + 5
    return parts


# --- clustering ----------------------------------------------------------------

def test_clustering_stacks_one_person():
    parts = person(100, 100, [("FACE_FEMALE", 0.8), ("FEMALE_BREAST_EXPOSED", 0.6),
                              ("BELLY_COVERED", 0.7)])
    clusters = cluster_persons(parts)
    assert len(clusters) == 1
    assert clusters[0].labels == {"FACE_FEMALE", "FEMALE_BREAST_EXPOSED",
                                  "BELLY_COVERED"}


def test_clustering_separates_side_by_side_people():
    a = person(100, 100, [("FACE_FEMALE", 0.8), ("FEMALE_BREAST_EXPOSED", 0.6)])
    b = person(400, 100, [("FACE_MALE", 0.8), ("MALE_BREAST_EXPOSED", 0.5)])
    clusters = cluster_persons(a + b)
    assert len(clusters) == 2
    labels = sorted(c.labels for c in clusters)
    assert labels[0] & {"FACE_FEMALE", "FEMALE_BREAST_EXPOSED"}
    assert labels[1] & {"FACE_MALE", "MALE_BREAST_EXPOSED"}


def test_clustering_empty():
    assert cluster_persons([]) == []


# --- explicit-hold bridging ------------------------------------------------------

def test_bridge_survives_face_only_flicker():
    """Explicit at t0, then only face detections: scene stays flagged."""
    br = TrackBridge()
    r0 = br.step(0.0, cluster_persons(
        person(100, 100, [("FACE_FEMALE", 0.8),
                          ("FEMALE_BREAST_EXPOSED", 0.6)])), EXPL)
    r1 = br.step(0.25, cluster_persons(
        person(100, 100, [("FACE_FEMALE", 0.8)])), EXPL)
    r2 = br.step(1.5, cluster_persons(
        person(100, 100, [("FACE_FEMALE", 0.8)])), EXPL)
    assert not r0["bridged"]          # spawn frame: hits=1 < min_hits
    assert r1["bridged"] and r2["bridged"]


def test_bridge_requires_track_confidence():
    """A one-frame explicit blip that never re-matches must not bridge."""
    br = TrackBridge()
    br.step(0.0, cluster_persons(
        person(100, 100, [("FEMALE_BREAST_EXPOSED", 0.6)])), EXPL)
    # person gone entirely
    r = br.step(0.25, [], EXPL)
    assert not r["bridged"]           # hits=1 track: no hold


def test_hold_expires_after_window():
    br = TrackBridge(explicit_hold_s=8.0)
    br.step(0.0, cluster_persons(
        person(100, 100, [("FACE_FEMALE", 0.8),
                          ("FEMALE_BREAST_EXPOSED", 0.6)])), EXPL)
    br.step(0.25, cluster_persons(person(100, 100, [("FACE_FEMALE", 0.8)])),
            EXPL)
    assert br.step(7.0, cluster_persons(
        person(100, 100, [("FACE_FEMALE", 0.8)])), EXPL)["bridged"]
    assert not br.step(8.6, cluster_persons(
        person(100, 100, [("FACE_FEMALE", 0.8)])), EXPL)["bridged"]


def test_coasting_track_bridges_then_retires():
    """Detector loses the person entirely: coast up to max_lost_samples."""
    br = TrackBridge(dt=0.25, max_lost_samples=3)
    br.step(0.0, cluster_persons(
        person(100, 100, [("FACE_FEMALE", 0.8),
                          ("FEMALE_BREAST_EXPOSED", 0.6)])), EXPL)
    br.step(0.25, cluster_persons(person(100, 100, [("FACE_FEMALE", 0.8)])),
            EXPL)
    assert br.step(0.5, [], EXPL)["bridged"]     # miss 1
    assert br.step(0.75, [], EXPL)["bridged"]    # miss 2
    assert br.step(1.0, [], EXPL)["bridged"]     # miss 3 (still coasting)
    assert not br.step(1.25, [], EXPL)["bridged"]  # miss 4: retired


def test_subthreshold_explicit_refreshes_hold():
    """Explicit at 0.25 (below a 0.30 trigger, above refresh floor) keeps
    the hold alive; expiry is measured from the LAST such refresh."""
    br = TrackBridge(explicit_hold_s=8.0, refresh_floor=0.15)
    br.step(0.0, cluster_persons(
        person(100, 100, [("FACE_FEMALE", 0.8),
                          ("FEMALE_BREAST_EXPOSED", 0.55)])), EXPL)
    # t=5: weak explicit hit refreshes last_explicit_t to 5
    br.step(5.0, cluster_persons(
        person(100, 100, [("FACE_FEMALE", 0.8),
                          ("FEMALE_BREAST_EXPOSED", 0.25)])), EXPL)
    assert br.step(11.0, cluster_persons(
        person(100, 100, [("FACE_FEMALE", 0.8)])), EXPL)["bridged"]  # 6s later
    assert not br.step(13.5, cluster_persons(
        person(100, 100, [("FACE_FEMALE", 0.8)])), EXPL)["bridged"]  # 8.5s


def test_reclothed_counterpart_clears_hold():
    br = TrackBridge()
    br.step(0.0, cluster_persons(
        person(100, 100, [("FACE_FEMALE", 0.8),
                          ("FEMALE_BREAST_EXPOSED", 0.6)])), EXPL)
    br.step(0.25, cluster_persons(person(100, 100, [("FACE_FEMALE", 0.8)])),
            EXPL)
    r = br.step(0.5, cluster_persons(
        person(100, 100, [("FACE_FEMALE", 0.8),
                          ("FEMALE_BREAST_COVERED", 0.8)])), EXPL)
    assert not r["bridged"]


def test_counterparts_table_covers_all_explicit_labels():
    for label in EXPL:
        assert label in EXPLICIT_COUNTERPARTS


def test_no_false_bridge_without_explicit_history():
    """Face-only person, never explicit: never bridged."""
    br = TrackBridge()
    for t in (0.0, 0.25, 0.5, 2.0):
        r = br.step(t, cluster_persons(
            person(100, 100, [("FACE_FEMALE", 0.8)])), EXPL)
        assert not r["bridged"]
