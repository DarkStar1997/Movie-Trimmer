"""SAM 3.1 segmenter: token-row selection and mask decode (CPU-only)."""

import numpy as np
import pytest

from scanner.sam3 import person_row, union_person_mask


class TestPersonRow:
    def test_exact_person(self):
        tokens = np.zeros((3, 32), dtype=np.int64)
        assert person_row(tokens, ["ball", "person", "field"]) == 1

    def test_containing_person(self):
        tokens = np.zeros((2, 32), dtype=np.int64)
        assert person_row(tokens, ["cricket player", "ball"]) == 0

    def test_exact_wins_over_containing(self):
        tokens = np.zeros((3, 32), dtype=np.int64)
        assert person_row(tokens, ["player", "person", "ball"]) == 1

    def test_no_person_row_raises(self):
        with pytest.raises(ValueError, match="no person row"):
            person_row(np.zeros((1, 32), dtype=np.int64), ["ball"])


def _fake_outs(torch, logits, presence, masks):
    """outs dict + out_names mimicking TrtEngine.run() at K=1."""
    return ({"pred_logits": torch.tensor(logits).reshape(1, 4, 1),
             "presence": torch.tensor(presence).reshape(1, 1),
             "pred_masks": torch.tensor(masks).reshape(1, 4, 4, 4),
             "pred_boxes": torch.zeros(1, 4, 4)},
            ["pred_logits", "pred_boxes", "pred_masks", "presence"])


class TestUnionPersonMask:
    def test_below_threshold_is_none(self):
        torch = pytest.importorskip("torch")
        outs, names = _fake_outs(torch, [0.0] * 4, [0.0],
                                 np.zeros((4, 4, 4)))
        assert union_person_mask(outs, names, 0.5, 8, 8, 16) is None

    def test_selected_queries_union(self):
        torch = pytest.importorskip("torch")
        # query 0 fires top-left quadrant, query 2 bottom-right
        m = np.zeros((4, 4, 4), dtype=np.float32)
        m[0, :2, :2] = 20.0     # sigmoid ~1 -> >0.5 after bilinear
        m[2, 2:, 2:] = 20.0
        # probs: q0 and q2 pass (logit 5 * presence 5), q1/q3 fail
        outs, names = _fake_outs(torch, [5.0, -9.0, 5.0, -9.0],
                                 [5.0], m)
        mask = union_person_mask(outs, names, 0.5, 8, 8, 16)
        assert isinstance(mask, np.ndarray) and mask.dtype == bool
        assert mask.shape == (8, 8)
        assert mask[:4, :4].any() and mask[4:, 4:].any()
        assert not mask[4:, :4].any()   # q1/q3 must not leak in

    def test_max_queries_caps_selection(self):
        torch = pytest.importorskip("torch")
        m = np.full((4, 4, 4), 20.0, dtype=np.float32)
        outs, names = _fake_outs(torch, [5.0] * 4, [5.0], m)
        # cap 1: only the first passing query contributes
        mask = union_person_mask(outs, names, 0.5, 8, 8, 1)
        assert mask is not None


class TestConfig:
    def test_default_off(self):
        from scanner import config
        assert config.MASK_SEGMENTER == "rfdetr"
        assert config.SAM3_THRESH_PERSON == 0.5
        assert (config.SAM3_KMIN, config.SAM3_KOPT, config.SAM3_KMAX) == \
            (1, 3, 8)
        assert config.SAM3_WORKSPACE == 20 << 30

    def test_tokens_npz_committed(self):
        import os
        from scanner import config
        p = os.path.join(os.path.dirname(config.__file__),
                         "sam3_tokens_person.npz")
        d = np.load(p, allow_pickle=True)
        row = d["text_tokens"][0]
        assert list(d["prompts"]) == ["person"]
        assert row[0] == 49406          # CLIP BOS
        assert row[1] == 2533           # 'person'
        assert row[2] == 49407          # CLIP EOS
