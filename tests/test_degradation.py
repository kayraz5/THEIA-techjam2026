import io
import numpy as np
from PIL import Image
from src.degradation import LEVELS, EVAL_CONDITIONS, apply_transform, RandomDegradation


def _img(seed=0, size=(320, 240)):
    rng = np.random.default_rng(seed)
    # smooth gradient + texture so blur/jpeg/noise all have measurable effects
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    base = np.stack([xx * 255 / size[0], yy * 255 / size[1], (xx + yy) % 256], -1)
    tex = rng.integers(0, 60, size=(size[1], size[0], 3))
    return Image.fromarray(np.clip(base + tex, 0, 255).astype(np.uint8))


def _arr(im): return np.asarray(im, dtype=np.float32)
def _hf_energy(im):
    a = _arr(im).mean(-1); return np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean()


def test_every_condition_changes_pixels_except_clean():
    im = _img()
    for name, lv in EVAL_CONDITIONS:
        out = apply_transform(im, name, lv, rng_seed=1)
        if name == "clean":
            assert out is im
        elif name == "crop":
            assert out.size == (256, 192)
        else:
            assert out.size == im.size
            assert np.abs(_arr(out) - _arr(im)).mean() > 0.1, name


def test_jpeg_quality_monotonic():
    im = _img()
    errs = [np.abs(_arr(apply_transform(im, "jpeg", q)) - _arr(im)).mean() for q in LEVELS["jpeg"]]
    assert errs == sorted(errs), errs  # lower quality -> more error


def test_blur_reduces_high_frequency_monotonically():
    im = _img()
    e = [_hf_energy(apply_transform(im, "blur", s)) for s in LEVELS["blur"]]
    assert e[0] < _hf_energy(im) and e == sorted(e, reverse=True)


def test_resize_reduces_high_frequency():
    im = _img()
    e = [_hf_energy(apply_transform(im, "resize", f)) for f in LEVELS["resize"]]
    assert e[0] < _hf_energy(im) and e[1] < e[0]


def test_noise_std_matches():
    im = Image.fromarray(np.full((256, 256, 3), 128, np.uint8))
    for s in LEVELS["noise"]:
        d = (_arr(apply_transform(im, "noise", s, rng_seed=0)) - 128) / 255
        assert abs(d.std() - s) < 0.005, (s, d.std())


def test_color_jitter_deterministic_and_bounded():
    im = _img()
    a = apply_transform(im, "color", 0.2, rng_seed=3); b = apply_transform(im, "color", 0.2, rng_seed=3)
    assert np.array_equal(_arr(a), _arr(b))
    ratio = _arr(a).mean() / _arr(im).mean()
    assert 0.5 < ratio < 1.6


def test_random_degradation_seeded_and_applies_1_to_3():
    im = _img()
    a1, p1 = RandomDegradation(seed=7)(im); a2, p2 = RandomDegradation(seed=7)(im)
    assert p1 == p2 and np.array_equal(_arr(a1), _arr(a2))
    plans = [RandomDegradation(seed=s).sample_plan() for s in range(200)]
    assert all(1 <= len(p) <= 3 for p in plans)
    assert all(len({f for f, _ in p}) == len(p) for p in plans)  # distinct families
    assert {f for p in plans for f, _ in p} == set(LEVELS)
    assert all(lv in LEVELS[f] for p in plans for f, lv in p)


def test_prob_zero_is_identity_up_to_flip():
    im = _img()
    out, plan = RandomDegradation(prob=0.0, hflip=0.0, seed=0)(im)
    assert plan == [] and np.array_equal(_arr(out), _arr(im))
