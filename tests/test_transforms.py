"""Q4: the transforms do what section 4 says (identity, determinism, dtype and shape, monotone severity)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_base, synthetic_image

from nuisance import spec
from nuisance import transforms as T

pytestmark = needs_base  # saturation uses DermaFair's CIELAB conversion


@pytest.mark.parametrize("family", spec.ALL)
def test_severity_zero_is_identity(img, family):
    out = T.cell(img, family, 0)
    assert np.array_equal(out, img) and out is not img


@pytest.mark.parametrize("family,severity", spec.cells())
def test_dtype_and_shape(img, family, severity):
    out = T.cell(img, family, severity, "internal", 3)
    assert out.dtype == np.uint8 and out.shape == img.shape


@pytest.mark.parametrize("family", spec.ALL)
def test_monotone_in_severity(family):
    img = synthetic_image(1)
    change = [np.abs(T.cell(img, family, s, "internal", 0).astype(float) - img).mean() for s in spec.SEVERITIES]
    assert all(b >= a - 1e-9 for a, b in zip(change, change[1:])), change
    assert change[0] > 0


def test_noise_is_deterministic_and_seeded_per_image(img):
    a = T.cell(img, "E", 3, "internal", 7)
    assert np.array_equal(a, T.cell(img, "E", 3, "internal", 7))
    assert not np.array_equal(a, T.cell(img, "E", 3, "internal", 8))
    assert not np.array_equal(a, T.cell(img, "E", 3, "scin", 7))


def test_noise_changes_luma_not_chroma(img):
    out = T.cell(img, "E", 5, "internal", 0).astype(int)
    d = out - img.astype(int)
    unclipped = (out > 0).all(-1) & (out < 255).all(-1)
    assert (d[unclipped, 0] == d[unclipped, 1]).all() and (d[unclipped, 1] == d[unclipped, 2]).all()


@pytest.mark.parametrize("ev", [-1.0, -0.5, 0.5])
def test_exposure_is_a_linear_light_gain(ev):
    img = np.full((8, 8, 3), [120, 100, 90], np.uint8)
    ratio = T.srgb_to_linear(T.exposure(img, ev)) / T.srgb_to_linear(img)
    assert np.allclose(ratio, 2.0**ev, rtol=0.03)


@pytest.mark.parametrize("c", [0.3, -0.3, 1.0, -1.0])
def test_cast_keeps_grey_luminance_and_tilts_red_blue(c):
    grey = np.full((8, 8, 3), 128, np.uint8)
    lin_in, lin_out = T.srgb_to_linear(grey), T.srgb_to_linear(T.cast(grey, c))
    assert abs(float(lin_out[0, 0] @ T.LUMA) - float(lin_in[0, 0] @ T.LUMA)) < 0.005
    assert np.sign(lin_out[0, 0, 0] - lin_out[0, 0, 2]) == np.sign(c)


def test_warm_and_cool_casts_are_opposite(img):
    warm, cool = T.cell(img, "G-", 3).astype(int), T.cell(img, "G+", 3).astype(int)
    assert (warm[..., 0] - warm[..., 2]).mean() > (img[..., 0].astype(int) - img[..., 2]).mean()
    assert (cool[..., 0] - cool[..., 2]).mean() < (img[..., 0].astype(int) - img[..., 2]).mean()


def test_full_desaturation_removes_chroma_and_keeps_lightness(img):
    from dermafair.data.skin_tone import rgb2lab

    lab_in, lab_out = rgb2lab(img), rgb2lab(T.cell(img, "H-", 5))  # k = 0
    assert np.abs(lab_out[..., 1:]).max() < 1.5
    assert np.abs(lab_out[..., 0] - lab_in[..., 0]).mean() < 0.5


def test_zoom_box_geometry():
    assert T.zoom_box(0.5, 256, 256) == (64, 64, 128)
    assert T.zoom_box(0.9, 256, 256) == (13, 13, 230)


def test_training_ranges_span_the_grid():
    for f in spec.FAMILIES.values():
        assert f.train == (min(f.levels), max(f.levels)), f.id


@pytest.mark.parametrize("families", [spec.M1_FAMILIES, spec.C1_FAMILIES])
def test_view_sampler_ranges_and_reproducibility(img, families):
    small = img[::4, ::4]
    seen = set()
    for v in range(120):
        out, fid, p = T.draw_view(small, T.view_rng(0, 5, v), families)
        again, fid2, p2 = T.draw_view(small, T.view_rng(0, 5, v), families)
        assert np.array_equal(out, again) and (fid, p) == (fid2, p2)
        fam = spec.FAMILIES[fid]
        assert fid in families and fam.train[0] <= p <= fam.train[1]
        if fam.integer:
            assert float(p).is_integer()
        seen.add(fid)
    assert seen == set(families)
