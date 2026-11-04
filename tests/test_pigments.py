"""Tests for the pigment table and the Kubelka-Munk maths."""

import numpy as np
import pytest

from backrun.pigments import (PIGMENT_NAMES, PIGMENTS, behaviour_table, granulation_deposit_term,
                              km_coefficients, km_composite, km_layer, ks_table, mix_ks,
                              pigment_index)
from backrun.render import PAPER_RGB, composite


def test_eight_named_pigments():
    assert PIGMENT_NAMES == (
        "quinacridone_rose", "cadmium_yellow", "french_ultramarine", "cerulean_blue",
        "burnt_umber", "hookers_green", "indian_red", "paynes_grey",
    )


@pytest.mark.parametrize("pigment", PIGMENTS, ids=lambda p: p.name)
def test_ks_reproduces_colour_pair(pigment):
    """A unit layer over white and over black gives back the two input colours."""
    k, s = pigment.ks
    assert np.all(k > 0) and np.all(s > 0)
    r, t = km_layer(k, s, 1.0)
    on_white, _ = km_composite(r, t, np.ones(3))
    on_black, _ = km_composite(r, t, np.zeros(3))
    np.testing.assert_allclose(on_white, pigment.on_white, atol=1e-4)
    np.testing.assert_allclose(on_black, pigment.on_black, atol=1e-4)


def test_ks_rejects_impossible_pairs():
    with pytest.raises(ValueError):
        km_coefficients((0.5, 0.5, 0.5), (0.6, 0.4, 0.4))


def test_empty_layer_is_invisible():
    k, s = ks_table()
    amounts = np.zeros((len(PIGMENTS), 4, 5), dtype=np.float32)
    refl = composite(amounts, None, k, s)
    np.testing.assert_allclose(refl, np.broadcast_to(PAPER_RGB, refl.shape), atol=1e-4)


def test_more_pigment_is_darker():
    k, s = ks_table()
    i = pigment_index("french_ultramarine")
    amounts = np.zeros((len(PIGMENTS), 1, 3), dtype=np.float32)
    amounts[i, 0] = [0.2, 0.6, 1.5]
    lum = composite(amounts, None, k, s).mean(axis=-1)[0]
    assert lum[0] > lum[1] > lum[2]


def test_mixing_yellow_and_blue_leans_green():
    k, s = ks_table()
    amounts = np.zeros((len(PIGMENTS), 1, 1), dtype=np.float32)
    amounts[pigment_index("cadmium_yellow")] = 0.6
    amounts[pigment_index("cerulean_blue")] = 0.6
    r, g, b = composite(amounts, None, k, s)[0, 0]
    assert g > r and g > b


def test_mix_is_linear_in_amount():
    k, s = ks_table()
    amounts = np.random.default_rng(0).uniform(0, 1, (len(PIGMENTS), 2, 2))
    k1, s1 = mix_ks(amounts, k, s)
    k2, s2 = mix_ks(2 * amounts, k, s)
    np.testing.assert_allclose(k2, 2 * k1)
    np.testing.assert_allclose(s2, 2 * s1)


def test_behaviour_numbers_in_range():
    table = behaviour_table()
    for name, values in table.items():
        assert values.shape == (len(PIGMENTS),), name
        assert np.all(values >= 0) and np.all(values <= 1), name
    assert table["granulation"][pigment_index("french_ultramarine")] > \
        table["granulation"][pigment_index("quinacridone_rose")]


def test_granulating_pigment_settles_faster_in_valleys():
    paper = np.array([[0.0, 1.0]], dtype=np.float32)
    term = granulation_deposit_term(np.array([0.0, 0.8], dtype=np.float32), paper)
    assert term[0, 0, 0] == pytest.approx(term[0, 0, 1])  # no granulation: uniform
    assert term[1, 0, 0] > term[1, 0, 1]  # valley > peak


def test_unknown_pigment():
    with pytest.raises(KeyError):
        pigment_index("unobtainium")
