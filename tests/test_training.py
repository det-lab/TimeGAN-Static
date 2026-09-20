"""Real, short training-run tests against the bundled small datasets
(timegan/data/stock_data.csv, energy_data.csv) and synthetic data, rather
than shape-only unit tests.

Why real training, not mocks: this codebase's actual failure modes are
things like "the graph doesn't build" (wrong arg count, undefined name,
malformed tf.concat -- all found while writing these, when this repo's
static-feature support lived in a second, entirely separate file,
timegan_static.py, since deleted and folded into timegan.py's own
embedder/recovery/generator/discriminator/train_timegan/train_timegan_timed/
load_timegan as an optional S/ori_data_static parameter) or "training
silently produces NaN/garbage" -- none of which a mocked-out training loop
would ever exercise.

Why statistical/structural assertions, not exact golden values: verified
directly (see report_training_comparison.py and the module docstring
there) that this TF1 graph-mode code is NOT bit-reproducible in this
environment even with every seed fixed and execution forced single-
threaded -- two consecutive runs of the identical script, same seed, same
config, produced different generated output. An exact-match test would be
flaky by construction, not a real regression signal. These tests instead
check what should hold true regardless of that run-to-run noise: shape,
finiteness, and that values land in the range the network's own
architecture guarantees (sigmoid-activated outputs, so roughly [0, 1]).

Small/fast by design: tiny hidden_dim, 2 iterations per phase, a handful
of events. These are meant to run in a couple seconds each, not to
produce a meaningful trained model.
"""

import numpy as np
import tensorflow as tf

from timegan.data_loading import real_data_loading
from timegan.timegan import (
    _band_texture_features,
    _band_texture_features_np,
    _band_texture_loss,
    _texture_reference,
    _windowed_log_rms_jitter,
    load_timegan,
    train_timegan,
    train_timegan_timed,
)

TINY_PARAMS = dict(hidden_dim=4, num_layer=2, batch_size=8, module="gru")
# supervisor() builds num_layers - 1 RNN cells -- num_layer=1 would pass an
# empty cell list to MultiRNNCell and fail with an unrelated-looking error
# ("Must specify at least one cell"). Found while writing this test; not a
# bug in these tests' config, a real constraint on the network itself.


def _subsample(data_name, seq_len, n_events):
    """A small, fast slice of a bundled real dataset -- real data loading
    all the way through the same real_data_loading() any actual caller
    uses, just fewer events."""
    data = real_data_loading(data_name, seq_len)
    return np.asarray(data[:n_events], dtype=np.float32)


def _assert_sane_generated_output(generated, expected_shape):
    generated = np.asarray(generated)
    assert generated.shape == expected_shape
    assert np.isfinite(generated).all(), "generated output contains NaN/Inf"
    # Every output path here ends in a sigmoid-activated layer (recovery()),
    # so values should land in roughly [0, 1] regardless of run-to-run
    # noise -- a real bound the architecture guarantees, not a tuned
    # tolerance.
    assert generated.min() >= -0.05
    assert generated.max() <= 1.05


def test_train_timegan_timed_on_stock(tmp_path):
    """The function Traces_GAN (and presumably every other real caller)
    actually uses."""
    np.random.seed(0)
    tf.compat.v1.set_random_seed(0)
    ori_data = _subsample("stock", seq_len=8, n_events=40)

    phase, info = train_timegan_timed(
        ori_data,
        dict(TINY_PARAMS, iterations=2),
        in_filename=str(tmp_path / "test_timed_stock"),
        seconds=120,
        phase=1,
        new=True,
        num_generate=5,
    )

    assert phase == 4, f"expected all phases to finish within the time budget, got phase {phase}"
    _assert_sane_generated_output(info, expected_shape=(5, 8, ori_data.shape[-1]))


def test_train_timegan_timed_on_energy_smoke(tmp_path):
    """Different dataset/dimensionality -- lighter check, just confirms
    the shape/no-NaN properties hold beyond the one dataset above."""
    np.random.seed(1)
    tf.compat.v1.set_random_seed(1)
    ori_data = _subsample("energy", seq_len=8, n_events=40)

    phase, info = train_timegan_timed(
        ori_data,
        dict(TINY_PARAMS, iterations=2),
        in_filename=str(tmp_path / "test_timed_energy"),
        seconds=120,
        phase=1,
        new=True,
        num_generate=5,
    )

    assert phase == 4
    _assert_sane_generated_output(info, expected_shape=(5, 8, ori_data.shape[-1]))


def test_train_timegan_timed_with_noise_injection(tmp_path):
    """inject_noise=True: exercises _extract_noise_bank actually finding
    real pulse-free rows and _build_timegan_graph's re-embedding path.
    Synthetic data with a deliberate mix of flat (pulse-free) and spiky
    rows -- inject_noise needs SOME pulse-free rows to build a noise bank
    from, which isn't guaranteed for the bundled stock/energy datasets."""
    np.random.seed(5)
    tf.compat.v1.set_random_seed(5)
    no, seq_len, dim = 40, 8, 1
    rng = np.random.default_rng(5)
    ori_data = (0.5 + 0.01 * rng.standard_normal((no, seq_len, dim))).astype(np.float32)
    # Every other row gets a real spike, so the rest are genuinely
    # pulse-free -- exactly what _extract_noise_bank looks for.
    for i in range(0, no, 2):
        ori_data[i, seq_len // 2, 0] = 1.0

    phase, info = train_timegan_timed(
        ori_data,
        dict(TINY_PARAMS, iterations=2, inject_noise=True),
        in_filename=str(tmp_path / "test_timed_noise_injection"),
        seconds=120,
        phase=1,
        new=True,
        num_generate=5,
    )

    assert phase == 4, f"expected all phases to finish within the time budget, got phase {phase}"
    _assert_sane_generated_output(info, expected_shape=(5, seq_len, dim))


def test_train_timegan_timed_with_normalized_g_loss_v(tmp_path):
    """normalize_g_loss_v=True: exercises the per-timestep-relative branch
    of G_loss_V (division by that timestep's own real std instead of a
    plain absolute mismatch) end to end, combined with inject_noise=True
    since that's the combination this option was added for."""
    np.random.seed(6)
    tf.compat.v1.set_random_seed(6)
    no, seq_len, dim = 40, 8, 1
    rng = np.random.default_rng(6)
    ori_data = (0.5 + 0.01 * rng.standard_normal((no, seq_len, dim))).astype(np.float32)
    for i in range(0, no, 2):
        ori_data[i, seq_len // 2, 0] = 1.0

    phase, info = train_timegan_timed(
        ori_data,
        dict(TINY_PARAMS, iterations=2, inject_noise=True, normalize_g_loss_v=True),
        in_filename=str(tmp_path / "test_timed_normalized_g_loss_v"),
        seconds=120,
        phase=1,
        new=True,
        num_generate=5,
    )

    assert phase == 4, f"expected all phases to finish within the time budget, got phase {phase}"
    _assert_sane_generated_output(info, expected_shape=(5, seq_len, dim))


def test_windowed_log_rms_jitter_matches_numpy():
    """_windowed_log_rms_jitter (the building block of G_loss_T) against a
    plain-numpy reference: RMS of first differences per window, log-scaled,
    with a remainder of (seq_len - 1) % window differences dropped."""
    rng = np.random.default_rng(0)
    seq_len, dim, window, eps = 20, 2, 6, 1e-9
    x = (0.5 + 0.01 * rng.standard_normal((7, seq_len, dim))).astype(np.float32)

    with tf.Graph().as_default():
        ph = tf.compat.v1.placeholder(tf.float32, [None, seq_len, dim])
        out = _windowed_log_rms_jitter(ph, seq_len, dim, window, eps)
        with tf.compat.v1.Session() as sess:
            got = sess.run(out, {ph: x})

    n_win = (seq_len - 1) // window  # 19 // 6 = 3, last difference dropped
    d = np.diff(x.astype(np.float64), axis=1)[:, : n_win * window, :]
    e = (d.reshape(7, n_win, window, dim) ** 2).mean(axis=2)
    want = 0.5 * np.log(e + eps)
    assert got.shape == (7, n_win, dim)
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)


def test_train_timegan_timed_with_texture_loss(tmp_path):
    """g_loss_t_weight>0: exercises G_loss_T end to end, with inject_noise
    OFF -- G_loss_T is meant to be used without noise injection, since the
    injected real noise already supplies the texture it measures."""
    np.random.seed(7)
    tf.compat.v1.set_random_seed(7)
    no, seq_len, dim = 40, 12, 1
    rng = np.random.default_rng(7)
    ori_data = (0.5 + 0.01 * rng.standard_normal((no, seq_len, dim))).astype(np.float32)
    for i in range(0, no, 2):
        ori_data[i, seq_len // 3, 0] = 1.0

    phase, info = train_timegan_timed(
        ori_data,
        dict(TINY_PARAMS, iterations=2, g_loss_t_weight=1.0, texture_window=4),
        in_filename=str(tmp_path / "test_timed_texture_loss"),
        seconds=120,
        phase=1,
        new=True,
        num_generate=5,
    )

    assert phase == 4, f"expected all phases to finish within the time budget, got phase {phase}"
    _assert_sane_generated_output(info, expected_shape=(5, seq_len, dim))


def test_train_timegan_timed_trajectory_logging(tmp_path):
    """snapshot_every / checkpoint_every / log_grad_norms, plus the always-on
    loss accounting: files land where documented with the documented
    contents, the history records carry the new fields, and a saved
    checkpoint can actually be restored (load_timegan on the p3_end one)."""
    import json
    import os

    np.random.seed(8)
    tf.compat.v1.set_random_seed(8)
    no, seq_len, dim = 40, 12, 1
    rng = np.random.default_rng(8)
    ori_data = (0.5 + 0.01 * rng.standard_normal((no, seq_len, dim))).astype(np.float32)
    for i in range(0, no, 2):
        ori_data[i, seq_len // 3, 0] = 1.0

    out = str(tmp_path / "test_trajectory")
    params = dict(
        TINY_PARAMS, iterations=2, g_loss_t_weight=1.0, texture_window=4,
        snapshot_every=1, snapshot_n=5, checkpoint_every=1, log_grad_norms=True,
    )
    phase, info = train_timegan_timed(
        ori_data, dict(params), in_filename=out, seconds=120, phase=1, new=True, num_generate=5
    )
    assert phase == 4
    _assert_sane_generated_output(info, expected_shape=(5, seq_len, dim))

    # fixed-noise sample snapshots: iterations 0 and 1, plus the end-of-phase-3 one (iteration == iterations)
    for it in (0, 1, 2):
        snap = np.load(f"{out}_samples/phase3_iter{it:05d}.npz")
        assert snap["samples"].shape == (5, seq_len, dim)
        assert np.isfinite(snap["samples"]).all()
        assert int(snap["iteration"]) == it
        assert {"elapsed", "min_val", "max_val"} <= set(snap.files)

    # full checkpoints under distinct names (index file exists; no .meta graph written)
    for tag in ("p1_end", "p2_end", "p3_iter00001", "p3_end"):
        assert os.path.exists(f"{out}_ckpt_{tag}.index"), tag
        assert not os.path.exists(f"{out}_ckpt_{tag}.meta"), tag

    # history: phase-3 record at iteration 0 carries the new accounting + gradient-norm fields
    recs = [json.loads(line) for line in open(out + "_history.jsonl")]
    p3 = [r for r in recs if r["phase"] == 3]
    assert p3
    for key in ("g_loss_u_e", "weighted_s", "weighted_v", "weighted_t", "d_loss_real", "d_loss_fake",
                "d_loss_fake_e", "d_update_frac", "e_loss_t0",
                "grad_norm_U", "grad_norm_U_e", "grad_norm_S", "grad_norm_V", "grad_norm_T", "grad_norm_total"):
        assert key in p3[0], key
        assert np.isfinite(p3[0][key]), key
    assert 0.0 <= p3[0]["d_update_frac"] <= 1.0
    # G_loss_S is computed from the supervisor run on REAL embeddings, so it reaches only the supervisor's
    # variables (not the generator's) -- its norm over generator+supervisor variables is finite and >= 0
    assert p3[0]["grad_norm_S"] >= 0.0

    # a saved checkpoint really restores
    restored = load_timegan(ori_data, dict(params), filename=f"{out}_ckpt_p3_end")
    _assert_sane_generated_output(restored, expected_shape=(no, seq_len, dim))


def test_band_texture_features_match_numpy():
    """_band_texture_features (TF) against its numpy twin: log band power per window, low/mid/high bands."""
    rng = np.random.default_rng(1)
    seq_len, dim, window = 64, 2, 16
    x = (0.3 + 0.02 * rng.standard_normal((6, seq_len, dim))).astype(np.float32)
    with tf.Graph().as_default():
        ph = tf.compat.v1.placeholder(tf.float32, [None, seq_len, dim])
        out = _band_texture_features(ph, seq_len, dim, window)
        with tf.compat.v1.Session() as sess:
            got = sess.run(out, {ph: x})
    want = _band_texture_features_np(x, window)
    assert got.shape == (6, 4, dim, 3)
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-2)


def test_band_texture_loss_separates_white_noise_from_cheap_imitations():
    """The reason G_loss_T has a "bands" kind: white noise at the true amplitude must score low, while smooth,
    slowly wandering and period-2 traces -- all of which a first-difference measure can be made to accept -- must
    score high, with pulse rows mixed into the batch (they must not be able to mask or drive the result)."""
    rng = np.random.default_rng(3)
    seq_len, dim, window, sig = 64, 1, 16, 0.002
    n = 4000
    real = 0.3 + sig * rng.standard_normal((n, seq_len, dim))
    ref = _texture_reference(real, window, noise_amp=np.array([0.05]))

    def batch(kind, n_rows=48):
        t = np.arange(seq_len)
        if kind == "white":
            x = 0.3 + sig * rng.standard_normal((n_rows, seq_len))
        elif kind == "smooth":
            x = 0.3 + 0.05 * sig * rng.standard_normal((n_rows, seq_len))
        elif kind == "wander":  # slow, big, with the SAME mean |first difference| as the white noise
            k = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2)
            z = np.array([np.convolve(r, k, mode="same") for r in rng.standard_normal((n_rows, seq_len))])
            z *= (sig * 2 / np.sqrt(np.pi)) / np.abs(np.diff(z, axis=1)).mean()
            x = 0.3 + z
        elif kind == "ringing":  # period-2 alternation with the same mean |first difference|
            x = 0.3 + (sig * 2 / np.sqrt(np.pi)) / 2 * np.tile((-1.0) ** t, (n_rows, 1))
        pulses = 0.3 + 0.6 * np.exp(-np.abs(t - 20) / 4.0)[None, :] + sig * rng.standard_normal((16, seq_len))
        return np.concatenate([x, pulses], axis=0)[:, :, None].astype(np.float32)

    with tf.Graph().as_default():
        ph = tf.compat.v1.placeholder(tf.float32, [None, seq_len, dim])
        loss = _band_texture_loss(ph, seq_len, dim, window, ref, np.array([0.05]), min_rows=3.0)
        with tf.compat.v1.Session() as sess:
            got = {k: float(sess.run(loss, {ph: batch(k)})) for k in ("white", "smooth", "wander", "ringing")}
            pulses_only = float(sess.run(loss, {ph: batch("white")[-16:]}))
    assert got["white"] < 0.6, got
    for k in ("smooth", "wander", "ringing"):
        assert got[k] > 3 * got["white"] and got[k] > 1.5, (k, got)
    assert pulses_only < 0.1, pulses_only  # no noise-like rows in the batch -> the loss switches itself off


def test_train_timegan_timed_with_band_texture_loss(tmp_path):
    """texture_kind="bands" end to end: reference table built from the training data's noise-only rows, the loss
    wired into phase 3 with inject_noise off (the intended use)."""
    np.random.seed(9)
    tf.compat.v1.set_random_seed(9)
    no, seq_len, dim = 60, 32, 1
    rng = np.random.default_rng(9)
    ori_data = (0.5 + 0.01 * rng.standard_normal((no, seq_len, dim))).astype(np.float32)
    for i in range(0, no, 3):
        ori_data[i, seq_len // 3, 0] = 1.0

    phase, info = train_timegan_timed(
        ori_data,
        dict(TINY_PARAMS, iterations=2, g_loss_t_weight=1.0, texture_kind="bands", texture_band_window=16,
             texture_noise_amp=0.2),
        in_filename=str(tmp_path / "test_timed_band_texture"),
        seconds=120,
        phase=1,
        new=True,
        num_generate=5,
    )

    assert phase == 4, f"expected all phases to finish within the time budget, got phase {phase}"
    _assert_sane_generated_output(info, expected_shape=(5, seq_len, dim))


def test_train_timegan_base_smoke(tmp_path):
    """timegan.py's train_timegan -- a thin wrapper around
    train_timegan_timed (was previously a full standalone copy of its
    graph-building/training-loop code -- the same "same fix needs applying
    in more than one place" risk that let timegan_static.py's copy of this
    code drift until it no longer even ran). Not used by Traces_GAN
    directly, but this is the test that actually exercises the wrapper end
    to end."""
    np.random.seed(2)
    tf.compat.v1.set_random_seed(2)
    ori_data = _subsample("stock", seq_len=8, n_events=24)

    generated = train_timegan(
        ori_data,
        dict(TINY_PARAMS, iterations=2),
        filename=str(tmp_path / "test_base_stock"),
    )

    _assert_sane_generated_output(generated, expected_shape=(24, 8, ori_data.shape[-1]))


def test_train_timegan_with_static_features_on_synthetic(tmp_path):
    """Static-feature support, folded into timegan.py's own train_timegan
    (via the optional ori_data_static param) from what used to be a
    second, entirely separate file (timegan_static.py) that could not
    previously be imported (unrelated `utils` package shadowing its
    intended local import) or called (wrong argument counts throughout;
    unpack_parameters never updated for the extra static-feature params; a
    bare undefined `sigmoid`; a typo'd TF method name; malformed
    tf.concat calls) -- all fixed, then merged, to get to this point.
    Static features are genuinely embedded/discriminated (HS/ES/XS_tilde/
    XS_hat/YS_* in timegan.py's component functions are real graph nodes),
    but intentionally don't participate in the loss yet -- see
    train_timegan_timed's docstring.

    Synthetic data, not a bundled dataset: none of the bundled data ships
    a static-feature column, so this constructs its own small windowed
    array plus a random per-sequence label (agreed as the stand-in for
    "real" static features for this first test).
    """
    np.random.seed(3)
    tf.compat.v1.set_random_seed(3)
    no, seq_len, dim = 32, 6, 3
    ori_data = np.random.uniform(0, 1, [no, seq_len, dim]).astype(np.float32)
    ori_data_static = np.random.randint(0, 2, size=(no, 1)).astype(np.float32)

    generated = train_timegan(
        ori_data,
        dict(TINY_PARAMS, iterations=2),
        filename=str(tmp_path / "test_static_synth"),
        ori_data_static=ori_data_static,
    )

    _assert_sane_generated_output(generated, expected_shape=(no, seq_len, dim))


def test_load_timegan_with_static_features_round_trip(tmp_path):
    """train then immediately load the same checkpoint, with static
    features -- confirms load_timegan's identical treatment (same
    optional ori_data_static param, same two-branch graph wiring) also
    works, not just train_timegan's."""
    np.random.seed(4)
    tf.compat.v1.set_random_seed(4)
    no, seq_len, dim = 32, 6, 3
    ori_data = np.random.uniform(0, 1, [no, seq_len, dim]).astype(np.float32)
    ori_data_static = np.random.randint(0, 2, size=(no, 1)).astype(np.float32)
    params = dict(TINY_PARAMS, iterations=2)

    train_timegan(
        ori_data,
        dict(params),
        filename=str(tmp_path / "test_roundtrip"),
        ori_data_static=ori_data_static,
    )
    generated = load_timegan(
        ori_data,
        dict(params),
        filename=str(tmp_path / "test_roundtrip-0"),
        ori_data_static=ori_data_static,
    )

    _assert_sane_generated_output(generated, expected_shape=(no, seq_len, dim))
