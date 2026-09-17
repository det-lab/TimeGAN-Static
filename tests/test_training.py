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
from timegan.timegan import load_timegan, train_timegan, train_timegan_timed

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
