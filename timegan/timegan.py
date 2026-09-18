"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

# mypy: ignore-errors
# Necessary Packages
import json
import os
from time import time_ns

import numpy as np
import tensorflow as tf
from tf_slim.layers import layers as _layers

from timegan.utils import batch_generator, extract_time, random_generator, rnn_cell

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()


# Min Max Scaler
def MinMaxScaler(data):
    """Min-Max Normalizer.

    Args:
      - data: raw data

    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val


# Unpack parameters -- just what embedder/recovery/generator/supervisor/
# discriminator actually use below. Used to also return iterations,
# batch_size, and a hardcoded gamma=1 that every one of those callers
# unpacked into a local variable and never referenced again; worse, that
# hardcoded gamma was a live lie for anyone who might have trusted it --
# train_timegan_timed's own gamma (configurable, read straight from
# parameters["gamma"], see _build_timegan_graph) is what's actually used
# during training and shadows this one entirely.
def unpack_parameters(parameters):
    return (
        parameters["hidden_dim"],
        parameters["num_layer"],
        parameters["module"],
        parameters["dim"],
    )


# Components of the network
def embedder(X, T, param, S=None):
    """Embedding network between original feature space to latent space.

    Args:
      - X: input time-series features
      - T: input time information
      - S: optional per-event static features, shape (batch, static_dim).
          When given, also embeds them (independently of the temporal
          branch: merging a per-timestep tensor with a per-event one
          needs a broadcast/tiling scheme nothing here has designed).
          param["static_dim"] must be set when S is given.

    Returns:
      - H: embeddings, if S is None (unchanged from before this had any
        static-feature support -- existing callers are unaffected)
      - (H, HS): embeddings and static embeddings, if S is given
    """
    hidden_dim, num_layers, module_name, dim = unpack_parameters(param)
    with tf.compat.v1.variable_scope("embedder", reuse=tf.compat.v1.AUTO_REUSE):
        e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length=T)
        H = _layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
        if S is None:
            return H
        static_dim = param["static_dim"]
        HS = _layers.stack(
            S, _layers.fully_connected, [static_dim for _ in range(num_layers)], activation_fn=tf.nn.sigmoid
        )
    return H, HS


def recovery(H, T, param, S=None):
    """Recovery network from latent space to original space.

    Args:
      - H: latent representation
      - T: input time information
      - S: optional static-feature latent representation (e.g. HS from
          embedder(), or ES from generator()) to recover back to raw
          static-feature space. param["static_dim"] must be set when
          given.

    Returns:
      - X_tilde: recovered data, if S is None (unchanged from before this
        had any static-feature support)
      - (X_tilde, XS_tilde): recovered temporal and static data, if S is
        given
    """
    hidden_dim, num_layers, module_name, dim = unpack_parameters(param)
    with tf.compat.v1.variable_scope("recovery", reuse=tf.compat.v1.AUTO_REUSE):
        r_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        r_outputs, r_last_states = tf.compat.v1.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length=T)
        X_tilde = _layers.fully_connected(r_outputs, dim, activation_fn=tf.nn.sigmoid)
        if S is None:
            return X_tilde
        static_dim = param["static_dim"]
        rs_outputs = _layers.stack(
            S, _layers.fully_connected, [static_dim for _ in range(num_layers)], activation_fn=tf.nn.sigmoid
        )
        XS_tilde = _layers.fully_connected(rs_outputs, static_dim, activation_fn=tf.nn.sigmoid)
    return X_tilde, XS_tilde


def generator(Z, T, param, S=None):
    """Generator function: Generate time-series data in latent space.

    Args:
      - Z: random variables
      - T: input time information
      - S: optional per-event random static-noise input, shape
          (batch, static_dim) -- the static-feature analog of Z.
          param["static_dim"] must be set when given.

    Returns:
      - E: generated embedding, if S is None (unchanged from before this
        had any static-feature support)
      - (E, ES): generated temporal and static embeddings, if S is given
    """
    hidden_dim, num_layers, module_name, dim = unpack_parameters(param)
    with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
        e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length=T)
        E = _layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
        if S is None:
            return E
        static_dim = param["static_dim"]
        ES = _layers.stack(
            S, _layers.fully_connected, [static_dim for _ in range(num_layers)], activation_fn=tf.nn.sigmoid
        )
    return E, ES


def supervisor(H, T, param):
    """Generate next sequence using the previous sequence.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """
    hidden_dim, num_layers, module_name, dim = unpack_parameters(param)
    with tf.compat.v1.variable_scope("supervisor", reuse=tf.compat.v1.AUTO_REUSE):
        e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
            [rnn_cell(module_name, hidden_dim) for _ in range(num_layers - 1)]
        )
        e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length=T)
        S = _layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
    return S


def discriminator(H, T, param, S=None):
    """Discriminate the original and synthetic time-series data.

    Args:
      - H: latent representation
      - T: input time information
      - S: optional static-feature latent representation (e.g. HS from
          embedder(), or ES from generator()) to discriminate real vs.
          synthetic static features. param["static_dim"] must be set
          when given.

    Returns:
      - Y_hat: classification results between original and synthetic
        time-series, if S is None (unchanged from before this had any
        static-feature support)
      - (Y_hat, YS_hat): temporal and static classification results, if S
        is given
    """
    hidden_dim, num_layers, module_name, dim = unpack_parameters(param)
    with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE):
        d_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        d_outputs, d_last_states = tf.compat.v1.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length=T)
        Y_hat = _layers.fully_connected(d_outputs, 1, activation_fn=None)
        if S is None:
            return Y_hat
        static_dim = param["static_dim"]
        ds_outputs = _layers.stack(
            S, _layers.fully_connected, [static_dim for _ in range(num_layers)], activation_fn=tf.nn.sigmoid
        )
        YS_hat = _layers.fully_connected(ds_outputs, 1, activation_fn=tf.nn.sigmoid)
    return Y_hat, YS_hat


def train_timegan(ori_data, parameters, filename="timegan_save", version=0, ori_data_static=None):
    """TimeGAN training function -- one-shot, non-resumable convenience
    wrapper around train_timegan_timed().

    This used to be a full, independent copy of train_timegan_timed's graph-
    building/training-loop code (predating the phase/resume/timing additions
    there), duplicating everything including its own copies of every
    embedder/generator/discriminator call and loss formula. That's the exact
    kind of drift risk that let timegan_static.py's copy of the same code
    silently break (see that file's history) -- two places to apply the same
    fix, easy to only remember one. train_timegan_timed's defaults
    (gamma=1, g_loss_s_weight=g_loss_v_weight=100) already reproduce this
    function's previously-hardcoded loss weights exactly, so delegating
    changes no behavior for any existing caller that isn't passing those
    keys.

    Use original data as training set to generater synthetic data (time-series)
    Trains timegan from scratch, in a single call, to completion.

    Args:
      - ori_data: original time-series data
      - parameters: TimeGAN network parameters
      - filename: filename to save the model in, default "timegan_save"
      - version: version of the snapshot, default 0
      - ori_data_static: optional per-event static features -- see
          train_timegan_timed's docstring, passed straight through

    Returns:
      - generated_data: generated time-series data

    Side effects new since the standalone-copy version: also writes
    `<filename>_history.jsonl` (loss/variance_check history -- harmless
    additional output, nothing removed) and names the Saver's checkpoint
    pointer file after `filename` instead of TF's shared default bare
    "checkpoint" (an improvement, not a regression -- that shared pointer
    file is exactly what used to get clobbered across different runs/
    filenames, which is why train_timegan_timed stopped using it).
    """
    phase, info = train_timegan_timed(
        ori_data,
        parameters,
        in_filename=filename,
        out_filename=filename,
        seconds=float("inf"),
        phase=1,
        current_iter=0,
        new=True,
        version=version,
        ori_data_static=ori_data_static,
    )
    if phase == -1:
        raise RuntimeError(f"train_timegan_timed failed: {info}")  # noqa: TRY003
    if phase != 4:
        # Unreachable with seconds=inf (the time-budget check can never
        # trip), but fail loudly rather than silently returning partial
        # training if that assumption is ever wrong.
        raise RuntimeError(  # noqa: TRY003
            f"train_timegan_timed did not reach phase 4 (got phase={phase}) despite seconds=inf"
        )
    return info


def _windowed_log_rms_jitter(x, seq_len, dim, window, eps=1e-9):
    """Log RMS sample-to-sample jitter in consecutive windows along the
    sequence, per row -- the building block of G_loss_T (texture loss).

    Why this and not a spectrum: G_loss_V compares values ACROSS events at one
    fixed timestep, so it can't see whether a single trace fluctuates
    realistically over time. A power spectrum looks like the natural fix, but
    for this project's data a batch-averaged spectrum is ~97% pulse rows at
    high frequency (a pulse's edges carry ~3500x the power of the noise
    floor, which is ~0.04% of the average), so it would match pulse shapes and
    give no incentive for noise texture at all. It is also stationary (one
    spectrum over the whole window), so a generator that is too rough early
    and too smooth late -- exactly the failure seen in every run -- can score
    well because the two errors cancel. So instead: the RMS of first
    differences (a high-pass energy) per window, on a log scale so the noise
    floor (~1e-6 scaled^2 of difference energy) and a smooth generated tail
    (~1e-10) are comparable, position-resolved so early and late errors can't
    cancel, and averaged over rows AFTER the log (a geometric mean) so tail
    windows are set by the typical noise-only row instead of the few
    high-power pulse rows.

    Args:
      - x: [batch, seq_len, dim]
      - seq_len, dim: static sizes (x's own static shape may be partly None)
      - window: number of first-differences per window; any remainder
          (seq_len - 1) % window at the end of the sequence is dropped.
      - eps: floor inside the log, in scaled^2 units. Deliberately ~1e-9: real
          noise difference-energy is ~1e-6, a perfectly smooth tail is ~1e-10.

    Returns:
      [batch, n_windows, dim] of 0.5 * log(mean squared first difference + eps).
    """
    n_win = (seq_len - 1) // window
    d = x[:, 1:, :] - x[:, :-1, :]
    d = d[:, : n_win * window, :]
    d = tf.reshape(d, [-1, n_win, window, dim])
    e = tf.reduce_mean(input_tensor=tf.square(d), axis=2)
    return 0.5 * tf.math.log(e + eps)


def _build_timegan_graph(parameters, max_seq_len, dim, static_dim=None):
    """Builds the full TimeGAN graph -- placeholders through the loss and
    optimizer ops -- shared by train_timegan_timed and load_timegan.

    load_timegan needs the loss/optimizer ops too, even though it never
    trains: purely so its Saver's variable list matches what was
    checkpointed (AdamOptimizer creates slot variables that get saved
    alongside the trainable ones -- a Saver built from a graph missing
    them can't restore a checkpoint that has them). Previously this whole
    function's worth of code was duplicated between the two callers,
    including the loss weights -- load_timegan's copy had drifted to
    hardcoded gamma=1/100x weights while train_timegan_timed's own copy
    had since gained the configurable gamma/g_loss_s_weight/g_loss_v_weight
    (harmless in practice, since load_timegan never executes those loss
    ops, but a real, easy-to-miss inconsistency sitting right next to the
    configurable version).

    Args:
      - parameters: network parameters. gamma/g_loss_s_weight/
          g_loss_v_weight defaults are applied here via setdefault, so
          they land back in the caller's own dict the same way
          parameters["dim"] does elsewhere.
      - max_seq_len, dim: from extract_time(ori_data)/ori_data.shape
      - static_dim: static-feature width, or None for no static support

    Returns a dict with everything a caller might need to feed/run:
      X, Z, T, S, S_z, Noise (placeholders; S/S_z are None if static_dim
      is None, Noise is None unless parameters["inject_noise"] is set),
      X_hat, E0_solver, E_solver, D_solver, G_solver, GS_solver,
      E_loss_T0, G_loss_S, G_loss_U, G_loss_V, G_loss_T, D_loss.
    """
    hidden_dim = parameters["hidden_dim"]
    num_layers = parameters["num_layer"]
    module_name = parameters["module"]
    z_dim = dim
    parameters.setdefault("gamma", 1)
    parameters.setdefault("g_loss_s_weight", 100.0)
    parameters.setdefault("g_loss_v_weight", 100.0)
    parameters.setdefault("inject_noise", False)
    parameters.setdefault("normalize_g_loss_v", False)
    parameters.setdefault("g_loss_t_weight", 0.0)
    parameters.setdefault("texture_window", 9)
    gamma = parameters["gamma"]

    # Input place holders
    X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
    Z = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, z_dim], name="myinput_z")
    T = tf.compat.v1.placeholder(tf.int32, [None], name="myinput_t")

    # Embedder, Recovery, Generator, Discriminator -- two branches (rather
    # than always passing S=None and letting each function's own S-is-None
    # check no-op) because embedder/recovery/generator/discriminator return
    # a plain value when S is None and a tuple when it's given: forcing the
    # same code to handle both would mean every line here unpacking a tuple
    # or not depending on a value only known at runtime.
    S = S_z = Noise = None
    if static_dim is not None:
        S = tf.compat.v1.placeholder(tf.float32, [None, static_dim], name="myinput_s")
        S_z = tf.compat.v1.placeholder(tf.float32, [None, static_dim], name="myinput_sz")

        H, HS = embedder(X, T, parameters, S=S)
        X_tilde, XS_tilde = recovery(H, T, parameters, S=HS)

        E_hat, ES = generator(Z, T, parameters, S=S_z)
        H_hat = supervisor(E_hat, T, parameters)
        H_hat_supervise = supervisor(H, T, parameters)

        X_hat, XS_hat = recovery(H_hat, T, parameters, S=ES)
        # Noise injection (see below) isn't wired up for the static-feature
        # branch -- no current caller needs it there, and it would double
        # this function's complexity for an unused case.
        X_hat_for_loss = X_hat

        Y_fake, YS_fake = discriminator(H_hat, T, parameters, S=ES)
        Y_real, YS_real = discriminator(H, T, parameters, S=HS)
        Y_fake_e, YS_fake_e = discriminator(E_hat, T, parameters, S=ES)
    else:
        H = embedder(X, T, parameters)
        X_tilde = recovery(H, T, parameters)

        E_hat = generator(Z, T, parameters)
        H_hat = supervisor(E_hat, T, parameters)
        H_hat_supervise = supervisor(H, T, parameters)

        X_hat = recovery(H_hat, T, parameters)
        X_hat_for_loss = X_hat
        H_hat_for_disc = H_hat
        if parameters["inject_noise"]:
            # The discriminator judges H_hat (embedding space), never the
            # recovered X_hat directly -- adding noise only to X_hat
            # wouldn't touch what actually drives G_loss_U, since nothing
            # downstream would see it. So: add a real, sampled noise
            # segment to X_hat (Noise is fed by the caller -- see
            # train_timegan_timed's noise bank, built from genuinely
            # pulse-free real events, pooled the same way real training
            # data is), then re-embed the result and let THAT stand in
            # for H_hat wherever the discriminator or moment-matching
            # loss look at the generator's output. Reuses the embedder's
            # existing trained weights (AUTO_REUSE) rather than a second,
            # untrained copy. The point: the generator's own weights are
            # only ever judged on pulse shape, not on faking noise
            # texture, since real noise is what actually gets judged.
            Noise = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_noise")
            X_hat_for_loss = X_hat + Noise
            H_hat_for_disc = embedder(X_hat_for_loss, T, parameters)

        Y_fake = discriminator(H_hat_for_disc, T, parameters)
        Y_real = discriminator(H, T, parameters)
        Y_fake_e = discriminator(E_hat, T, parameters)

    # Variables
    e_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith("embedder")]
    r_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith("recovery")]
    g_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith("generator")]
    s_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith("supervisor")]
    d_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith("discriminator")]

    # Discriminator loss
    D_loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss_fake_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
    D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

    # Generator loss
    # 1. Adversarial loss
    G_loss_U = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    G_loss_U_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)

    # 2. Supervised loss
    G_loss_S = tf.compat.v1.losses.mean_squared_error(H[:, 1:, :], H_hat_supervise[:, :-1, :])

    # 3. Two Momments
    # X_hat_for_loss is X_hat + injected noise when parameters["inject_noise"]
    # is set (non-static branch only -- see there), X_hat unchanged otherwise.
    real_mean, real_var = tf.nn.moments(x=X, axes=[0])
    gen_mean, gen_var = tf.nn.moments(x=X_hat_for_loss, axes=[0])
    real_std = tf.sqrt(real_var + 1e-6)
    gen_std = tf.sqrt(gen_var + 1e-6)
    if parameters["normalize_g_loss_v"]:
        # Both terms below are an ABSOLUTE mismatch at each timestep, then
        # averaged over timesteps with equal formal weight -- but real
        # per-timestep variance is highly non-uniform over time (e.g. this
        # project's pulses cluster their onset in roughly the first third
        # of the pooled window, so real std there runs ~1.7x higher than
        # in the tail). Equal formal weight over an unequal-scale quantity
        # means the timesteps with the largest real variance dominate the
        # absolute loss and get most of the gradient pressure, while
        # low-variance timesteps (the tail here) are nearly free to get
        # wrong. Dividing each timestep's mismatch by that timestep's own
        # real_std makes every timestep compete on relative terms instead,
        # so the tail's low-amplitude structure isn't cheap to ignore just
        # because it's low-amplitude. real_std is already bounded away
        # from 0 by the +1e-6 above.
        G_loss_V1 = tf.reduce_mean(input_tensor=tf.abs(gen_std - real_std) / real_std)
        G_loss_V2 = tf.reduce_mean(input_tensor=tf.abs(gen_mean - real_mean) / real_std)
    else:
        G_loss_V1 = tf.reduce_mean(input_tensor=tf.abs(gen_std - real_std))
        G_loss_V2 = tf.reduce_mean(input_tensor=tf.abs(gen_mean - real_mean))

    G_loss_V = G_loss_V1 + G_loss_V2

    # 4. Texture loss (see _windowed_log_rms_jitter's docstring for why this
    # is not redundant with G_loss_V, and why it is position-resolved and
    # log-scaled instead of a batch-averaged spectrum). X_hat_for_loss again
    # -- when inject_noise is set the generator's output is judged together
    # with the injected real noise, same reasoning as G_loss_V. NOTE: with
    # inject_noise the injected real noise already supplies the texture this
    # term measures, so the two together give the generator no incentive of
    # its own to make texture; use this term with inject_noise off.
    texture_window = max(1, min(int(parameters["texture_window"]), max_seq_len - 1))
    G_loss_T = tf.reduce_mean(
        input_tensor=tf.abs(
            tf.reduce_mean(
                input_tensor=_windowed_log_rms_jitter(X_hat_for_loss, max_seq_len, dim, texture_window), axis=0
            )
            - tf.reduce_mean(input_tensor=_windowed_log_rms_jitter(X, max_seq_len, dim, texture_window), axis=0)
        )
    )

    # 5. Summation
    G_loss = (
        G_loss_U
        + gamma * G_loss_U_e
        + parameters["g_loss_s_weight"] * tf.sqrt(G_loss_S + 1e-6)
        + parameters["g_loss_v_weight"] * G_loss_V
        + parameters["g_loss_t_weight"] * G_loss_T
    )

    # Embedder network loss
    E_loss_T0 = tf.compat.v1.losses.mean_squared_error(X, X_tilde)
    E_loss0 = 10 * tf.sqrt(E_loss_T0 + 1e-6)
    E_loss = E_loss0 + 0.1 * G_loss_S

    # optimizer
    E0_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss0, var_list=e_vars + r_vars)
    E_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss, var_list=e_vars + r_vars)
    D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list=d_vars)
    G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=g_vars + s_vars)
    GS_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss_S, var_list=g_vars + s_vars)

    return dict(
        X=X,
        Z=Z,
        T=T,
        S=S,
        S_z=S_z,
        Noise=Noise,
        X_hat=X_hat,
        E0_solver=E0_solver,
        E_solver=E_solver,
        D_solver=D_solver,
        G_solver=G_solver,
        GS_solver=GS_solver,
        E_loss_T0=E_loss_T0,
        G_loss_S=G_loss_S,
        G_loss_U=G_loss_U,
        G_loss_V=G_loss_V,
        G_loss_T=G_loss_T,
        D_loss=D_loss,
    )


def _extract_noise_bank(ori_data, onset_sigma=5):
    """Genuinely pulse-free rows of ori_data (already MinMax-scaled, the
    same space X_hat lives in), each centered on its own median -- for
    sampling real noise to inject into the generator's output during
    phase 3 (see _build_timegan_graph's inject_noise). Not just the
    pre-trigger segment of pulse-bearing events: a full pulse-free event
    gives a genuine noise realization at the same resolution/length
    train_timegan_timed actually trains on.

    Uses a dataset-wide MAD (not each row's own) to decide what counts as
    "pulse-free": a large enough pulse inflates its OWN row's MAD, which
    can loosen that row's own onset threshold enough to hide its own
    pulse (confirmed empirically in Traces_GAN's noise-characterization
    work) -- a global, robust reference avoids that.

    Returns:
        - noise_bank: array, shape (n_noise_events, seq_len, dim), or
          None if no pulse-free rows were found (caller should disable
          injection and warn rather than fail)
    """
    arr = np.asarray(ori_data).reshape(len(ori_data), -1)
    baselines = np.median(arr, axis=1)
    centered = arr - baselines[:, None]
    mads = np.median(np.abs(centered), axis=1) * 1.4826 + 1e-9
    global_mad = np.median(mads)
    is_noise = ~np.any(centered > onset_sigma * global_mad, axis=1)
    if not np.any(is_noise):
        return None
    noise_bank = centered[is_noise]
    return noise_bank.reshape(len(noise_bank), arr.shape[1], 1)


def train_timegan_timed(
    ori_data,
    parameters,
    in_filename,
    out_filename=None,
    seconds=3600,
    phase=1,
    current_iter=0,
    new=False,
    version=0,
    num_generate=None,
    on_training_complete=None,
    ori_data_static=None,
):
    """Trains a TimeGAN model for a specific number of seconds, then stops and saves the session.

    Resumable across four phases (embedding, supervised, joint/adversarial training, then
    generation), so a caller can invoke this repeatedly with a time budget per call until phase 4
    is reached.

    Phase 3 also periodically (same cadence as its loss logging) prints a variance_check line:
    the ratio of generated to real per-timestep standard deviation on the current minibatch, mean
    and range across timesteps. Loss values alone can look stable even when the generator has
    collapsed toward reproducing the dominant mode instead of the real data's actual spread --
    watch for this ratio trending toward 0 as training progresses.

    Every periodic print above (all three phases) is also appended as a JSON line to
    `<out_filename>_history.jsonl` (fields match whatever's in that phase's print line), so a
    caller can plot loss curves / the variance_check ratio over iterations after the fact instead
    of only reading them from scrolling log output. Appended across resumed invocations, same as
    the checkpoint itself.

    Args:
      - ori_data: the original data
      - parameters: network parameters (iterations is changed to remaining iterations)
      - in_filename: the filename to load from
      - out_filename: the filename to save to; if None, will save to in_filename instead
      - seconds: maximum number of seconds to train for
      - phase: the current phase of training (integer)
          - 1: Embedding network training
          - 2: Supervised loss
          - 3: Joint training
          - 4: Data generation
      - current_iter: the current iteration that training paused on
      - new: whether or not to start training network from scratch - defaults to False
      - version: version of the savefile
      - num_generate: how many synthetic sequences to produce in the phase-4 generation step;
          defaults to None, meaning one per training example (len(ori_data)), matching the
          original one-to-one behavior. Useful to request fewer (e.g. for a quick quality check)
          or more than the training set size; ori_data/ori_time are cycled with modulo indexing
          to support either direction.
      - parameters may also optionally include (defaults match the original TimeGAN paper's
          fixed values, so omitting them reproduces prior behavior exactly):
          - gamma: weight on the embedding-space adversarial terms (G_loss_U_e, D_loss_fake_e).
              Default 1.
          - g_loss_s_weight, g_loss_v_weight: weight on G_loss_S (supervised: MSE between a real
              sequence's next embedded step and the generator's teacher-forced prediction of it)
              and G_loss_V (moment matching: batch-aggregate mean/std alignment) in the generator's
              total loss. Default 100 each, matching the original paper. Both are regression-style
              losses, not adversarial ones -- for any timestep where multiple real sequences share
              a similar state but diverge afterward, their MSE-optimal target is the conditional
              *average* of those futures, not a faithful sample of one. At the paper's default
              weight this term can dominate the adversarial one (G_loss_U) in the total gradient,
              biasing the generator toward "the average-looking sequence" regardless of its noise
              input -- mode collapse that per-batch loss values alone don't clearly reveal. Lower
              these (the two terms independently, since they push in the same mean-seeking
              direction for different reasons) if generated output looks collapsed toward one or a
              few dominant modes despite reasonable-looking loss curves; going to zero removes the
              stabilizing effect these terms provide over pure adversarial training on sequences,
              so tune down rather than eliminate.
          - inject_noise: default False. When True, phase 3 adds a real, sampled noise segment
              (drawn from genuinely pulse-free rows of ori_data itself -- see
              _extract_noise_bank) onto the generator's recovered output, then re-embeds the
              result and uses THAT wherever the discriminator or moment-matching loss (G_loss_V)
              would otherwise see the generator's own output -- not just tacked onto the final
              output after the fact, since the discriminator judges the embedded H_hat, never the
              recovered X_hat directly, so noise added only there would never reach what actually
              drives G_loss_U. The point: the generator's own weights are only ever judged on
              pulse shape, not on also faking noise texture, since real noise is what actually
              gets judged. Only affects phase 3 (where the discriminator is active) -- phases 1/2
              and phase 4 generation are unaffected, and require no matching change from a caller.
              Disabled (and a warning printed) if ori_data has no genuinely pulse-free rows to
              build a noise bank from.
          - normalize_g_loss_v: default False (reproduces prior behavior exactly). G_loss_V's two
              terms each average an ABSOLUTE per-timestep mismatch (std, then mean) over all
              timesteps with equal formal weight, but real per-timestep variance is not uniform
              over time -- for this project's data, pulse onsets cluster early in the pooled
              window, so real std runs substantially higher there than in the tail. Equal formal
              weight over that unequal scale means high-variance timesteps dominate the absolute
              loss and get most of the gradient pressure, while low-variance timesteps are nearly
              free for the generator to get wrong -- observed in practice as generated output
              that reproduces structure only where real variance is largest (e.g. the front of a
              pulse trace) and goes flat elsewhere, especially pronounced when combined with
              inject_noise (which lowers the already-weak tail pressure further during training,
              since injected real noise can satisfy it there without the generator doing
              anything, but isn't present at generation time to cover the resulting gap). When
              True, both G_loss_V terms are instead divided by that timestep's own real std
              before averaging, so every timestep competes on relative rather than absolute
              terms.
          - g_loss_t_weight: default 0.0 (off -- reproduces prior behavior exactly, this term
              doesn't exist in the original paper). G_loss_V, normalized or not, only ever
              compares values ACROSS events at one fixed timestep -- it never looks at how a
              single sequence's own values fluctuate over time, so it can't tell a smooth trace
              from a jittery one. G_loss_T compares, per position window along the sequence, the
              batch-mean of log RMS sample-to-sample jitter (first differences) of X_hat_for_loss
              against X's -- see _windowed_log_rms_jitter's docstring. Position-resolved so a
              generator that is too rough early and too smooth late can't score well by the two
              errors cancelling; log-scaled so the noise floor and a smooth tail are comparable;
              averaged over rows after the log so tail windows reflect the typical noise-only
              row rather than pulse edges. Roughly 0.4 between two real batches vs 1.3-2.8
              (noise floor missing in the tail, position-dependent) for the runs that motivated
              it. Use with inject_noise off: injected real noise already supplies the texture
              this term measures.
          - texture_window: default 9. Number of first-differences per window for G_loss_T
              (clipped to seq_len - 1; any remainder at the end of the sequence is dropped).
      - on_training_complete: optional zero-argument callback invoked right after phase 3's
          post-training checkpoint save, before generation begins. Lets a caller record that
          training is done (e.g. write its own progress marker) in case generation itself gets
          killed before this function returns -- a resubmit with phase=4 and new=False resumes
          directly at generation using that checkpoint, skipping training entirely.
      - ori_data_static: optional per-event static features, shape
          (no, static_dim). When given, static features are genuinely
          embedded/discriminated (real graph nodes throughout: HS/ES/
          XS_tilde/XS_hat/YS_* below), but intentionally don't participate
          in any loss term yet -- training still only optimizes the
          temporal reconstruction/adversarial objective. Wiring static
          loss terms in is separate follow-up work (how should they
          combine with the temporal ones?); this makes the plumbing
          correct for what it does do, without changing anything about
          the temporal path when omitted (the default).

    Returns:
      A tuple of variable length; the first element of the tuple determines its length and content.
        - First value of -1 (error): (-1, msg)
            - -1: phase indicator; indicates that there is an error
            - msg: error message
        - First value of 1, 2 or 3 (the network still needs to train): (next_phase, iter)
            - next_phase: the phase to be run in the next use of this function; either phase, or
              phase+1 depending upon if the current phase has completed
            - iter: how many iterations have been completed
        - First value of 4 (training is finished; data is generated): (4, generated_data)
            - 4: phase indicator; should be 4 to indicate that training has been finished/data has
              been generated
            - generated_data: generated data
    """
    # Set output filename to input filename if no output filename is provided
    if out_filename is None:
        out_filename = in_filename

    # Saver.save()'s pointer file defaults to a bare "checkpoint" in the
    # working directory, shared (and clobbered) across every out_filename --
    # name it after out_filename instead so concurrent/different runs don't
    # overwrite each other's pointer.
    checkpoint_filename = os.path.basename(out_filename) + "_checkpoint"

    # Loss/variance_check history, one JSON object per line, same fields
    # as whatever's printed at each periodic checkpoint below. Opened in
    # append mode so a killed-and-resumed run's history survives across
    # invocations, same as the checkpoint itself -- iter numbers don't
    # repeat across a resume since `current_iter` picks up where the
    # previous invocation left off.
    history_filename = out_filename + "_history.jsonl"

    def log_history(record):
        with open(history_filename, "a") as f:
            f.write(json.dumps(record) + "\n")

    # Initialization on the Graph
    tf.compat.v1.reset_default_graph()

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    static_dim = np.asarray(ori_data_static).shape[-1] if ori_data_static is not None else None

    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = extract_time(ori_data)

    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    ## Build a RNN networks

    # Network Parameters -- just what this function's own training loop
    # needs directly; _build_timegan_graph reads hidden_dim/num_layer/
    # module/gamma/etc. itself from the same parameters dict.
    iterations = parameters["iterations"]
    batch_size = parameters["batch_size"]
    parameters["dim"] = dim
    parameters["static_dim"] = static_dim
    z_dim = dim  # still used below for random_generator() calls

    graph = _build_timegan_graph(parameters, max_seq_len, dim, static_dim)
    X, Z, T, S, S_z = graph["X"], graph["Z"], graph["T"], graph["S"], graph["S_z"]
    Noise = graph["Noise"]
    X_hat = graph["X_hat"]
    E0_solver, E_solver = graph["E0_solver"], graph["E_solver"]
    D_solver, G_solver, GS_solver = graph["D_solver"], graph["G_solver"], graph["GS_solver"]
    E_loss_T0, G_loss_S = graph["E_loss_T0"], graph["G_loss_S"]
    G_loss_U, G_loss_V, D_loss = graph["G_loss_U"], graph["G_loss_V"], graph["D_loss"]
    G_loss_T = graph["G_loss_T"]

    noise_bank = None
    if parameters["inject_noise"]:
        noise_bank = _extract_noise_bank(ori_data)
        if noise_bank is None:
            print(
                "WARNING: inject_noise was requested but no pulse-free events were found in "
                "this training data -- disabling noise injection for this run.",
                flush=True,
            )

    def _draw_noise(n):
        """A batch of real, sampled noise segments for injection -- None
        if inject_noise wasn't requested (or no pulse-free events were
        found to build a bank from)."""
        if noise_bank is None:
            return None
        idx = np.random.randint(0, len(noise_bank), size=n)
        return noise_bank[idx]

    def _noise_extras(noise_mb):
        """Extra feed_dict entry for Noise, or {} when there's nothing to
        inject (only references Noise, which only exists as a graph
        tensor when inject_noise was requested, if noise_mb is actually
        not None, which only happens in that same case)."""
        if noise_mb is not None:
            return {Noise: noise_mb}
        return {}

    def _draw_batch():
        """batch_generator(), plus a static-feature slice (aligned to the
        same drawn indices) when ori_data_static was given -- always
        returns a 3-tuple so every call site below can unpack the same
        way regardless, with S_mb simply None when there's nothing to
        slice."""
        if ori_data_static is not None:
            return batch_generator(ori_data, ori_time, batch_size, static=ori_data_static)
        x_mb, t_mb = batch_generator(ori_data, ori_time, batch_size)
        return x_mb, t_mb, None

    def _draw_static_noise(n):
        """Sz_mb: the static-feature analog of Z_mb (random_generator) --
        no time dimension, since static features aren't per-timestep."""
        if ori_data_static is None:
            return None
        return np.random.uniform(0.0, 1, [n, static_dim])

    def _static_extras(s_mb=None, sz_mb=None):
        """Extra feed_dict entries for S/S_z, or {} when there's nothing
        static in play -- merge into a feed_dict with **, safe to call
        unconditionally (only references S/S_z, which only exist as names
        when ori_data_static was given, if s_mb/sz_mb are actually not
        None, which only happens in that same case)."""
        extras = {}
        if s_mb is not None:
            extras[S] = s_mb
        if sz_mb is not None:
            extras[S_z] = sz_mb
        return extras

    ## TimeGAN training
    sess = tf.compat.v1.Session()
    # Initialize if training from scratch
    if new:
        phase = 1
        current_iter = 0
        sess.run(tf.compat.v1.global_variables_initializer())

    # Create Saver
    saver = tf.compat.v1.train.Saver()
    # Load data if continuing training
    if not new:
        saver.restore(sess, in_filename + "-" + str(version))

    # Set up time stuff
    start_time = time_ns()  # Get the start time of the training
    max_time_ns = seconds * (10**9)  # Convert seconds to nanoseconds

    # 1. Embedding network training
    if phase == 1:
        print("Start Embedding Network Training", flush=True)

        for itt in range(current_iter, iterations):
            # Set mini-batch
            X_mb, T_mb, S_mb = _draw_batch()
            # Train embedder
            _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb, **_static_extras(S_mb)})

            if itt % 50 == 0:
                elapsed = (time_ns() - start_time) / 1e9
                print(
                    f"phase 1 iter {itt}/{iterations} e_loss={step_e_loss:.4f} " f"elapsed={elapsed:.0f}s", flush=True
                )
                log_history({"phase": 1, "iter": itt, "elapsed": elapsed, "e_loss": float(step_e_loss)})

            # End/suspend training if time is over max
            now = time_ns()
            if now - start_time >= max_time_ns:
                saver.save(sess, out_filename, global_step=version, latest_filename=checkpoint_filename)
                return (1, itt)

        # Training phase finished, save model and increment phase
        print("Finish Embedding Network Training", flush=True)
        saver.save(sess, out_filename, global_step=version, latest_filename=checkpoint_filename)
        phase = 2
    if phase == 2:
        # 2. Training only with supervised loss
        print("Start Training with Supervised Loss Only", flush=True)

        for itt in range(current_iter, iterations):
            # Set mini-batch
            X_mb, T_mb, S_mb = _draw_batch()
            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            Sz_mb = _draw_static_noise(batch_size)
            # Train generator
            _, step_g_loss_s = sess.run(
                [GS_solver, G_loss_S],
                feed_dict={Z: Z_mb, X: X_mb, T: T_mb, **_static_extras(S_mb, Sz_mb)},
            )

            if itt % 50 == 0:
                elapsed = (time_ns() - start_time) / 1e9
                print(
                    f"phase 2 iter {itt}/{iterations} g_loss_s={step_g_loss_s:.4f} " f"elapsed={elapsed:.0f}s",
                    flush=True,
                )
                log_history({"phase": 2, "iter": itt, "elapsed": elapsed, "g_loss_s": float(step_g_loss_s)})

            # End/suspend training if time is over max
            now = time_ns()
            if now - start_time >= max_time_ns:
                saver.save(sess, out_filename, global_step=version, latest_filename=checkpoint_filename)
                return (2, itt)

        # Training phase finished, save model and increment phase
        print("Finish Training with Supervised Loss Only", flush=True)
        saver.save(sess, out_filename, global_step=version, latest_filename=checkpoint_filename)
        phase = 3
    if phase == 3:
        # 3. Joint Training
        print("Start Joint Training", flush=True)

        for itt in range(current_iter, iterations):
            # Generator training (twice more than discriminator training)
            for kk in range(2):
                # Set mini-batch
                X_mb, T_mb, S_mb = _draw_batch()
                # Random vector generation
                Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
                Sz_mb = _draw_static_noise(batch_size)
                Noise_mb = _draw_noise(batch_size)
                # Train generator
                _, step_g_loss_u, step_g_loss_s, step_g_loss_v, step_g_loss_t = sess.run(
                    [G_solver, G_loss_U, G_loss_S, G_loss_V, G_loss_T],
                    feed_dict={
                        Z: Z_mb,
                        X: X_mb,
                        T: T_mb,
                        **_static_extras(S_mb, Sz_mb),
                        **_noise_extras(Noise_mb),
                    },
                )
                # Train embedder
                _, step_e_loss_t0 = sess.run(
                    [E_solver, E_loss_T0],
                    feed_dict={Z: Z_mb, X: X_mb, T: T_mb, **_static_extras(S_mb, Sz_mb)},
                )

            # Discriminator training
            # Set mini-batch
            X_mb, T_mb, S_mb = _draw_batch()
            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            Sz_mb = _draw_static_noise(batch_size)
            Noise_mb = _draw_noise(batch_size)
            # Check discriminator loss before updating
            check_d_loss = sess.run(
                D_loss,
                feed_dict={
                    X: X_mb,
                    T: T_mb,
                    Z: Z_mb,
                    **_static_extras(S_mb, Sz_mb),
                    **_noise_extras(Noise_mb),
                },
            )
            # Train discriminator (only when the discriminator does not work well)
            if check_d_loss > 0.15:
                _, step_d_loss = sess.run(
                    [D_solver, D_loss],
                    feed_dict={
                        X: X_mb,
                        T: T_mb,
                        Z: Z_mb,
                        **_static_extras(S_mb, Sz_mb),
                        **_noise_extras(Noise_mb),
                    },
                )

            if itt % 50 == 0:
                elapsed = (time_ns() - start_time) / 1e9
                print(
                    f"phase 3 iter {itt}/{iterations} g_loss_u={step_g_loss_u:.4f} "
                    f"g_loss_s={step_g_loss_s:.4f} g_loss_v={step_g_loss_v:.4f} "
                    f"g_loss_t={step_g_loss_t:.4f} "
                    f"d_loss={check_d_loss:.4f} elapsed={elapsed:.0f}s",
                    flush=True,
                )

                # Per-timestep variance check on the same minibatch used for
                # the discriminator loss above: catches a generator
                # collapsing toward the mean while the run is still in
                # progress, rather than only after generation finishes and
                # the output looks flat. Loss values alone don't show this
                # -- a stable-looking d_loss/g_loss_u is consistent with
                # both a healthy adversarial game AND a generator that's
                # settled for reproducing the dominant mode.
                gen_batch = sess.run(X_hat, feed_dict={Z: Z_mb, X: X_mb, T: T_mb, **_static_extras(S_mb, Sz_mb)})
                real_std = np.std(X_mb, axis=0)
                gen_std = np.std(gen_batch, axis=0)
                ratio = gen_std / (real_std + 1e-8)
                print(
                    f"phase 3 iter {itt}/{iterations} variance_check "
                    f"(generated_std/real_std, per timestep): "
                    f"mean={ratio.mean():.3f} min={ratio.min():.3f} max={ratio.max():.3f}",
                    flush=True,
                )
                log_history(
                    {
                        "phase": 3,
                        "iter": itt,
                        "elapsed": elapsed,
                        "g_loss_u": float(step_g_loss_u),
                        "g_loss_s": float(step_g_loss_s),
                        "g_loss_v": float(step_g_loss_v),
                        "g_loss_t": float(step_g_loss_t),
                        "d_loss": float(check_d_loss),
                        "variance_ratio_mean": float(ratio.mean()),
                        "variance_ratio_min": float(ratio.min()),
                        "variance_ratio_max": float(ratio.max()),
                    }
                )

            # End/suspend training if time is over max
            now = time_ns()
            if now - start_time >= max_time_ns:
                saver.save(sess, out_filename, global_step=version, latest_filename=checkpoint_filename)
                return (3, itt)
        # Final training phase finished, proceed to data generation
        print("Finish Joint Training", flush=True)
        # Checkpoint here, before generation -- generation runs a separate
        # sess.run() below that a large `no` can crash (see the chunking
        # note below), and until now the only save() after joint training
        # completes was the one at the very end of this function, past that
        # risk. Without this, a generation-step crash loses all of phase
        # 3's training with no way to resume it.
        saver.save(sess, out_filename, global_step=version, latest_filename=checkpoint_filename)
        # A caller-supplied hook to record that training is done and only
        # generation remains, e.g. writing a progress marker -- this function
        # doesn't return control between the checkpoint above and the
        # generation loop below, so without this hook a hard kill (SLURM
        # walltime, etc.) during generation looks indistinguishable from a
        # kill before training ever finished, and a resubmit would restart
        # from phase 1 despite the checkpoint above holding a trained model.
        # A resubmit with phase=4 and new=False resumes correctly, since the
        # phase==1/2/3 blocks above are simple `if`s (not elif) that are
        # skipped when phase is already 4, falling straight through to
        # generation using the restored checkpoint.
        if on_training_complete is not None:
            on_training_complete()
    if phase > 4 or phase <= 0:
        # Invalid phase number
        return (-1, str(phase) + " is not a valid phase indicator")

    ## Synthetic data generation
    # Chunked at batch_size rather than one sess.run() over all `no`
    # events -- confirmed on an A100 that a single-shot call with
    # no=289,636 (hidden_dim=128) crashes TensorFlow's GPU concat kernel
    # ("invalid configuration argument"), a fatal C++-level abort that
    # skips Python exception handling entirely. batch_size is already
    # proven safe at this hardware/model size (every training step above
    # uses it), so reuse it here rather than guess a new threshold.
    #
    # num_generate lets a caller ask for fewer (or more) synthetic sequences
    # than there are training examples -- e.g. a debug run only needs a few
    # hundred samples to sanity-check quality, not one per training event.
    # Defaults to `no` to match prior behavior exactly. ori_time/ori_data are
    # cycled with modulo indexing so num_generate can exceed `no` too.
    if num_generate is None:
        num_generate = no
    print(f"Start Data Generation, {num_generate} sequences in batches of {batch_size}", flush=True)
    gen_time = [ori_time[i % no] for i in range(num_generate)]
    generated_data_curr = np.zeros([num_generate, max_seq_len, dim])
    for start in range(0, num_generate, batch_size):
        end = min(start + batch_size, num_generate)
        T_chunk = gen_time[start:end]
        Z_chunk = random_generator(end - start, z_dim, T_chunk, max_seq_len)
        X_chunk = ori_data[[i % no for i in range(start, end)]]
        S_chunk = ori_data_static[[i % no for i in range(start, end)]] if ori_data_static is not None else None
        Sz_chunk = _draw_static_noise(end - start)
        generated_data_curr[start:end] = sess.run(
            X_hat, feed_dict={Z: Z_chunk, X: X_chunk, T: T_chunk, **_static_extras(S_chunk, Sz_chunk)}
        )

        # Chunks, not individual sequences, so this can't reuse the `itt % 50`
        # pacing above -- print roughly as often instead (every 50 chunks).
        if (start // batch_size) % 50 == 0:
            elapsed = (time_ns() - start_time) / 1e9
            print(f"generation {end}/{num_generate} elapsed={elapsed:.0f}s", flush=True)
    print("Finish Data Generation", flush=True)

    generated_data = list()

    for i in range(num_generate):
        temp = generated_data_curr[i, : gen_time[i], :]
        generated_data.append(temp)

    # Renormalization
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val

    # Save
    saver.save(sess, out_filename, global_step=version, latest_filename=checkpoint_filename)

    return (4, generated_data)


def load_timegan(ori_data, parameters, filename, ori_data_static=None):
    """TimeGAN function.

    Use original data as training set to generater synthetic data (time-series)
    Loads from snapshot

    Args:
      - ori_data: original time-series data
      - parameters: TimeGAN network parameters
      - filename: filename of the snapshot to load
      - ori_data_static: optional per-event static features, shape
          (no, static_dim) -- see train_timegan_timed's docstring for what
          this does and doesn't do. Only matters here insofar as the
          checkpoint being restored was itself trained with (or without)
          static features -- pass whatever was used for training.

    Returns:
      - generated_data: generated time-series data
    """
    # Initialization on the Graph
    tf.compat.v1.reset_default_graph()

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    static_dim = np.asarray(ori_data_static).shape[-1] if ori_data_static is not None else None

    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = extract_time(ori_data)

    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    ## Build a RNN networks
    parameters["dim"] = dim
    parameters["static_dim"] = static_dim
    z_dim = dim  # still used below for random_generator()

    # Same graph-building as train_timegan_timed (shared via
    # _build_timegan_graph): the checkpoint being restored must have been
    # built with the exact same graph shape it was saved with, so this
    # needs to match whichever branch (static features or not) actually
    # trained it -- pass the same ori_data_static that run used.
    graph = _build_timegan_graph(parameters, max_seq_len, dim, static_dim)
    X, Z, T, S, S_z = graph["X"], graph["Z"], graph["T"], graph["S"], graph["S_z"]
    X_hat = graph["X_hat"]

    # Load snapshot
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, filename)

    # Synthetic data generation
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    static_extras = {}
    if ori_data_static is not None:
        static_extras = {S: ori_data_static, S_z: np.random.uniform(0.0, 1, [no, static_dim])}
    generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time, **static_extras})

    generated_data = list()

    for i in range(no):
        temp = generated_data_curr[i, : ori_time[i], :]
        generated_data.append(temp)

    # Renormalization
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val

    return generated_data
