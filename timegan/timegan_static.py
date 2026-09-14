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


# Unpack parameters
def unpack_parameters(parameters):
    # hidden_dim, static_dim, num_layers, iterations, batch_size, module_name,
    # z_dim, static_z_dim, gamma -- static_dim doubles as static_z_dim, same
    # way dim doubles as z_dim below: the static noise fed to the generator
    # has the same width as the real static features it's standing in for.
    return (
        parameters["hidden_dim"],
        parameters["static_dim"],
        parameters["num_layer"],
        parameters["iterations"],
        parameters["batch_size"],
        parameters["module"],
        parameters["dim"],
        parameters["static_dim"],
        1,
    )


# Components of the network
def embedder(X, T, S, param):
    """Embedding network between original feature space to latent space.

    Args:
      - X: input time-series features
      - T: input time information
      - S: static features

    Returns:
      - HT: temporal embeddings
      - HS: static embeddings
    """
    (
        hidden_dim,
        static_dim,
        num_layers,
        iterations,
        batch_size,
        module_name,
        z_dim,
        static_z_dim,
        gamma,
    ) = unpack_parameters(param)
    dim = z_dim
    with tf.compat.v1.variable_scope("embedder", reuse=tf.compat.v1.AUTO_REUSE):
        e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length=T)
        HT = _layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)

        # Static features, embedded independently of the temporal branch
        # for now (not concatenated into HT): merging a per-timestep tensor
        # (e_outputs) with a per-event one (HS) needs a broadcast/tiling
        # scheme this file never designed -- the original attempt here
        # called tf.concat(e_outputs, HS), which isn't valid usage (missing
        # an axis, and shapes don't line up regardless). HT is exactly the
        # non-static computation, so this can't regress the temporal path;
        # HS is real and valid but not yet consumed by any loss term -- see
        # train_timegan.
        HS = _layers.stack(
            S, _layers.fully_connected, [static_dim for _ in range(num_layers)], activation_fn=tf.nn.sigmoid
        )
    return HT, HS


def recovery(H, T, S, param):
    """Recovery network from latent space to original space.

    Args:
      - H: latent representation
      - T: input time information
      - S: latent representation of static features

    Returns:
      - XT_tilde: recovered temporal data
      - XS_tilde: recovered static data
    """
    (
        hidden_dim,
        static_dim,
        num_layers,
        iterations,
        batch_size,
        module_name,
        z_dim,
        static_z_dim,
        gamma,
    ) = unpack_parameters(param)
    dim = z_dim
    with tf.compat.v1.variable_scope("recovery", reuse=tf.compat.v1.AUTO_REUSE):
        r_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        r_outputs, r_last_states = tf.compat.v1.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length=T)
        XT_tilde = _layers.fully_connected(r_outputs, dim, activation_fn=tf.nn.sigmoid)

        # Static features
        rs_outputs = _layers.stack(
            S, _layers.fully_connected, [static_dim for _ in range(num_layers)], activation_fn=tf.nn.sigmoid
        )
        XS_tilde = _layers.fully_connected(rs_outputs, static_z_dim, activation_fn=tf.nn.sigmoid)
    return XT_tilde, XS_tilde


def generator(Z, T, S, param):
    """Generator function: Generate time-series data in latent space.

    Args:
      - Z: random temporal variables
      - T: input time information
      - S: random variables (static features)

    Returns:
      - ET: generated temporal embedding
      - ES: generated static embedding
    """
    (
        hidden_dim,
        static_dim,
        num_layers,
        iterations,
        batch_size,
        module_name,
        z_dim,
        static_z_dim,
        gamma,
    ) = unpack_parameters(param)
    dim = z_dim
    with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
        e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length=T)
        ET = _layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)

        # Static features
        ES = _layers.stack(
            S, _layers.fully_connected, [static_dim for _ in range(num_layers)], activation_fn=tf.nn.sigmoid
        )
    return ET, ES


# I don't think this needs the static features...
def supervisor(H, T, param):
    """Generate next sequence using the previous sequence.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """
    (
        hidden_dim,
        static_dim,
        num_layers,
        iterations,
        batch_size,
        module_name,
        z_dim,
        static_z_dim,
        gamma,
    ) = unpack_parameters(param)
    dim = z_dim
    with tf.compat.v1.variable_scope("supervisor", reuse=tf.compat.v1.AUTO_REUSE):
        e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
            [rnn_cell(module_name, hidden_dim) for _ in range(num_layers - 1)]
        )
        e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length=T)
        S = _layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
    return S


def discriminator(H, T, S, param):
    """Discriminate the original and synthetic time-series data.

    Args:
      - H: latent representation of temporal data
      - T: input time information
      - S: latent representation of static data

    Returns:
      - YT_hat: classification results between original and synthetic time-series
      - YS_hat: classification results between original and synthetic static features
    """
    (
        hidden_dim,
        static_dim,
        num_layers,
        iterations,
        batch_size,
        module_name,
        z_dim,
        static_z_dim,
        gamma,
    ) = unpack_parameters(param)
    dim = z_dim
    with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE):
        # Temporal discriminator -- reverted to the simple, single-
        # dynamic_rnn form this file already had preserved as a comment,
        # rather than the bidirectional/concat-with-S attempt that replaced
        # it: tf.concat(H, S) isn't valid (missing an axis, and mixing a
        # per-timestep tensor with a per-event one needs a broadcast/tiling
        # scheme this file never designed), and biderectional_dynamic_rnn
        # doesn't exist (typo for bidirectional_dynamic_rnn) regardless.
        # YT_hat is unaffected by S for now, same reasoning as embedder().
        d_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        d_outputs, d_last_states = tf.compat.v1.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length=T)
        YT_hat = _layers.fully_connected(d_outputs, 1, activation_fn=None)

        # Static discriminator -- real and valid, but (like HS/ES/XS_tilde)
        # not yet wired into any loss term; also fixes a bare, undefined
        # `sigmoid` name (was never tf.nn.sigmoid or tf.sigmoid).
        ds_outputs = _layers.stack(
            S, _layers.fully_connected, [static_dim for _ in range(num_layers)], activation_fn=tf.nn.sigmoid
        )
        YS_hat = _layers.fully_connected(ds_outputs, 1, activation_fn=tf.nn.sigmoid)

    return YT_hat, YS_hat


def _batch_generator_with_static(data, time, static, batch_size):
    """Same draw as timegan.utils.batch_generator, but also slices static
    features by the identical indices so S_mb stays aligned with X_mb/T_mb.
    Kept local to this file (not added to timegan.utils) since the non-
    static train_timegan/train_timegan_timed never need a static slice.
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]
    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)
    S_mb = list(static[i] for i in train_idx)
    return X_mb, T_mb, S_mb


def train_timegan(ori_data, ori_data_static, parameters, filename="timegan_save", version=0):
    """TimeGAN training function.

    Use original data as training set to generater synthetic data (time-series)
    Trains timegan from scratch

    Args:
      - ori_data: original time-series data
      - ori_data_static: per-event static features, shape (no, static_dim).
          Embedded/discriminated for real (HS/ES/XS_tilde/XS_hat/YS_* below
          are genuine graph nodes, not stubs), but not yet wired into any
          loss term -- training only optimizes the temporal reconstruction/
          adversarial objective, same as the non-static implementation.
          Finishing that (deciding how static loss terms should combine
          with the temporal ones) is separate follow-up work; this just
          makes the function callable and correct for what it does do.
      - parameters: TimeGAN network parameters
      - filename: filename to save the model in, default "timegan_save"
      - version: version of the snapshot, default 0

    Returns:
      - generated_data: generated time-series data (temporal only, same
        shape/meaning as the non-static implementation's output)
    """
    # Initialization on the Graph
    tf.compat.v1.reset_default_graph()

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    static_dim = np.asarray(ori_data_static).shape[-1]

    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = extract_time(ori_data)

    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    ## Build a RNN networks

    # Network Parameters
    hidden_dim = parameters["hidden_dim"]
    num_layers = parameters["num_layer"]
    iterations = parameters["iterations"]
    batch_size = parameters["batch_size"]
    module_name = parameters["module"]
    parameters["dim"] = dim
    parameters["static_dim"] = static_dim
    z_dim = dim
    gamma = 1

    # Input place holders
    X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
    Z = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, z_dim], name="myinput_z")
    T = tf.compat.v1.placeholder(tf.int32, [None], name="myinput_t")
    S = tf.compat.v1.placeholder(tf.float32, [None, static_dim], name="myinput_s")
    S_z = tf.compat.v1.placeholder(tf.float32, [None, static_dim], name="myinput_sz")

    # Embedder & Recovery
    HT, HS = embedder(X, T, S, parameters)
    XT_tilde, XS_tilde = recovery(HT, T, HS, parameters)

    # Generator
    ET, ES = generator(Z, T, S_z, parameters)
    H_hat = supervisor(ET, T, parameters)
    H_hat_supervise = supervisor(HT, T, parameters)

    # Synthetic data
    X_hat, XS_hat = recovery(H_hat, T, ES, parameters)

    # Discriminator
    Y_fake, YS_fake = discriminator(H_hat, T, ES, parameters)
    Y_real, YS_real = discriminator(HT, T, HS, parameters)
    Y_fake_e, YS_fake_e = discriminator(ET, T, ES, parameters)

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
    G_loss_S = tf.compat.v1.losses.mean_squared_error(HT[:, 1:, :], H_hat_supervise[:, :-1, :])

    # 3. Two Momments
    G_loss_V1 = tf.reduce_mean(
        input_tensor=tf.abs(
            tf.sqrt(tf.nn.moments(x=X_hat, axes=[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(x=X, axes=[0])[1] + 1e-6)
        )
    )
    G_loss_V2 = tf.reduce_mean(
        input_tensor=tf.abs((tf.nn.moments(x=X_hat, axes=[0])[0]) - (tf.nn.moments(x=X, axes=[0])[0]))
    )

    G_loss_V = G_loss_V1 + G_loss_V2

    # 4. Summation
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S + 1e-6) + 100 * G_loss_V

    # Embedder network loss
    E_loss_T0 = tf.compat.v1.losses.mean_squared_error(X, XT_tilde)
    E_loss0 = 10 * tf.sqrt(E_loss_T0 + 1e-6)
    E_loss = E_loss0 + 0.1 * G_loss_S

    # optimizer
    E0_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss0, var_list=e_vars + r_vars)
    E_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss, var_list=e_vars + r_vars)
    D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list=d_vars)
    G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=g_vars + s_vars)
    GS_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss_S, var_list=g_vars + s_vars)

    ## TimeGAN training
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Create Saver
    saver = tf.compat.v1.train.Saver()

    # 1. Embedding network training
    print("Start Embedding Network Training")

    for itt in range(iterations):
        # Set mini-batch
        X_mb, T_mb, S_mb = _batch_generator_with_static(ori_data, ori_time, ori_data_static, batch_size)
        # Train embedder
        _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb, S: S_mb})
        # Checkpoint
        if itt % 1000 == 0:
            print(
                "step: "
                + str(itt)
                + "/"
                + str(iterations)
                + ", e_loss: "
                + str(np.round(np.sqrt(step_e_loss + 1e-6), 4))
            )

    print("Finish Embedding Network Training")

    # 2. Training only with supervised loss
    print("Start Training with Supervised Loss Only")

    for itt in range(iterations):
        # Set mini-batch
        X_mb, T_mb, S_mb = _batch_generator_with_static(ori_data, ori_time, ori_data_static, batch_size)
        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        Sz_mb = np.random.uniform(0.0, 1, [batch_size, static_dim])
        # Train generator
        _, step_g_loss_s = sess.run(
            [GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb, S: S_mb, S_z: Sz_mb}
        )
        # Checkpoint
        if itt % 1000 == 0:
            print(
                "step: "
                + str(itt)
                + "/"
                + str(iterations)
                + ", s_loss: "
                + str(np.round(np.sqrt(step_g_loss_s + 1e-6), 4))
            )

    print("Finish Training with Supervised Loss Only")

    # 3. Joint Training
    print("Start Joint Training")

    for itt in range(iterations):
        # Generator training (twice more than discriminator training)
        for kk in range(2):
            # Set mini-batch
            X_mb, T_mb, S_mb = _batch_generator_with_static(ori_data, ori_time, ori_data_static, batch_size)
            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            Sz_mb = np.random.uniform(0.0, 1, [batch_size, static_dim])
            # Train generator
            _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run(
                [G_solver, G_loss_U, G_loss_S, G_loss_V],
                feed_dict={Z: Z_mb, X: X_mb, T: T_mb, S: S_mb, S_z: Sz_mb},
            )
            # Train embedder
            _, step_e_loss_t0 = sess.run(
                [E_solver, E_loss_T0], feed_dict={Z: Z_mb, X: X_mb, T: T_mb, S: S_mb, S_z: Sz_mb}
            )

        # Discriminator training
        # Set mini-batch
        X_mb, T_mb, S_mb = _batch_generator_with_static(ori_data, ori_time, ori_data_static, batch_size)
        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        Sz_mb = np.random.uniform(0.0, 1, [batch_size, static_dim])
        # Check discriminator loss before updating
        check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb, S: S_mb, S_z: Sz_mb})
        # Train discriminator (only when the discriminator does not work well)
        if check_d_loss > 0.15:
            _, step_d_loss = sess.run(
                [D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb, S: S_mb, S_z: Sz_mb}
            )

        # Print multiple checkpoints
        if itt % 1000 == 0:
            print(
                "step: "
                + str(itt)
                + "/"
                + str(iterations)
                + ", d_loss: "
                + str(np.round(step_d_loss, 4))
                + ", g_loss_u: "
                + str(np.round(step_g_loss_u, 4))
                + ", g_loss_s: "
                + str(np.round(np.sqrt(step_g_loss_s + 1e-6), 4))
                + ", g_loss_v: "
                + str(np.round(step_g_loss_v, 4))
                + ", e_loss_t0: "
                + str(np.round(np.sqrt(step_e_loss_t0 + 1e-6), 4))
            )
    print("Finish Joint Training")

    ## Synthetic data generation
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    Sz_all = np.random.uniform(0.0, 1, [no, static_dim])
    generated_data_curr = sess.run(
        X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time, S: ori_data_static, S_z: Sz_all}
    )

    generated_data = list()

    for i in range(no):
        temp = generated_data_curr[i, : ori_time[i], :]
        generated_data.append(temp)

    # Renormalization
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val

    # Save
    saver.save(sess, filename, global_step=version)

    return generated_data


def load_timegan(ori_data, ori_data_static, parameters, filename):
    """TimeGAN function.

    Use original data as training set to generater synthetic data (time-series)
    Loads from snapshot

    Args:
      - ori_data: original time-series data
      - ori_data_static: per-event static features, shape (no, static_dim) --
        see train_timegan's docstring for what this does and doesn't do yet
      - parameters: TimeGAN network parameters
      - filename: filename of the snapshot to load

    Returns:
      - generated_data: generated time-series data
    """
    # Initialization on the Graph
    tf.compat.v1.reset_default_graph()

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    static_dim = np.asarray(ori_data_static).shape[-1]

    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = extract_time(ori_data)

    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    ## Build a RNN networks

    # Network Parameters
    hidden_dim = parameters["hidden_dim"]
    num_layers = parameters["num_layer"]
    iterations = parameters["iterations"]
    batch_size = parameters["batch_size"]
    module_name = parameters["module"]
    parameters["dim"] = dim
    parameters["static_dim"] = static_dim
    z_dim = dim
    gamma = 1

    # Input place holders
    X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
    Z = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, z_dim], name="myinput_z")
    T = tf.compat.v1.placeholder(tf.int32, [None], name="myinput_t")
    S = tf.compat.v1.placeholder(tf.float32, [None, static_dim], name="myinput_s")
    S_z = tf.compat.v1.placeholder(tf.float32, [None, static_dim], name="myinput_sz")

    # Embedder & Recovery
    HT, HS = embedder(X, T, S, parameters)
    XT_tilde, XS_tilde = recovery(HT, T, HS, parameters)

    # Generator
    ET, ES = generator(Z, T, S_z, parameters)
    H_hat = supervisor(ET, T, parameters)
    H_hat_supervise = supervisor(HT, T, parameters)

    # Synthetic data
    X_hat, XS_hat = recovery(H_hat, T, ES, parameters)

    # Discriminator
    Y_fake, YS_fake = discriminator(H_hat, T, ES, parameters)
    Y_real, YS_real = discriminator(HT, T, HS, parameters)
    Y_fake_e, YS_fake_e = discriminator(ET, T, ES, parameters)

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
    G_loss_S = tf.compat.v1.losses.mean_squared_error(HT[:, 1:, :], H_hat_supervise[:, :-1, :])

    # 3. Two Momments
    G_loss_V1 = tf.reduce_mean(
        input_tensor=tf.abs(
            tf.sqrt(tf.nn.moments(x=X_hat, axes=[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(x=X, axes=[0])[1] + 1e-6)
        )
    )
    G_loss_V2 = tf.reduce_mean(
        input_tensor=tf.abs((tf.nn.moments(x=X_hat, axes=[0])[0]) - (tf.nn.moments(x=X, axes=[0])[0]))
    )

    G_loss_V = G_loss_V1 + G_loss_V2

    # 4. Summation
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S + 1e-6) + 100 * G_loss_V

    # Embedder network loss
    E_loss_T0 = tf.compat.v1.losses.mean_squared_error(X, XT_tilde)
    E_loss0 = 10 * tf.sqrt(E_loss_T0 + 1e-6)
    E_loss = E_loss0 + 0.1 * G_loss_S

    # optimizer
    E0_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss0, var_list=e_vars + r_vars)
    E_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss, var_list=e_vars + r_vars)
    D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list=d_vars)
    G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=g_vars + s_vars)
    GS_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss_S, var_list=g_vars + s_vars)

    # Load snapshot
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, filename)

    # Synthetic data generation
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    Sz_all = np.random.uniform(0.0, 1, [no, static_dim])
    generated_data_curr = sess.run(
        X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time, S: ori_data_static, S_z: Sz_all}
    )

    generated_data = list()

    for i in range(no):
        temp = generated_data_curr[i, : ori_time[i], :]
        generated_data.append(temp)

    # Renormalization
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val

    return generated_data
