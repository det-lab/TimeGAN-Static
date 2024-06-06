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
from utils import batch_generator, extract_time, random_generator, rnn_cell

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
    # hidden_dim, num_layers, iterations, batch_size, module_name, z_dim, gamma
    return (
        parameters["hidden_dim"],
        parameters["num_layer"],
        parameters["iterations"],
        parameters["batch_size"],
        parameters["module"],
        parameters["dim"],
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
        print("e_cell")  # TEMP
        e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        print("e_output")  # TEMP
        e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length=T)
        print("H")  # TEMP
        # H = _layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)

        # Static features
        HS = _layers.stack(
            S, _layers.fully_connected, [static_dim for _ in range(num_layers)], activation_fn=tf.nn.sigmoid
        )
        h_inputs = tf.concat(e_outputs, HS)
        HT = _layers.fully_connected(h_inputs, hidden_dim, activation_fn=tf.nn.sigmoid)
    return HT, HS


def recovery(H, T, param):
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
        """
      d_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      d_outputs, d_last_states = tf.compat.v1.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length = T)
      YT_hat = _layers.fully_connected(d_outputs, 1, activation_fn=None)
      """
        # Bidirectional rnn cell
        fw_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        bw_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        c_inputs = tf.concat(H, S)
        c_outputs, c_last_states = tf.compat.v1.nn.biderectional_dynamic_rnn(
            fw_cell, bw_cell, c_inputs, dtype=tf.float32, sequence_length=T
        )

        # Temporal discriminator
        d_inputs = tf.concat(c_outputs, 2)
        d_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        dt_outputs, dt_last_states = tf.compat.v1.nn.dynamic_rnn(d_cell, d_inputs, dtype=tf.float32, sequence_length=T)
        YT_hat = _layers.fully_connected(dt_outputs, 1, activation_fn=None)

        # Static discriminator
        ds_outputs = _layers.stack(
            S, _layers.fully_connected, [static_dim for _ in range(num_layers)], activation_fn=tf.nn.sigmoid
        )
        YS_hat = _layers.fully_connected(ds_outputs, 1, activation_fn=sigmoid)

    return YT_hat, YS_hat


# TODO: UPDATE train_timegan AND load_timegan TO USE STATIC FEATURES
def train_timegan(ori_data, parameters, filename="timegan_save", version=0):
    """TimeGAN training function.

    Use original data as training set to generater synthetic data (time-series)
    Trains timegan from scratch

    Args:
      - ori_data: original time-series data
      - parameters: TimeGAN network parameters
      - filename: filename to save the model in, default "timegan_save"
      - version: version of the snapshot, default 0

    Returns:
      - generated_data: generated time-series data
    """
    # Initialization on the Graph
    tf.compat.v1.reset_default_graph()

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

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
    z_dim = dim
    gamma = 1

    # Input place holders
    X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
    Z = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, z_dim], name="myinput_z")
    T = tf.compat.v1.placeholder(tf.int32, [None], name="myinput_t")

    # Embedder & Recovery
    H = embedder(X, T, parameters)
    X_tilde = recovery(H, T, parameters)

    # Generator
    E_hat = generator(Z, T, parameters)
    H_hat = supervisor(E_hat, T, parameters)
    H_hat_supervise = supervisor(H, T, parameters)

    # Synthetic data
    X_hat = recovery(H_hat, T, parameters)

    # Discriminator
    Y_fake = discriminator(H_hat, T, parameters)
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
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V

    # Embedder network loss
    E_loss_T0 = tf.compat.v1.losses.mean_squared_error(X, X_tilde)
    E_loss0 = 10 * tf.sqrt(E_loss_T0)
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
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        # Train embedder
        _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb})
        # Checkpoint
        if itt % 1000 == 0:
            print("step: " + str(itt) + "/" + str(iterations) + ", e_loss: " + str(np.round(np.sqrt(step_e_loss), 4)))

    print("Finish Embedding Network Training")

    # 2. Training only with supervised loss
    print("Start Training with Supervised Loss Only")

    for itt in range(iterations):
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        # Train generator
        _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
        # Checkpoint
        if itt % 1000 == 0:
            print("step: " + str(itt) + "/" + str(iterations) + ", s_loss: " + str(np.round(np.sqrt(step_g_loss_s), 4)))

    print("Finish Training with Supervised Loss Only")

    # 3. Joint Training
    print("Start Joint Training")

    for itt in range(iterations):
        # Generator training (twice more than discriminator training)
        for kk in range(2):
            # Set mini-batch
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            # Train generator
            _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run(
                [G_solver, G_loss_U, G_loss_S, G_loss_V], feed_dict={Z: Z_mb, X: X_mb, T: T_mb}
            )
            # Train embedder
            _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})

        # Discriminator training
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        # Check discriminator loss before updating
        check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        # Train discriminator (only when the discriminator does not work well)
        if check_d_loss > 0.15:
            _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})

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
                + str(np.round(np.sqrt(step_g_loss_s), 4))
                + ", g_loss_v: "
                + str(np.round(step_g_loss_v, 4))
                + ", e_loss_t0: "
                + str(np.round(np.sqrt(step_e_loss_t0), 4))
            )
    print("Finish Joint Training")

    ## Synthetic data generation
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})

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


def load_timegan(ori_data, parameters, filename):
    """TimeGAN function.

    Use original data as training set to generater synthetic data (time-series)
    Loads from snapshot

    Args:
      - ori_data: original time-series data
      - parameters: TimeGAN network parameters
      - filename: filename of the snapshot to load

    Returns:
      - generated_data: generated time-series data
    """
    # Initialization on the Graph
    tf.compat.v1.reset_default_graph()

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

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
    z_dim = dim
    gamma = 1

    # Input place holders
    X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
    Z = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, z_dim], name="myinput_z")
    T = tf.compat.v1.placeholder(tf.int32, [None], name="myinput_t")

    # Embedder & Recovery
    H = embedder(X, T, parameters)
    X_tilde = recovery(H, T, parameters)

    # Generator
    E_hat = generator(Z, T, parameters)
    H_hat = supervisor(E_hat, T, parameters)
    H_hat_supervise = supervisor(H, T, parameters)

    # Synthetic data
    X_hat = recovery(H_hat, T, parameters)

    # Discriminator
    Y_fake = discriminator(H_hat, T, parameters)
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
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V

    # Embedder network loss
    E_loss_T0 = tf.compat.v1.losses.mean_squared_error(X, X_tilde)
    E_loss0 = 10 * tf.sqrt(E_loss_T0)
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
    generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})

    generated_data = list()

    for i in range(no):
        temp = generated_data_curr[i, : ori_time[i], :]
        generated_data.append(temp)

    # Renormalization
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val

    return generated_data
