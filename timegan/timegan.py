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

# Necessary Packages
import tensorflow as tf
import numpy as np
from utils import extract_time, rnn_cell, random_generator, batch_generator
# import wandb

tf.config.run_functions_eagerly(True)


# TODO: check rnn_cell
# TODO: check extract_time

def timegan(ori_data, parameters):
    """TimeGAN function (TF2 version)."""
    
    # Convert original data to NumPy array
    ori_data = np.asarray(ori_data)
    no, seq_len, dim = ori_data.shape
    
    # Extract time information
    ori_time, max_seq_len = extract_time(ori_data)
    
    def min_max_scaler(data):
        # finding min value per feature per sample 
        min_val = np.min(data, axis=(0, 1)) # for each sample, across all time, considering the minimum value for each feature across all time
        max_val = np.max(data, axis=(0, 1))
        norm_data = (data - min_val) / (max_val + 1e-7)
        return norm_data, min_val, max_val
    
    ori_data, min_val, max_val = min_max_scaler(ori_data)
    
    # Network parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    z_dim = dim
    gamma = 1
    
    # Define RNN layers
    def build_rnn(num_layers, hidden_dim, output_dim, activation='sigmoid', return_sequences=True):
        return tf.keras.Sequential([
            tf.keras.layers.RNN(
                [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)],
                return_sequences=return_sequences
            ),
            tf.keras.layers.Dense(output_dim, activation='sigmoid')
        ])

    class TimeGAN(tf.keras.Model):
        def __init__(self):
            super(TimeGAN, self).__init__()
            self.embedder = build_rnn(num_layers, hidden_dim, output_dim=hidden_dim)
            self.recovery = build_rnn(num_layers, dim, output_dim=dim)
            self.generator = build_rnn(num_layers, hidden_dim, output_dim=hidden_dim)
            self.supervisor = build_rnn(num_layers - 1, hidden_dim, output_dim=hidden_dim)
            # self.discriminator = build_rnn(num_layers, 1, return_sequences=False
            self.discriminator = build_rnn(num_layers, hidden_dim, output_dim=1, activation=None)
            # self.dense = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')
            # self.output_dense = tf.keras.layers.Dense(dim, activation='sigmoid')
            self.classifier = tf.keras.layers.Dense(1, activation=None)
        
        def call(self, X, Z, T):
            H = self.embedder(X)
            # H = self.dense(H)

            X_tilde = self.recovery(H)
            # X_tilde = self.output_dense(X_tilde)

            E_hat = self.generator(Z)
            # E_hat = self.dense(E_hat)

            H_hat = self.supervisor(E_hat) # called S in og code
            # H_hat = self.dense(H_hat)

            H_hat_supervise = self.supervisor(H)
            # H_hat_supervise = self.dense(H_hat_supervise)

            X_hat = self.recovery(H_hat)
            # X_hat = self.output_dense(X_hat)

            Y_fake = self.discriminator(H_hat)
            # Y_fake = self.classifier(Y_fake)

            Y_real = self.discriminator(H)
            # Y_real = self.classifier(Y_real)

            Y_fake_e = self.discriminator(E_hat)
            # Y_fake_e = self.classifier(Y_fake_e)

            return H, X_tilde, H_hat, H_hat_supervise, X_hat, Y_fake, Y_real, Y_fake_e
    
    model = TimeGAN()
    optimizer_emb = tf.keras.optimizers.Adam()
    optimizer_sup = tf.keras.optimizers.Adam()
    optimizer_gen = tf.keras.optimizers.Adam()
    optimizer_joint_emb = tf.keras.optimizers.Adam()
    optimizer_disc = tf.keras.optimizers.Adam()
    # optimizer = tf.keras.optimizers.Adam()
    
    
    @tf.function
    def train_step(X, Z, T, itt, train_type):
        with tf.GradientTape() as tape:
            # print(type(X), type(Z), type(T))
            H, X_tilde, H_hat, H_hat_supervise, X_hat, Y_fake, Y_real, Y_fake_e = model(X, Z, T)
            
            D_loss_real = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(Y_real), Y_real)
            D_loss_fake = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(Y_fake), Y_fake)
            D_loss_fake_e = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(Y_fake_e), Y_fake_e)
            D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
            
            G_loss_U = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(Y_fake), Y_fake)
            G_loss_U_e = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(Y_fake_e), Y_fake_e)

            G_loss_S = tf.keras.losses.MeanSquaredError()(H[:, 1:, :], H_hat_supervise[:, :-1, :])

            G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.math.reduce_variance(X_hat, axis=0) + 1e-6) - tf.sqrt(tf.math.reduce_variance(X, axis=0) + 1e-6)))
            G_loss_V2 = tf.reduce_mean(tf.abs(tf.reduce_mean(X_hat, axis=0) - tf.reduce_mean(X, axis=0)))
            
            G_loss_V = G_loss_V1 + G_loss_V2
            G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V
            
            E_loss_T0 = tf.keras.losses.MeanSquaredError()(X, X_tilde)
            E_loss0 = 10 * tf.sqrt(E_loss_T0)
            E_loss = E_loss0 + 0.1 * G_loss_S
        
        if train_type == "embedder":
          encoder_and_recovery_vars = model.embedder.trainable_variables + model.recovery.trainable_variables
          gradients = tape.gradient(E_loss0, encoder_and_recovery_vars)  
          optimizer_emb.apply_gradients(zip(gradients, encoder_and_recovery_vars))
          # optimizer.apply_gradients(zip(gradients, encoder_and_recovery_vars))
          step_e_loss = E_loss_T0 # take without scaling and square root 
          return step_e_loss
        

        if train_type == "supervised":
          supervised_vars = model.generator.trainable_variables + model.supervisor.trainable_variables
          gradients = tape.gradient(G_loss_S, supervised_vars)
          optimizer_sup.apply_gradients(zip(gradients, supervised_vars))
          # optimizer.apply_gradients(zip(gradients, supervised_vars))
          step_g_loss_s = G_loss_S
          return step_g_loss_s

        
        if train_type == "generator":
          # train generator 
          generator_vars = model.generator.trainable_variables + model.supervisor.trainable_variables
          # for kk in range(2): # train generator twice more than discriminator
          step_g_loss_u, step_g_loss_s, step_g_loss_v = G_loss_U, G_loss_S, G_loss_V
          # gradients = tape.gradient(G_loss, generator_vars)
          # optimizer_gen.apply_gradients(zip(gradients, generator_vars))
          # optimizer.apply_gradients(zip(gradients, generator_vars))

          encoder_and_recovery_vars = model.embedder.trainable_variables + model.recovery.trainable_variables
          # gradients = tape2.gradient(E_loss, encoder_and_recovery_vars)  # has some supervised loss in here
          step_e_loss_t0 = E_loss_T0
          gradients = tape.gradient([G_loss, E_loss], generator_vars + encoder_and_recovery_vars)
          optimizer_joint_emb.apply_gradients(zip(gradients, generator_vars+ encoder_and_recovery_vars))
          # optimizer.apply_gradients(zip(gradients, encoder_and_recovery_vars))
          return step_g_loss_u, step_g_loss_s, step_g_loss_v, step_e_loss_t0


        if train_type == "discriminator":
          check_d_loss = D_loss
          step_d_loss = D_loss 
          # Train discriminator (only when the discriminator does not work well)
          if (check_d_loss > 0.15): 
            discrim_vars = model.discriminator.trainable_variables
            gradients = tape.gradient(D_loss, discrim_vars)
            optimizer_disc.apply_gradients(zip(gradients, discrim_vars))
            # optimizer.apply_gradients(zip(gradients, discrim_vars))

          return step_d_loss
            
     
    
    # Training loop

    # 1. Embedding Training
    for it in range(iterations):
      X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
      Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
      X_mb, T_mb, Z_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32), tf.convert_to_tensor(T_mb, dtype=tf.float32), tf.convert_to_tensor(Z_mb, dtype=tf.float32)
      step_e_loss = train_step(X_mb, Z_mb, T_mb, it, train_type='embedder')
      step_e_loss = step_e_loss.numpy()
      if it % 10 == 0:
        print('step: '+ str(it) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)) )
        model.save_weights(f'weights/embedding_{it}.weights.h5')
      # wandb.log({"step_e_loss": step_e_loss}) 
      
      

    # 2. Supervised Training
    for it in range(iterations):
      X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
      Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
      X_mb, T_mb, Z_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32), tf.convert_to_tensor(T_mb, dtype=tf.float32), tf.convert_to_tensor(Z_mb, dtype=tf.float32)
      step_g_loss_s = train_step(X_mb, Z_mb, T_mb, it, train_type='supervised')
      step_g_loss_s = step_g_loss_s.numpy()
      if it % 10 == 0:
        print('step: '+ str(it)  + '/' + str(iterations) +', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s),4)) )
        model.save_weights(f"weights/supervised_{it}.weights.h5")
      # wandb.log({"step_g_loss_s": step_g_loss_s})
      

    # 3. Joint Training
    for it in range(iterations):
      for kk in range(2): # train generator twice as discrim
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        X_mb, T_mb, Z_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32), tf.convert_to_tensor(T_mb, dtype=tf.float32), tf.convert_to_tensor(Z_mb, dtype=tf.float32)
        step_g_loss_u, step_g_loss_s, step_g_loss_v, step_e_loss_t0 = train_step(X_mb, Z_mb, T_mb, it, train_type='generator')
        step_g_loss_u, step_g_loss_s, step_g_loss_v, step_e_loss_t0 = step_g_loss_u.numpy(), step_g_loss_s.numpy(), step_g_loss_v.numpy(), step_e_loss_t0.numpy()
        # wandb.log({"step_g_loss_u": step_g_loss_u, "step_g_loss_s": step_g_loss_s,"step_g_loss_v": step_g_loss_v, "step_e_loss_t0": step_e_loss_t0})

      X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
      Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
      X_mb, T_mb, Z_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32), tf.convert_to_tensor(T_mb, dtype=tf.float32), tf.convert_to_tensor(Z_mb, dtype=tf.float32)
      step_d_loss= train_step(X_mb, Z_mb, T_mb, it, train_type='discriminator')
      step_d_loss = step_d_loss.numpy()
      # wandb.log({"step_d_loss": step_d_loss})

      if it % 10 == 0:
        print('step: '+ str(it) + '/' + str(iterations) + 
              ', d_loss: ' + str(np.round(step_d_loss,4)) + 
              ', g_loss_u: ' + str(np.round(step_g_loss_u,4)) + 
              ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s),4)) + 
              ', g_loss_v: ' + str(np.round(step_g_loss_v,4)) + 
              ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0),4))  )
        model.save_weights(f'weights/joint_{it}.weights.h5')

    print('Finish Joint Training')


      
        # if it % 100 == 0:
        #     print(f'Iteration {it}, D Loss: {D_loss_curr:.4f}, G Loss: {G_loss_curr:.4f}, E Loss: {E_loss_curr:.4f}')
    
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)
    generated_data = model.generator.predict(Z_mb)
    generated_data = model.supervisor.predict(generated_data)
    generated_data = model.recovery.predict(generated_data)
    
    return generated_data