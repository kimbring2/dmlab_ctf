import tensorflow as tf
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
import numpy as np

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)

    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(84, 84, 3)),
                layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                layers.Flatten(),
                layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(latent_dim,)),
                layers.Dense(units=21*21*32, activation=tf.nn.relu),
                layers.Reshape(target_shape=(21, 21, 32)),
                layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same')
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
    
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
    
        return logits



class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""
  def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.num_actions = num_actions
    
    latent_dim = int(256)
    self.CVAE = CVAE(latent_dim)

    self.lstm = layers.LSTM(128, return_sequences=True, return_state=True)
    
    self.common = layers.Dense(num_hidden_units, activation="relu")
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'num_actions': self.num_actions,
        'num_hidden_units': self.num_hidden_units
    })

    return config
    
  def call(self, inputs: tf.Tensor, memory_state: tf.Tensor, carry_state: tf.Tensor, training) -> Tuple[tf.Tensor, tf.Tensor, 
                                                                                                        tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(inputs)[0]

    #tf.print("inputs.shape: ", inputs.shape)

    mean, logvar = self.CVAE.encode(inputs)
    cvae_output = tf.concat((mean, logvar), axis=1)
    cvae_output_reshaped = layers.Reshape((16,32))(cvae_output)

    initial_state = (memory_state, carry_state)
    lstm_output, final_memory_state, final_carry_state  = self.lstm(cvae_output_reshaped, initial_state=initial_state, 
                                                                    training=training)
    X_input = layers.Flatten()(lstm_output)

    #tf.print("X_input.shape: ", X_input.shape)
    x = self.common(X_input)

    #tf.print("")

    z = self.CVAE.reparameterize(mean, logvar)
    x_logit = self.CVAE.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=inputs)
    
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    cvae_loss = logpx_z + logpz - logqz_x
    
    return self.actor(x), self.critic(x), final_memory_state, final_carry_state, cvae_loss