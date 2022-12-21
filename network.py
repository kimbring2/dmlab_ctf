import tensorflow as tf
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""
  def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.num_actions = num_actions
    
    #self.conv_1 = layers.Conv2D(16, 8, 4, padding="valid", activation="relu", kernel_regularizer='l2')
    #self.conv_2 = layers.Conv2D(32, 4, 2, padding="valid", activation="relu", kernel_regularizer='l2')
    #self.conv_3 = layers.Conv2D(64, 3, 1, padding="valid", activation="relu", kernel_regularizer='l2')
    self.dense_1 = layers.Dense(512, activation='relu')

    self.lstm = layers.LSTM(64, return_sequences=True, return_state=True)
    
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

    X_input = layers.Flatten()(inputs)
    #conv_1 = self.conv_1(inputs)
    #conv_2 = self.conv_2(conv_1)
    #conv_3 = self.conv_3(conv_2)
    dense_1 = self.dense_1(X_input)

    #conv_3_reshaped = layers.Reshape((64,16))(conv_3)
    dense_1_reshaped = layers.Reshape((32,16))(dense_1)

    initial_state = (memory_state, carry_state)
    lstm_output, final_memory_state, final_carry_state  = self.lstm(dense_1_reshaped, initial_state=initial_state, 
                                                                    training=training)
    
    X_input = layers.Flatten()(lstm_output)
    x = self.common(X_input)
    
    return self.actor(x), self.critic(x), final_memory_state, final_carry_state