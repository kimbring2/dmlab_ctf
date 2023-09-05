import collections
import gym
import retro
import numpy as np
import statistics
import tensorflow as tf
import tqdm
import glob
import random
import cv2
import tensorflow_probability as tfp
import argparse
import os
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
from absl import flags
import network
import deepmind_lab

tfd = tfp.distributions

parser = argparse.ArgumentParser(description='Ctf Evaluation')

parser.add_argument('--workspace_path', type=str, help='root directory of project')
parser.add_argument('--gpu_use', action='store_true', default=False)
parser.add_argument('--model_name', type=str, help='name of saved model')

arguments = parser.parse_args()

workspace_path = arguments.workspace_path
gpu_use = arguments.gpu_use
model_name = arguments.model_name

if gpu_use == True:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def _action(*entries):
  return np.array(entries, dtype=np.intc)


ACTIONS = {
      'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
      'look_right': _action(20, 0, 0, 0, 0, 0, 0),
      'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
      'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
      'forward': _action(0, 0, 0, 1, 0, 0, 0),
      'backward': _action(0, 0, 0, -1, 0, 0, 0),
      'fire': _action(0, 0, 0, 0, 1, 0, 0),
  }

ACTION_LIST = list(ACTIONS)


num_actions = len(ACTION_LIST)
num_hidden_units = 512

model = network.ActorCritic(num_actions, num_hidden_units)

if model_name:
    model.load_weights(workspace_path + '/model/' + model_name)

seed = 980
tf.random.set_seed(seed)
np.random.seed(seed)

def render(obs, name):
    cv2.imshow('name', obs)
    cv2.waitKey(1)

state_size = (84,84,3)

reward_sum = 0
for i_episode in range(0, 10000):
    # Create a new environment object.
    env = deepmind_lab.Lab("ctf_simple", ['RGB_INTERLEAVED'], 
                            {'fps': '30', 'width': '640', 'height': '640'})

    env.reset()
    obs = env.observations()
    obs_raw = cv2.cvtColor(obs['RGB_INTERLEAVED'], cv2.COLOR_BGR2RGB)
    obs_resized = cv2.resize(obs_raw, dsize=(84,84), interpolation=cv2.INTER_AREA)
    state = obs_resized / 255.0

    memory_state_obs = tf.zeros([1,128], dtype=np.float32)
    carry_state_obs = tf.zeros([1,128], dtype=np.float32)
    step = 0
    done = False

    action_index = 0
    while True:
        step += 1

        state_reshaped = np.reshape(state, (1, *state_size))
        #print("state_reshaped.shape: ", state_reshaped.shape)
        action_probs, _, memory_state_obs, carry_state_obs, _ = model(state_reshaped, memory_state_obs, carry_state_obs, 
                                                                      training=False)
        action_dist = tfd.Categorical(logits=action_probs)

        mean, logvar = model.CVAE.encode(tf.expand_dims(state, 0))
        z = model.CVAE.reparameterize(mean, logvar)
        prediction = model.CVAE.sample(z)
        prediction = np.array(prediction)
        prediction = cv2.resize(prediction[0], dsize=(320,224), interpolation=cv2.INTER_AREA)
        prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
        cv2.imshow("prediction", prediction)
        cv2.waitKey(1)

        action = int(action_dist.sample()[0])
        reward = env.step(ACTIONS[ACTION_LIST[action]], num_steps=2)
        done = not env.is_running()
        
        obs1 = env.observations()
        obs1_raw = cv2.cvtColor(obs1['RGB_INTERLEAVED'], cv2.COLOR_BGR2RGB)
        obs1_resized = cv2.resize(obs1_raw, dsize=(84,84), interpolation=cv2.INTER_AREA)
        next_state = obs1_resized / 255.0

        render(obs1_raw, "obs")

        reward_sum += reward

        pre_state = state
        state = next_state
        if done:
            print("Total reward: {:.2f},  Total step: {:.2f}".format(reward_sum, step))
            step = 0
            reward_sum = 0  
            break

env.close()
