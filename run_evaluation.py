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

parser.add_argument('--exp_name', type=str, default="kill", help='name of experiment')
parser.add_argument('--gpu_use', action='store_true', default=False)
parser.add_argument('--model_name', type=str, help='name of saved model')

arguments = parser.parse_args()

exp_name = arguments.exp_name
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
    if exp_name == 'kill':
        model.load_weights("kill/model/" + model_name)
    elif exp_name == 'flag':
        model.load_weights("flag/model/" + model_name)

seed = 980
tf.random.set_seed(seed)
np.random.seed(seed)

def render(obs, name):
    cv2.imshow('name', obs)
    cv2.waitKey(1)

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a])

screen_size = (64,64,3)

width = 320
height = 320
fps = 15
action_repeat = int(60 / fps)

# Create a new environment object.
env = deepmind_lab.Lab("ctf_simple", ['RGB_INTERLEAVED', 'DEBUG.GADGET_AMOUNT', 'DEBUG.GADGET', 'DEBUG.HAS_RED_FLAG'],
                       {'fps': str(fps), 'width': str(width), 'height': str(height)})

reward_sum = 0
for i_episode in range(0, 10000):
    env.reset()

    obs = env.observations()

    obs_screen = obs['RGB_INTERLEAVED']
    obs_screen = cv2.cvtColor(obs_screen, cv2.COLOR_BGR2RGB)
    obs_screen = cv2.resize(obs_screen, dsize=(64,64), interpolation=cv2.INTER_AREA)

    obs_weapon_amount = obs['DEBUG.GADGET_AMOUNT']
    obs_weapon = obs['DEBUG.GADGET']
    obs_has_red_flag = obs['DEBUG.HAS_RED_FLAG']

    state_screen = obs_screen / 255.0

    has_red_flag_onehot = one_hot(int(obs_has_red_flag), 2) 
    obs_inv = np.concatenate([obs_weapon_amount / 100.0, has_red_flag_onehot])
    state_inv = obs_inv

    memory_state_obs = tf.zeros([1,128], dtype=np.float32)
    carry_state_obs = tf.zeros([1,128], dtype=np.float32)
    step = 0
    done = False

    action_index = 0
    while True:
        step += 1

        state_screen_reshaped = np.reshape(state_screen, (1,*screen_size))
        state_inv_reshaped = np.reshape(state_inv, (1, 3))

        action_probs, _, memory_state_obs, carry_state_obs, _ = model(state_screen_reshaped, state_inv_reshaped,
                                                                      memory_state_obs, carry_state_obs, 
                                                                      training=False)
        action_dist = tfd.Categorical(logits=action_probs)

        mean, logvar = model.CVAE.encode(state_screen_reshaped)
        z = model.CVAE.reparameterize(mean, logvar)
        prediction = model.CVAE.sample(z)
        prediction = np.array(prediction)
        prediction = cv2.resize(prediction[0], dsize=(320,224), interpolation=cv2.INTER_AREA)
        prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
        cv2.imshow("prediction", prediction)
        cv2.waitKey(1)

        action = int(action_dist.sample()[0])
        reward = env.step(ACTIONS[ACTION_LIST[action]], num_steps=action_repeat)
        done = not env.is_running()
        
        obs1 = env.observations()

        obs1_screen = obs1['RGB_INTERLEAVED']
        obs1_screen = cv2.cvtColor(obs1_screen, cv2.COLOR_BGR2RGB)
        obs1_screen = cv2.resize(obs1_screen, dsize=(64,64), interpolation=cv2.INTER_AREA)

        obs1_weapon_amount = obs1['DEBUG.GADGET_AMOUNT']
        obs1_weapon = obs1['DEBUG.GADGET']
        obs1_has_red_flag = obs1['DEBUG.HAS_RED_FLAG']

        next_state_screen = obs1_screen / 255.0

        has_red_flag_onehot1 = one_hot(int(obs1_has_red_flag), 2) 
        obs1_inv = np.concatenate([obs1_weapon_amount / 100.0, has_red_flag_onehot1])
        next_state_inv = obs1_inv

        print("next_state_screen.shape: ", next_state_screen.shape)
        cv2.imshow('state_screen', state_screen)
        cv2.waitKey(1)

        #render(obs1_raw, "obs")

        reward_sum += reward
        state_screen = next_state_screen
        state_inv = next_state_inv

        if done or step == 250:
            print("Total reward: {:.2f},  Total step: {:.2f}".format(reward_sum, step))
            step = 0
            reward_sum = 0  
            break

        #print("step: ", step)

env.close()
