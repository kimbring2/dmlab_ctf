import collections
import zmq
import gym
import numpy as np
import statistics
import tqdm
import glob
import random
import tensorflow as tf
import cv2
import argparse
import os
from matplotlib import pyplot as plt
from typing import Any, List, Sequence, Tuple
from absl import flags
import deepmind_lab

parser = argparse.ArgumentParser(description='CTF IMPALA Actor')
parser.add_argument('--env_id', type=int, default=0, help='ID of environment')
arguments = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

writer = tf.summary.create_file_writer("tensorboard")

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:" + str(5555 + arguments.env_id))

# Create a new environment object.
env = deepmind_lab.Lab("ctf_simple", ['RGB_INTERLEAVED'],
                       {'fps': '30', 'width': '80', 'height': '60'})

def _action(*entries):
  return np.array(entries, dtype=np.intc)

ACTIONS = {
      'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
      'look_right': _action(20, 0, 0, 0, 0, 0, 0),
      'look_up': _action(0, 10, 0, 0, 0, 0, 0),
      'look_down': _action(0, -10, 0, 0, 0, 0, 0),
      'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
      'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
      'forward': _action(0, 0, 0, 1, 0, 0, 0),
      'backward': _action(0, 0, 0, -1, 0, 0, 0),
      'fire': _action(0, 0, 0, 0, 1, 0, 0),
      'jump': _action(0, 0, 0, 0, 0, 1, 0),
      'crouch': _action(0, 0, 0, 0, 0, 0, 1)
  }


ACTION_LIST = list(ACTIONS)
#print("ACTION_LIST: ", ACTION_LIST)

num_actions = 11
state_size = (80,80,3)


def render(obs):
    cv2.imshow('obs', obs)
    cv2.waitKey(1)


scores = []
episodes = []
average = []
for episode_step in range(0, 2000000):
    env.reset(seed=1)
    obs = env.observations()
    obs = cv2.cvtColor(obs['RGB_INTERLEAVED'], cv2.COLOR_BGR2RGB)

    obs_resized = cv2.resize(obs, dsize=(80,80), interpolation=cv2.INTER_AREA)
    state = obs_resized / 255.0

    done = False
    reward = 0.0
    reward_sum = 0
    step = 0
    while True:
        try:
            step += 1

            state_reshaped = np.reshape(state, (1,*state_size)) 

            env_output = {"env_id": np.array([arguments.env_id]), 
                          "reward": reward,
                          "done": done, 
                          "observation": state_reshaped}
            socket.send_pyobj(env_output)
            action = int(socket.recv_pyobj()['action'])
            
            #obs1, reward, done, _ = env.step(ACTION_LIST[action])
            reward = env.step(ACTIONS[ACTION_LIST[action]])
            obs1 = env.observations()
            done = not env.is_running()

            obs1 = cv2.cvtColor(obs1['RGB_INTERLEAVED'], cv2.COLOR_BGR2RGB)
            obs1_resized = cv2.resize(obs1, dsize=(80,80), interpolation=cv2.INTER_AREA)
            next_state = obs1_resized / 255.0

            if arguments.env_id == 0: 
                render(obs1_resized)

            reward_sum += reward
            state = next_state
            if done:
                if arguments.env_id == 0:
                    scores.append(reward_sum)
                    episodes.append(episode_step)
                    average.append(sum(scores[-50:]) / len(scores[-50:]))

                    with writer.as_default():
                        #print("average[-1]: ", average[-1])
                        tf.summary.scalar("average_reward", average[-1], step=episode_step)
                        writer.flush()

                    print("average_reward: " + str(average[-1]))
                else:
                    print("reward_sum: " + str(reward_sum))

                break

        except (tf.errors.UnavailableError, tf.errors.CancelledError):
            logging.info('Inference call failed. This is normal at the end of training.')

env.close()
