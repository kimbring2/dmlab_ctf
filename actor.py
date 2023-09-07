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

env_id = arguments.env_id

if env_id == 0:
    writer = tf.summary.create_file_writer("tensorboard_actor")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:" + str(5555 + env_id))

# Create a new environment object.
env = deepmind_lab.Lab("ctf_simple", ['RGB_INTERLEAVED'],
                       {'fps': '30', 'width': '640', 'height': '640'})
#env = deepmind_lab.Lab("contributed/dmlab30/explore_goal_locations_small", ['RGB_INTERLEAVED'],
#                       {'fps': '60', 'width': '640', 'height': '640'})

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


REWARDS = {
    'PICKUP_REWARD': 0,
    'PICKUP_GOAL': 0,
    'TARGET_SCORE': 0,
    'TAG_SELF': 0,
    'TAG_PLAYER': 0,
    'CTF_FLAG_BONUS': 1,
    'CTF_CAPTURE_BONUS': 1,
    'CTF_TEAM_BONUS': 0,
    'CTF_FRAG_CARRIER_BONUS': 0,
    'CTF_RECOVERY_BONUS': 0,
    'CTF_CARRIER_DANGER_PROTECT_BONUS': 0,
    'CTF_FLAG_DEFENSE_BONUS': 0,
    'CTF_CARRIER_PROTECT_BONUS': 0,
    'CTF_RETURN_FLAG_ASSIST_BONUS': 0,
    'CTF_FRAG_CARRIER_ASSIST_BONUS': 0
}


ACTION_LIST = list(ACTIONS)
#print("ACTION_LIST: ", ACTION_LIST)

num_actions = 7
state_size = (84,84,3)


def render(obs, name):
    cv2.imshow('name', obs)
    cv2.waitKey(1)


scores = []
episodes = []
average = []
for episode_step in range(0, 2000000):
    env.reset(seed=arguments.env_id)
    obs = env.observations()
    obs_raw = cv2.cvtColor(obs['RGB_INTERLEAVED'], cv2.COLOR_BGR2RGB)
    obs = cv2.resize(obs_raw, dsize=(84,84), interpolation=cv2.INTER_AREA)

    state = obs / 255.0

    done = False
    reward = 0.0
    reward_sum = 0
    step = 0
    while True:
        try:
            #print("step: ", step)
            state_reshaped = np.reshape(state, (1,*state_size)) 

            env_output = {"env_id": np.array([arguments.env_id]), 
                          "reward": reward,
                          "done": done, 
                          "observation": state_reshaped}
            socket.send_pyobj(env_output)
            action = int(socket.recv_pyobj()['action'])

            reward_game = env.step(ACTIONS[ACTION_LIST[action]], num_steps=2)

            reward = 0
            events = env.events()
            if len(events) != 0:
                for event in events:
                    if event[0] == 'reward':
                        event_info = event[1]

                        reason = event_info[0]
                        team = event_info[1]
                        score = event_info[2]
                        player_id = event_info[3]
                        location = event_info[4]
                        other_player_id = event_info[5]

                        if team == 'blue':
                            reward = float(REWARDS[reason])

                        print("reason: {0}, team: {1}, score: {2}, reward: {3}".format(reason, team, score, reward))

            done = not env.is_running()
            if done or step == 1000:
                if env_id == 0:
                    scores.append(reward_sum)
                    episodes.append(episode_step)
                    average.append(sum(scores[-50:]) / len(scores[-50:]))

                    with writer.as_default():
                        tf.summary.scalar("average_reward", average[-1], step=episode_step)
                        writer.flush()

                    #print("average_reward: " + str(average[-1]))
                else:
                    #print("reward_sum: " + str(reward_sum))
                    pass

                break

            obs1 = env.observations()
            obs1_raw = cv2.cvtColor(obs1['RGB_INTERLEAVED'], cv2.COLOR_BGR2RGB)
            obs1 = cv2.resize(obs1_raw, dsize=(84,84), interpolation=cv2.INTER_AREA)
            next_state = obs1 / 255.0

            #if env_id == 0: 
            #    render(obs1_raw, "obs")

            reward_sum += reward
            state = next_state

            step += 1
        except (tf.errors.UnavailableError, tf.errors.CancelledError):
            print('Inference call failed. This is normal at the end of training.')
            logging.info('Inference call failed. This is normal at the end of training.')

env.close()
