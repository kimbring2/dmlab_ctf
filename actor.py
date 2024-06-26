import collections
import zmq
import gym
import numpy as np
import statistics
import glob
import random
import tensorflow as tf
import cv2
import argparse
import os
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
socket.connect("tcp://localhost:" + str(6555 + arguments.env_id))

# Create a new environment object.
width = 640
height = 640
fps = 15
action_repeat = int(60 / fps)
env = deepmind_lab.Lab("ctf_simple", ['RGB_INTERLEAVED', 'DEBUG.GADGET_AMOUNT', 'DEBUG.GADGET', 'DEBUG.HAS_RED_FLAG'],
                        {'fps': str(fps), 'width': str(width), 'height': str(height)})
#env = deepmind_lab.Lab("contributed/dmlab30/explore_goal_locations_small", ['RGB_INTERLEAVED'],
#                       {'fps': str(fps), 'width': str(width), 'height': str(height)})

def _action(*entries):
    return np.array(entries, dtype=np.intc)


ACTION_LIST = []
for look_horizontal in [-10, -60, 0, 10, 60]:
    for look_vertical in [-5, 0, 5]:
        for strafe_horizontal in [-1, 0, 1]:
            for strafe_vertical in [-1, 0, 1]:
                for fire in [0, 1]:
                    for jump in [0, 1]:
                        ACTION_LIST.append(_action(look_horizontal, look_vertical, strafe_horizontal, strafe_vertical, fire, jump, 0))



exp_name = 'kill'
if exp_name == 'kill':
    REWARDS = {
        'PICKUP_REWARD': 0,
        'PICKUP_GOAL': 0,
        'TARGET_SCORE': 0,
        'TAG_SELF': 0,
        'TAG_PLAYER': 1,
        'CTF_FLAG_BONUS': 0,
        'CTF_CAPTURE_BONUS': 0,
        'CTF_TEAM_BONUS': 0,
        'CTF_FRAG_CARRIER_BONUS': 0,
        'CTF_RECOVERY_BONUS': 0,
        'CTF_CARRIER_DANGER_PROTECT_BONUS': 0,
        'CTF_FLAG_DEFENSE_BONUS': 0,
        'CTF_CARRIER_PROTECT_BONUS': 0,
        'CTF_RETURN_FLAG_ASSIST_BONUS': 0,
        'CTF_FRAG_CARRIER_ASSIST_BONUS': 0
    }
elif exp_name == 'flag':
    REWARDS = {
        'PICKUP_REWARD': 0,
        'PICKUP_GOAL': 0,
        'TARGET_SCORE': 0,
        'TAG_SELF': 0,
        'TAG_PLAYER': 0,
        'CTF_FLAG_BONUS': 0,
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


#ACTION_LIST = list(ACTIONS)
#print("ACTION_LIST: ", ACTION_LIST)

num_actions = len(ACTION_LIST)
screen_size = (128, 128, 3)


def render(obs, name):
    cv2.imshow('name', obs)
    cv2.waitKey(1)


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a])


scores = []
episodes = []
average = []
for episode_step in range(0, 2000000):
    env.reset(seed=arguments.env_id)

    obs = env.observations()

    obs_screen = obs['RGB_INTERLEAVED']
    obs_screen = cv2.cvtColor(obs_screen, cv2.COLOR_BGR2RGB)
    obs_screen = cv2.resize(obs_screen, dsize=(128, 128), interpolation=cv2.INTER_AREA)

    obs_weapon_amount = obs['DEBUG.GADGET_AMOUNT']
    obs_weapon = obs['DEBUG.GADGET']
    obs_has_red_flag = obs['DEBUG.HAS_RED_FLAG']

    state_screen = obs_screen / 255.0

    has_red_flag_onehot = one_hot(int(obs_has_red_flag), 2) 
    obs_inv = np.concatenate([obs_weapon_amount / 100.0, has_red_flag_onehot])
    state_inv = obs_inv

    done = False
    reward = 0.0
    reward_sum = 0
    step = 0
    while True:
        try:
            print("step: ", step)
            state_screen_reshaped = np.reshape(state_screen, (1,*screen_size))
            state_inv_reshaped = np.reshape(state_inv, (1, 3))
            #state_inv_reshaped = np.zeros((1, 3))

            env_output = {"env_id": np.array([arguments.env_id]), 
                          "reward": reward, "done": done, 
                          "obs_screen": state_screen_reshaped,
                          "obs_inv": state_inv_reshaped}
            socket.send_pyobj(env_output)
            action_index = int(socket.recv_pyobj()['action'])

            reward_game = env.step(ACTION_LIST[action_index], num_steps=action_repeat)
            #reward = reward_game
            
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

                        #print("reason: {0}, team: {1}, score: {2}, reward: {3}".format(reason, team, score, reward))
            
            done = not env.is_running()
            if done or step == 500:
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

            obs1_screen = obs1['RGB_INTERLEAVED']
            obs1_screen = cv2.cvtColor(obs1_screen, cv2.COLOR_BGR2RGB)

            #if env_id == 0: 
            #    render(obs1_screen, "obs1_screen")

            obs1_screen = cv2.resize(obs1_screen, dsize=(128,128), interpolation=cv2.INTER_AREA)

            obs1_weapon_amount = obs1['DEBUG.GADGET_AMOUNT']
            obs1_weapon = obs1['DEBUG.GADGET']
            obs1_has_red_flag = obs1['DEBUG.HAS_RED_FLAG']

            #print("obs1_weapon_amount: ", obs1_weapon_amount)
            #print("obs1_weapon: ", obs1_weapon)
            #print("obs1_has_red_flag: ", obs1_has_red_flag)
            #print("")

            next_state_screen = obs1_screen / 255.0

            has_red_flag_onehot1 = one_hot(int(obs1_has_red_flag), 2) 
            obs1_inv = np.concatenate([obs1_weapon_amount / 100.0, has_red_flag_onehot1])
            next_state_inv = obs1_inv

            reward_sum += reward
            state_screen = next_state_screen
            state_inv = next_state_inv

            step += 1
        except (tf.errors.UnavailableError, tf.errors.CancelledError):
            print('Inference call failed. This is normal at the end of training.')
            #logging.info('Inference call failed. This is normal at the end of training.')

env.close()
