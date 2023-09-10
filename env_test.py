import deepmind_lab
import numpy as np
import cv2
import random

print("deepmind_lab.__file__: ", deepmind_lab.__file__)

## Create a new environment object.
env = deepmind_lab.Lab("ctf_simple", ['RGB_INTERLEAVED'],
                       {'fps': '30', 'width': '640', 'height': '640'})

#env = deepmind_lab.Lab("contributed/dmlab30/lasertag_one_opponent_small", ['RGB_INTERLEAVED'],
#                       {'fps': '30', 'width': '640', 'height': '640'})

#env = deepmind_lab.Lab("tests/event_test", ['RGB_INTERLEAVED'],
#                       {'fps': '30', 'width': '640', 'height': '640'})

def _action(*entries):
  return np.array(entries, dtype=np.intc)

ACTIONS = {
      'noop': _action(0, 0, 0, 0, 0, 0, 0),
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
print("ACTION_LIST: ", ACTION_LIST)

def render(obs):
    cv2.imshow('obs', obs)
    cv2.waitKey(1)


if __name__ == '__main__':
	num_states = (80, 60)
	num_actions = len(ACTION_LIST)

	print("num_states: ", num_states)
	print("num_actions: ", num_actions)

	reward_sum = 0
	for episode in range(1000):
		print("episode: ", episode)

		env.reset(seed=1)
		obs = env.observations()
		obs_raw = cv2.cvtColor(obs['RGB_INTERLEAVED'], cv2.COLOR_BGR2RGB)
		obs = cv2.resize(obs_raw, dsize=(84,84), interpolation=cv2.INTER_AREA)

		step = 0
		while True:
			print("step: ", step)

			done = not env.is_running()
			#print("done: ", done)
			render(obs_raw)

			if done:
				print('Environment stopped early')
				break

			action = random.randint(0, len(ACTION_LIST) - 1)
			reward = env.step(ACTIONS[ACTION_LIST[0]], num_steps=4)
			obs = env.observations()
			#print("obs: ", obs)
			#print("obs['RGB_INTERLEAVED'].shape: ", obs['RGB_INTERLEAVED'].shape)
			obs_raw = cv2.cvtColor(obs['RGB_INTERLEAVED'], cv2.COLOR_BGR2RGB)
			obs = cv2.resize(obs_raw, dsize=(84,84), interpolation=cv2.INTER_AREA)

			step += 1

	env.close()
