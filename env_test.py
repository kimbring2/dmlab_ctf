import deepmind_lab
import numpy as np
import cv2
import random

print("deepmind_lab.__file__: ", deepmind_lab.__file__)

## Create a new environment object.
#env = deepmind_lab.Lab("ctf_simple", ['RGB_INTERLEAVED', 'DEBUG.GADGET_AMOUNT', 'DEBUG.GADGET', 'DEBUG.HAS_RED_FLAG'],
#                       {'fps': '30', 'width': '640', 'height': '640'})

width = 640
height = 640
fps = 15
action_repeat = int(60 / fps)
env = deepmind_lab.Lab("ctf_middle", ['RGB_INTERLEAVED', 'DEBUG.GADGET_AMOUNT', 'DEBUG.GADGET', 'DEBUG.HAS_RED_FLAG'],
                           {'fps': str(fps), 'width': str(width), 'height': str(height)})
#env = deepmind_lab.Lab("contributed/dmlab30/explore_goal_locations_small", ['RGB_INTERLEAVED'],
#                       {'fps': str(fps), 'width': str(width), 'height': str(height)})

#env = deepmind_lab.Lab("tests/event_test", ['RGB_INTERLEAVED'],
#                       {'fps': '30', 'width': '640', 'height': '640'})

def _action(*entries):
  return np.array(entries, dtype=np.intc)

'''
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
 '''

ACTION_LIST = []
for look_horizontal in [-10, -60, 0, 10, 60]:
    for look_vertical in [-5, 0, 5]:
        for strafe_horizontal in [-1, 0, 1]:
            for strafe_vertical in [-1, 0, 1]:
                for fire in [0, 1]:
                    for jump in [0, 1]:
                        ACTION_LIST.append(_action(look_horizontal, look_vertical, strafe_horizontal, strafe_vertical, fire, jump, 0))


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
		#print("obs.keys(): ", obs.keys())
		#print("obs['RGB_INTERLEAVED'].shape: ", obs['RGB_INTERLEAVED'].shape)
		#print("obs['DEBUG.GADGET_AMOUNT'].shape: ", obs['DEBUG.GADGET_AMOUNT'].shape)
		#print("obs['DEBUG.GADGET'].shape: ", obs['DEBUG.GADGET'].shape)
		#print("obs['DEBUG.HAS_RED_FLAG'].shape: ", obs['DEBUG.HAS_RED_FLAG'].shape)

		obs_raw = cv2.cvtColor(obs['RGB_INTERLEAVED'], cv2.COLOR_BGR2RGB)
		obs = cv2.resize(obs_raw, dsize=(512,512), interpolation=cv2.INTER_AREA)

		step = 0
		while True:
			#print("step: ", step)

			#print("done: ", done)
			render(obs_raw)

			action_index = random.randint(0, len(ACTION_LIST) - 1)
			reward = env.step(ACTION_LIST[action_index], num_steps=4)

			done = not env.is_running()
			if done:
				print("step: " , step)
				print('Environment stopped early')
				break

			obs = env.observations()

			#print("obs['RGB_INTERLEAVED'].shape: ", obs['RGB_INTERLEAVED'].shape)
			#print("obs['DEBUG.GADGET_AMOUNT']: ", obs['DEBUG.GADGET_AMOUNT'])
			#print("obs['DEBUG.GADGET']: ", obs['DEBUG.GADGET'])
			#print("obs['DEBUG.HAS_RED_FLAG']: ", obs['DEBUG.HAS_RED_FLAG'])

			#print("obs: ", obs)
			#print("obs['RGB_INTERLEAVED'].shape: ", obs['RGB_INTERLEAVED'].shape)
			obs_raw = cv2.cvtColor(obs['RGB_INTERLEAVED'], cv2.COLOR_BGR2RGB)
			obs = cv2.resize(obs_raw, dsize=(512,512), interpolation=cv2.INTER_AREA)
			step += 1

	env.close()
