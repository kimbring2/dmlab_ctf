import deepmind_lab
import numpy as np
import cv2
import random
import time
import pygame, sys

pygame.init()

gameDisplay = pygame.display.set_mode((720,640))
pygame.display.set_caption("Platypus")

# Create a new environment object.
env = deepmind_lab.Lab("ctf_simple", ['RGB_INTERLEAVED'],
                       {'fps': '60', 'width': '720', 'height': '640'})

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

	for episode in range(1000):
		print("episode: ", episode)

		env.reset(seed=1)
		obs = env.observations()
		obs_raw = cv2.cvtColor(obs['RGB_INTERLEAVED'], cv2.COLOR_BGR2RGB)
		obs = cv2.resize(obs_raw, dsize=(84,84), interpolation=cv2.INTER_AREA)

		step = 0
		while True:
			pygame.event.set_grab(True)

			exit = False
			mouse_move = (0, 0)
			keyboard_move = [0, 0, 0, 0]
			weapon_fire = 0
			jump = 0
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					exit = True

				if event.type == pygame.MOUSEMOTION:
					mouse_move = event.rel

				left, middle, right = pygame.mouse.get_pressed()
				if left:
					weapon_fire = 1

				if event.type == pygame.KEYDOWN:
					if event.key == pygame.K_ESCAPE:
						exit = True

					if event.key == pygame.K_SPACE:
						print("jump")
						jump = 1

				keys = pygame.key.get_pressed()
				if keys[pygame.K_w]:
					#print("w")
					keyboard_move[0] = 1

				if keys[pygame.K_a]:
					#print("a")
					keyboard_move[1] = 1

				if keys[pygame.K_s]:
					#print("s")
					keyboard_move[2] = 1

				if keys[pygame.K_d]:
					#print("d")
					keyboard_move[3] = 1

				if keys[pygame.K_h]:
					#print("d")
					weapon_fire = 1

			if exit:
				quit()

			action_look_horizontal = 0
			action_look_vertical = 0
			action_strafe_horizontal = 0
			action_forward_backward = 0
			action_fire = 0
			action_jump = 0
			action_crouch = 0

			#print("mouse_move: ", mouse_move)
			if mouse_move[0] < 0:
				action_look_horizontal = -1 * mouse_move[0]
			elif mouse_move[0] > 0:
				action_look_horizontal = -1 * mouse_move[0]

			if mouse_move[1] < 0:
				action_look_vertical = -1 * mouse_move[1]
			elif mouse_move[1] > 0:
				action_look_vertical = -1 * mouse_move[1]	 

			if keyboard_move[0] == 1:
				action_forward_backward = 1
			elif keyboard_move[1] == 1:
				action_strafe_horizontal = -1
			elif keyboard_move[2] == 1:
				action_forward_backward = -1
			elif keyboard_move[3] == 1:
				action_strafe_horizontal = 1

			action = _action(action_look_horizontal, 
							 action_look_vertical, 
							 action_strafe_horizontal, 
							 action_forward_backward, 
							 weapon_fire, 
							 jump, 
							 0)
			reward = env.step(action, num_steps=1)

			done = not env.is_running()
			if done:
				print('Environment stopped early')
				break

			obs = env.observations()
			#print("obs['RGB_INTERLEAVED'].shape: ", obs['RGB_INTERLEAVED'].shape)
			#obs_raw = cv2.cvtColor(obs['RGB_INTERLEAVED'], cv2.COLOR_BGR2RGB)
			obs = obs['RGB_INTERLEAVED']
			obs = cv2.rotate(obs, cv2.ROTATE_90_COUNTERCLOCKWISE)
			obs = cv2.flip(obs, 0)

			#obs = cv2.resize(obs_raw, dsize=(84,84), interpolation=cv2.INTER_AREA)
			#obs = obs_raw

			#surf = pygame.surfarray.make_surface(obs)
			surf = pygame.surfarray.make_surface(obs)
			gameDisplay.blit(surf, (0, 0))
			pygame.display.update()

			pygame.mouse.set_pos([720 / 2, 640 / 2])
			step += 1

	env.close()
