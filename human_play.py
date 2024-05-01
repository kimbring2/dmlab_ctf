import deepmind_lab
import numpy as np
import cv2
import random
import time
import pygame, sys
from datetime import datetime

pygame.init()

width = 640
height = 640

gameDisplay = pygame.display.set_mode((width, height))
pygame.display.set_caption("Platyp us")
pygame.mouse.set_visible(False)

map_name = "ctf_simple"

# Create a new environment object.
# 'DEBUG.HAS_RED_FLAG'

fps = 60
action_repeat = int(60 / fps)

env = deepmind_lab.Lab(map_name, ['RGB_INTERLEAVED', 'DEBUG.GADGET_AMOUNT', 'DEBUG.GADGET', 'DEBUG.HAS_RED_FLAG'],
                           {'fps': str(fps), 'width': str(width), 'height': str(height)})
#env = deepmind_lab.Lab("contributed/dmlab30/explore_goal_locations_small", ['RGB_INTERLEAVED'],
#                       {'fps': '30', 'width': '1280', 'height': '960'})
#env = deepmind_lab.Lab("tests/update_inventory_test", ['RGB_INTERLEAVED', 'DEBUG.AMOUNT', 'DEBUG.GADGET'],
#                       {'fps': '60', 'width': '640', 'height': '640'})

def _action(*entries):
  return np.array(entries, dtype=np.intc)

REWARDS = {
	'PICKUP_REWARD': 0,
	'PICKUP_GOAL': 0,
	'TARGET_SCORE': 0,
	'TAG_SELF': 0,
	'TAG_PLAYER': 0,
	'CTF_FLAG_BONUS': 1,
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

def render(obs):
    cv2.imshow('obs', obs)
    cv2.waitKey(1)



save_path = '/media/kimbring2/be356a87-def6-4be8-bad2-077951f0f3da/cfg_data/'

clock = pygame.time.Clock()


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a])


if __name__ == '__main__':
	num_states = (80, 60)
	print("num_states: ", num_states)
	for episode in range(1000):
		print("episode: ", episode)

		env.reset(seed=1)
		events = env.events()

		obs = env.observations()
		obs_raw = cv2.cvtColor(obs['RGB_INTERLEAVED'], cv2.COLOR_BGR2RGB)
		obs = cv2.resize(obs_raw, dsize=(84, 84), interpolation=cv2.INTER_AREA)

		frame_list = []
		episode_list = []
		action_list = []
		reward_list = []
		done_list = []
		step_list = []
		state_list = []
		next_state_list = []

		step = 0

		now = datetime.now()
		dt_string = now.strftime("%d%m%Y_%H%M%S")
		#print("date and time =", dt_string)
		save_file = dt_string
		#save_file = 'test'
		path_video = save_path + map_name + '_' + save_file + '.avi'
		size = (720, 640)
		video_out = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
		while True:
			#print("step: ", step)

			pygame.event.set_grab(True)

			exit = False
			mouse_move = (0, 0)
			keyboard_move = [0, 0, 0, 0]
			weapon_fire = 0
			jump = 0
			crouch = 0
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					exit = True

				if event.type == pygame.MOUSEMOTION:
					mouse_move = event.rel

				left, middle, right = pygame.mouse.get_pressed()
				if left:
					weapon_fire = 1

				if event.type == pygame.KEYDOWN:
					#print("if event.type == pygame.KEYDOWN")

					if event.key == pygame.K_ESCAPE:
						exit = True

					if event.key == pygame.K_SPACE:
						#print("jump")
						jump = 1

					if event.key == pygame.K_w:
						keyboard_move[0] = 1

					if event.key == pygame.K_a:
						keyboard_move[1] = 1

					if event.key == pygame.K_s:
						keyboard_move[2] = 1

					if event.key == pygame.K_d:
						keyboard_move[3] = 1

			left, middle, right = pygame.mouse.get_pressed()
			if left:
				weapon_fire = 1

			keys = pygame.key.get_pressed()
			if keys[pygame.K_w]:
				keyboard_move[0] = 1

			if keys[pygame.K_a]:
				keyboard_move[1] = 1

			if keys[pygame.K_s]:
				keyboard_move[2] = 1

			if keys[pygame.K_d]:
				keyboard_move[3] = 1

			if keys[pygame.K_z]:
				crouch = 1
				
			if exit:
				print("exit program")

				# Save camera frame as video
				frame_array = np.array(frame_list)
				for i in range(len(frame_array)):
					# writing to a image array
					video_out.write(frame_array[i])

				video_out.release()

				save_data = {'episode': episode, 'step': step_list,
							   'action': action_list, 'reward': reward_list, 'done': done_list}

				path_npy = save_path + save_file + '.npy'
				#np.save(path_npy, save_data)
				
				quit()

			action_look_horizontal = 0
			action_look_vertical = 0
			action_strafe_horizontal = 0
			action_forward_backward = 0

			#print("mouse_move: ", mouse_move)
			if mouse_move[0] < 0:
				action_look_horizontal = 10 * mouse_move[0]
			elif mouse_move[0] > 0:
				action_look_horizontal = 10 * mouse_move[0]

			if mouse_move[1] < 0:
				action_look_vertical = 10 * mouse_move[1]
			elif mouse_move[1] > 0:
				action_look_vertical = 10 * mouse_move[1]	 

			if keyboard_move[0] == 1:
				action_forward_backward = 1
			elif keyboard_move[1] == 1:
				action_strafe_horizontal = -1
			elif keyboard_move[2] == 1:
				action_forward_backward = -1
			elif keyboard_move[3] == 1:
				action_strafe_horizontal = 1

			action = _action(action_look_horizontal, action_look_vertical,  action_strafe_horizontal, action_forward_backward, 
							 	weapon_fire, jump, crouch)
			reward_game = env.step(action, num_steps=action_repeat)
			#print("reward_game: ", reward_game)

			reward = 0
			events = env.events()
			#print("events: ", events)

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

						#print("reason: ", reason)
						#print("team: ", team)
						#print("score: ", score)
						#print("player_id: ", player_id)
						#print("location: ", location)
						#print("other_player_id: ", other_player_id)

						if team == 'blue':
							#print("team == 'blue'")
							reward = REWARDS[reason]

						print("reason: {0}, team: {1}, score: {2}, reward: {3}".format(reason, team, score, reward))

				print("")

			if reward != 0:
				print("reward: ", reward)

			done = not env.is_running()
			if done or step == 2000:
				print('Environment stopped early')
				print("exit program")

				# Save camera frame as video
				frame_array = np.array(frame_list)
				for i in range(len(frame_array)):
					# writing to a image array
					video_out.write(frame_array[i])

				video_out.release()

				save_data = {'episode': episode, 'step': step_list,
							   'action': action_list, 'reward': reward_list, 'done': done_list}

				path_npy = save_path + save_file + '.npy'
				np.save(path_npy, save_data)

				break

			obs = env.observations()
			#print("obs.keys(): ", obs.keys())

			#cus_obs = env.custom_observations()

			#print("obs['RGB_INTERLEAVED'].shape: ", obs['RGB_INTERLEAVED'].shape)
			#obs_raw = cv2.cvtColor(obs['RGB_INTERLEAVED'], cv2.COLOR_BGR2RGB)
			obs_screen = obs['RGB_INTERLEAVED']
			#obs_gadget_amount = obs['DEBUG.GADGET_AMOUNT']
			#obs_gadget = obs['DEBUG.GADGET']
			#obs_has_red_flag = obs['DEBUG.HAS_RED_FLAG']
			#print("obs_has_red_flag: ", obs_has_red_flag)

			#has_red_flag_onehot = one_hot(int(obs_has_red_flag), 2) 

			#print("obs_gadget_amount: ", obs_gadget_amount)
			#print("has_red_flag_onehot: ", has_red_flag_onehot)

			#obs_inv = np.concatenate([obs_gadget_amount / 100.0, has_red_flag_onehot])
			#print("obs_inv: ", obs_inv)
			#print("")

			episode_list.append(episode)
			step_list.append(step)
			
			frame = cv2.cvtColor(obs_screen, cv2.COLOR_BGR2RGB)
			frame = cv2.resize(frame, dsize=(720, 640), interpolation=cv2.INTER_AREA)
			frame_list.append(frame)

			action_list.append([action_look_horizontal, 
								action_look_vertical, 
								action_strafe_horizontal, 
								action_forward_backward, 
								weapon_fire, 
								jump, 
								crouch])
			reward_list.append(reward)
			done_list.append(done)

			#surf = pygame.surfarray.make_surface(obs)
			obs_surf = cv2.rotate(obs_screen, cv2.ROTATE_90_COUNTERCLOCKWISE)
			obs_surf = cv2.flip(obs_surf, 0)
			surf = pygame.surfarray.make_surface(obs_surf)
			gameDisplay.blit(surf, (0, 0))
			pygame.display.update()

			#pygame.mouse.set_pos([720 / 2, 640 / 2])
			step += 1
			#print("step: ", step)

			#pygame.event.pump()
			#pygame.time.delay(100)
			clock.tick(60)

	env.close()