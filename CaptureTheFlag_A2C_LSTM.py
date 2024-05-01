import os
import random
import gym
import pylab
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, LSTM, Reshape, InputLayer, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import cv2
import threading
from threading import Thread, Lock
import time
import tensorflow_probability as tfp
from typing import Any, List, Sequence, Tuple
import deepmind_lab

tfd = tfp.distributions

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
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


REWARDS = {
    'PICKUP_REWARD': 0,
    'PICKUP_GOAL': 0,
    'TARGET_SCORE': 0,
    'TAG_SELF': 0,
    'TAG_PLAYER': 0,
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

ACTION_LIST = list(ACTIONS)
#print("ACTION_LIST: ", ACTION_LIST)

num_actions = 7
screen_size = (64,64,3)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)

    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(64, 64, 3)),
                Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu'),
                Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                Flatten(),
                Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dim,)),
                Dense(units=16*16*16, activation=tf.nn.relu),
                Reshape(target_shape=(16, 16, 16)),
                Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu'),
                Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same')
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
    
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
    
        return logits



class OurModel(tf.keras.Model):
    def __init__(self, action_space):
        super(OurModel, self).__init__()
        
        #self.flatten = Flatten()
        #self.conv_1 = Conv2D(8, 8, 4, padding="valid", activation="relu")
        #self.conv_2 = Conv2D(16, 4, 2, padding="valid", activation="relu")
        #self.conv_3 = Conv2D(16, 3, 1, padding="valid", activation="relu")
        
        self.common_1 = Dense(512, activation="relu", kernel_regularizer='l2')
        self.common_2 = Dense(64, activation="relu", kernel_regularizer='l2')
        self.common_3 = Dense(512, activation="relu", kernel_regularizer='l2')
        
        latent_dim = int(256)
        self.CVAE = CVAE(latent_dim)
        self.lstm = LSTM(128, return_sequences=True, return_state=True)
        self.actor = Dense(action_space)
        self.critic = Dense(1)
        
    def call(self, input_screen, input_inv, memory_state, carry_state, training):
        batch_size = input_screen.shape[0]
        
        mean, logvar = self.CVAE.encode(input_screen)
        cvae_output = tf.concat((mean, logvar), axis=1)
        cvae_output_reshaped = Reshape((16,32))(cvae_output)

        initial_state = (memory_state, carry_state)
        lstm_output, final_memory_state, final_carry_state  = self.lstm(cvae_output_reshaped, initial_state=initial_state, 
                                                                        training=training)
        X_input_screen = Flatten()(lstm_output)
        X_input_screen = self.common_1(X_input_screen)

        X_input_inv = self.common_2(input_inv)

        X_input = tf.concat([X_input_screen, X_input_inv], 1)
        x = self.common_3(X_input)
        
        z = self.CVAE.reparameterize(mean, logvar)
        x_logit = self.CVAE.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=input_screen)

        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        cvae_loss = logpx_z + logpz - logqz_x
        
        action_logit = self.actor(x)
        value = self.critic(x)
        
        return action_logit, value, final_memory_state, final_carry_state, cvae_loss


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
mse_loss = tf.keras.losses.MeanSquaredError()


class A3CAgent:
    # Actor-Critic Main Optimization Algorithm
    def __init__(self, env_name):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        self.action_size = 7
        self.EPISODES, self.episode, self.max_average = 2000000, 0, -21.0 # specific for pong
        self.lock = Lock()
        self.lr = 0.0001

        # Instantiate plot memory
        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models'
        
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A3C_{}'.format(self.env_name, self.lr)
        self.model_name = os.path.join(self.Save_Path, self.path)

        # Create Actor-Critic network model
        self.ActorCritic_CVAE = OurModel(action_space=self.action_size)
        
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        
        # DMLab parameter
        self.fps = 15
        self.action_repeat = int(60 / self.fps)
        self.width = 320
        self.height = 320
        self.screen_size = (64,64,3)

        self.env = deepmind_lab.Lab(env_name, ['RGB_INTERLEAVED', 'DEBUG.GADGET_AMOUNT', 'DEBUG.GADGET', 'DEBUG.HAS_RED_FLAG'],
                                    {'fps': str(self.fps), 'width': str(self.width), 'height': str(self.height)})
        
    @tf.function
    def act(self, state_screen, state_inv, memory_state, carry_state):
        # Use the network to predict the next action to take, using the model
        prediction = self.ActorCritic_CVAE(state_screen, state_inv, 
                                           memory_state, carry_state, training=False)
        action = tf.random.categorical(prediction[0], 1)

        memory_state = prediction[2]
        carry_state = prediction[3]
        
        return action[0][0], memory_state, carry_state

    @tf.function
    def get_expected_return(self, rewards: tf.Tensor, dones: tf.Tensor, 
                            gamma: float = 0.99, standardize: bool = True) -> tf.Tensor:
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        dones = tf.cast(dones[::-1], dtype=tf.bool)
        
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            done = dones[i]
            if tf.cast(done, tf.bool):
                discounted_sum = tf.constant(0.0)
            
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + self.eps))

        return returns
    
    def replay(self, states_screen, states_inv, actions, rewards, dones, memory_states, carry_states):
        # reshape memory to appropriate shape for training
        batch_size = states_screen.shape[0]
        
        # Compute discounted rewards
        discounted_r = self.get_expected_return(rewards, dones)
        discounted_r_ = tf.stack(discounted_r)
        with tf.GradientTape() as tape:
            action_logits = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            cvae_losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            
            memory_state = tf.expand_dims(memory_states[0], 0)
            carry_state = tf.expand_dims(carry_states[0], 0)
            for i in tf.range(0, batch_size):
                prediction = self.ActorCritic_CVAE(tf.expand_dims(states_screen[i], 0), 
                                                   tf.expand_dims(states_inv[i], 0),
                                                   memory_state, carry_state, training=True)
                
                action_logits = action_logits.write(i, prediction[0][0])
                values = values.write(i, prediction[1][0])
                cvae_losses = cvae_losses.write(i, prediction[2][0])
                
                memory_state = prediction[2]
                carry_state = prediction[3]
                
            action_logits = action_logits.stack()
            values = values.stack()
            cvae_losses = cvae_losses.stack()
            
            advantages = discounted_r - tf.stop_gradient(tf.stack(values)[:, 0])
            
            action_prob = tf.nn.softmax(action_logits)
            dist = tfd.Categorical(probs=action_prob)
            action_log_prob = dist.prob(actions)
            action_log_prob = tf.math.log(action_log_prob)
            #print("action_logits_selected_probs: ", action_logits_selected_probs)
            #print("action_log_prob.shape: ", action_log_prob)
            
            actor_loss = -tf.math.reduce_mean(action_log_prob * advantages) 
            
            critic_loss = mse_loss(tf.stack(values)[:, 0], discounted_r)
            critic_loss = tf.cast(critic_loss, 'float32')
            total_loss = actor_loss + critic_loss + cvae_losses
        
        #print("total_loss: ", total_loss)
        #print("")
            
        grads = tape.gradient(total_loss, self.ActorCritic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.ActorCritic.trainable_variables))
        
    def load(self, model_name):
        self.ActorCritic = load_model(model_name, compile=False)
        #self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.ActorCritic.save(self.model_name)
        #self.Critic.save(self.Model_name + '_Critic.h5')

    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))

        return self.average[-1]
    
    def one_hot(self, a, num_classes):
        return np.squeeze(np.eye(num_classes)[a])
    
    def reset(self):
        self.env.reset(seed=14)
        obs = self.env.observations()
        obs_screen = obs['RGB_INTERLEAVED']
        obs_screen = cv2.cvtColor(obs_screen, cv2.COLOR_BGR2RGB)
        obs_screen = cv2.resize(obs_screen, dsize=(64,64), interpolation=cv2.INTER_AREA)

        obs_weapon_amount = obs['DEBUG.GADGET_AMOUNT']
        obs_weapon = obs['DEBUG.GADGET']
        obs_has_red_flag = obs['DEBUG.HAS_RED_FLAG']

        state_screen = obs_screen / 255.0

        has_red_flag_onehot = self.one_hot(int(obs_has_red_flag), 2) 
        state_inv = np.concatenate([obs_weapon_amount / 100.0, has_red_flag_onehot])

        return state_screen, state_inv
    
    def step(self, action):
        reward = self.env.step(ACTIONS[ACTION_LIST[action]], num_steps=self.action_repeat)
        obs1 = self.env.observations()

        obs1_screen = obs1['RGB_INTERLEAVED']
        obs1_screen = cv2.cvtColor(obs1_screen, cv2.COLOR_BGR2RGB)

        self.render(obs1_screen)

        obs1_screen = cv2.resize(obs1_screen, dsize=(64,64), interpolation=cv2.INTER_AREA)

        obs1_weapon_amount = obs1['DEBUG.GADGET_AMOUNT']
        obs1_weapon = obs1['DEBUG.GADGET']
        obs1_has_red_flag = obs1['DEBUG.HAS_RED_FLAG']

        next_state_screen = obs1_screen / 255.0

        has_red_flag_onehot1 = self.one_hot(int(obs1_has_red_flag), 2) 
        next_state_inv = np.concatenate([obs1_weapon_amount / 100.0, has_red_flag_onehot1])
        
        done = not self.env.is_running()
        
        return next_state_screen, next_state_inv, None, done, None
    
    def parse_ctf_reward(self, events):
        reward = 0
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
    
        return reward
    
    def render(self, obs):
        cv2.imshow('obs', obs)
        cv2.waitKey(1)
    
    def train(self):
        while self.episode < self.EPISODES:
            # Reset episode
            score, done, SAVING = 0, False, ''
            state_screen, state_inv = self.reset()

            states_screen, states_inv, actions, rewards, dones = [], [], [], [], []
            memory_states, carry_states = [], []
            
            memory_state = np.zeros([1,128], dtype=np.float32)
            carry_state = np.zeros([1,128], dtype=np.float32)
            
            step = 0
            while not done:
                print("step :", step)
                #print("done :", done)
                print("state_screen.shape :", state_screen.shape)
                
                #cv2.imshow('state_screen', state_screen)
                #cv2.waitKey(1)
                
                print("")
                
                action, memory_state, carry_state = self.act(np.reshape(state_screen, (1,*self.screen_size)), 
                                                             np.reshape(state_inv, (1, 3)),
                                                             memory_state, carry_state)
                action = action.numpy()
                memory_state = memory_state.numpy()
                carry_state = carry_state.numpy()
                
                next_screen_state, next_inv_state, _, done, _ = self.step(action)
                events = self.env.events()
                reward = self.parse_ctf_reward(events)
                
                states_screen.append(state_screen)
                states_inv.append(state_inv)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                
                memory_states.append(memory_state)
                carry_states.append(carry_state)

                score += reward
                state_screen = next_screen_state
                state_inv = next_inv_state
                
                step += 1
                
            # Test CVAE result
            mean, logvar = ActorCritic_CVAE.CVAE.encode(states_screen)
            z = model.CVAE.reparameterize(mean, logvar)
            reconstruced = ActorCritic_CVAE.CVAE.sample(z)
            reconstruced = np.array(reconstruced)
            reconstruced = cv2.resize(reconstruced[0], dsize=(320,224), interpolation=cv2.INTER_AREA)
            reconstruced = cv2.cvtColor(reconstruced, cv2.COLOR_BGR2RGB)
            #print("reconstruced.shape: ", reconstruced.shape)
            cv2.imwrite("images/state_" + str(index) + ".png", np.array(screen_state[0] * 255.0).astype(np.uint8))
            cv2.imwrite("images/reconstruced_" + str(index) + ".png", (reconstruced * 255.0).astype(np.uint8))
                    
            # Train RL model
            self.lock.acquire()
            self.replay(np.array(states), np.array(actions), np.array(rewards), np.array(dones),
                        np.array(memory_states), np.array(carry_states))
            self.lock.release()
            
            states, actions, rewards, dones = [], [], [], []
                    
            # Update episode count
            with self.lock:
                average = self.PlotModel(score, self.episode)
                # saving best models
                if average >= self.max_average:
                    self.max_average = average
                    #self.save()
                    SAVING = "SAVING"
                else:
                    SAVING = ""

                print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
                if(self.episode < self.EPISODES):
                    self.episode += 1

        self.env.close()            

    def test(self, Actor_name, Critic_name):
        self.load(Actor_name, Critic_name)
        for e in range(100):
            state = self.reset(self.env)
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax(self.Actor.predict(state))
                state, reward, done, _ = self.step(action, self.env, state)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break

if __name__ == "__main__":
    env_name = 'ctf_simple'
    agent = A3CAgent(env_name)
    
    agent.train() # use as A3C
    #agent.test('Models/Pong-v0_A3C_2.5e-05_Actor.h5', '')