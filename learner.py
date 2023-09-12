import time
import math
import zmq
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, LSTMCell
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import threading
import random
import collections
import argparse
from absl import flags
from absl import logging
from typing import Any, List, Sequence, Tuple
from gym.spaces import Dict, Discrete, Box, Tuple
import network
import gym
from parametric_distribution import get_parametric_distribution_for_action_space

parser = argparse.ArgumentParser(description='CTF IMPALA Server')
parser.add_argument('--exp_name', type=str, default="kill", help='name of experiment')
parser.add_argument('--env_num', type=int, default=2, help='ID of environment')
parser.add_argument('--batch_size', type=int, default=2, help='batch size of training')
parser.add_argument('--unroll_length', type=int, default=50, help='unroll length of trajectory')
parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')
parser.add_argument('--pretrained_model', type=str, help='pretrained model name')
arguments = parser.parse_args()

tfd = tfp.distributions

if arguments.gpu_use == True:
    print("if arguments.gpu_use == True")
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


socket_list = []
for i in range(0, arguments.env_num):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:" + str(6555 + i))

    socket_list.append(socket)


num_actions = 7
screen_size = (64,64,3)    

batch_size = arguments.batch_size

unroll_length = 50
queue = tf.queue.FIFOQueue(1, dtypes=[tf.int32, tf.float32, tf.bool, tf.float32, tf.float32, tf.float32, tf.int32, tf.float32, tf.float32], 
                           shapes=[[unroll_length+1],[unroll_length+1],[unroll_length+1],[unroll_length+1,*screen_size],[unroll_length+1,3],
                                   [unroll_length+1,num_actions],[unroll_length+1],[unroll_length+1,128],[unroll_length+1,128]])
Unroll = collections.namedtuple('Unroll', 'env_id reward done obs_screen obs_inv policy action memory_state carry_state')

num_hidden_units = 512
model = network.ActorCritic(num_actions, num_hidden_units)

print("Load Pretrained Model")
model.load_weights("kill/model/" + "model_1000")

num_action_repeats = 1
total_environment_frames = int(4e7)

iter_frame_ratio = (batch_size * unroll_length * num_action_repeats)
final_iteration = int(math.ceil(total_environment_frames / iter_frame_ratio))
    
lr = tf.keras.optimizers.schedules.PolynomialDecay(0.0001, final_iteration, 0)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)

writer = tf.summary.create_file_writer(arguments.exp_name + "/tensorboard_learner")

def take_vector_elements(vectors, indices):
    return tf.gather_nd(vectors, tf.stack([tf.range(tf.shape(vectors)[0]), indices], axis=1))


parametric_action_distribution = get_parametric_distribution_for_action_space(Discrete(num_actions))
kl = tf.keras.losses.KLDivergence()

def update(screen_states, inv_states, actions, agent_policies, rewards, dones, memory_states, carry_states):
    screen_states = tf.transpose(screen_states, perm=[1, 0, 2, 3, 4])
    inv_states = tf.transpose(inv_states, perm=[1, 0, 2])
    actions = tf.transpose(actions, perm=[1, 0])
    agent_policies = tf.transpose(agent_policies, perm=[1, 0, 2])
    rewards = tf.transpose(rewards, perm=[1, 0])
    dones = tf.transpose(dones, perm=[1, 0])
    memory_states = tf.transpose(memory_states, perm=[1, 0, 2])
    carry_states = tf.transpose(carry_states, perm=[1, 0, 2])
    
    batch_size = screen_states.shape[0]
    
    online_variables = model.trainable_variables
    with tf.GradientTape() as tape:
        tape.watch(online_variables)
        
        learner_policies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        learner_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        cvae_losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        memory_state = memory_states[0]
        carry_state = carry_states[0]
        for i in tf.range(0, batch_size):
            prediction = model(screen_states[i], inv_states[i], memory_state, carry_state, training=True)

            learner_policies = learner_policies.write(i, prediction[0])
            learner_values = learner_values.write(i, prediction[1])
            
            memory_state = prediction[2]
            carry_state = prediction[3]
            cvae_loss = prediction[4]

            cvae_losses = cvae_losses.write(i, cvae_loss)

        learner_policies = learner_policies.stack()
        learner_values = learner_values.stack()
        cvae_losses = cvae_losses.stack()

        learner_policies = tf.reshape(learner_policies, [screen_states.shape[0], screen_states.shape[1], -1])
        learner_values = tf.reshape(learner_values, [screen_states.shape[0], screen_states.shape[1], -1])

        agent_logits = tf.nn.softmax(agent_policies[:-1])
        actions = actions[:-1]
        rewards = rewards[1:]
        dones = dones[1:]
        
        learner_logits = tf.nn.softmax(learner_policies[:-1])
            
        learner_values = tf.squeeze(learner_values, axis=2)
            
        bootstrap_value = learner_values[-1]
        learner_values = learner_values[:-1]

        discounting = 0.99
        discounts = tf.cast(~dones, tf.float32) * discounting

        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        
        target_action_log_probs = parametric_action_distribution.log_prob(learner_policies[:-1], actions)
        behaviour_action_log_probs = parametric_action_distribution.log_prob(agent_policies[:-1], actions)
        
        lambda_ = 1.0
        
        log_rhos = target_action_log_probs - behaviour_action_log_probs
        
        log_rhos = tf.convert_to_tensor(log_rhos, dtype=tf.float32)
        discounts = tf.convert_to_tensor(discounts, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        values = tf.convert_to_tensor(learner_values, dtype=tf.float32)
        bootstrap_value = tf.convert_to_tensor(bootstrap_value, dtype=tf.float32)
        
        clip_rho_threshold = tf.convert_to_tensor(1.0, dtype=tf.float32)
        clip_pg_rho_threshold = tf.convert_to_tensor(1.0, dtype=tf.float32)
        
        rhos = tf.math.exp(log_rhos)
        
        clipped_rhos = tf.minimum(clip_rho_threshold, rhos, name='clipped_rhos')
        
        cs = tf.minimum(1.0, rhos, name='cs')
        cs *= tf.convert_to_tensor(lambda_, dtype=tf.float32)

        values_t_plus_1 = tf.concat([values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)
        
        acc = tf.zeros_like(bootstrap_value)
        vs_minus_v_xs = []
        for i in range(int(discounts.shape[0]) - 1, -1, -1):
            discount, c, delta = discounts[i], cs[i], deltas[i]
            acc = delta + discount * c * acc
            vs_minus_v_xs.append(acc)  
        
        vs_minus_v_xs = vs_minus_v_xs[::-1]
        
        vs = tf.add(vs_minus_v_xs, values, name='vs')
        vs_t_plus_1 = tf.concat([vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        clipped_pg_rhos = tf.minimum(clip_pg_rho_threshold, rhos, name='clipped_pg_rhos')
        
        pg_advantages = (clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values))
        
        vs = tf.stop_gradient(vs)
        pg_advantages = tf.stop_gradient(pg_advantages)
        
        actor_loss = -tf.reduce_mean(target_action_log_probs * pg_advantages)
        
        baseline_cost = 0.5
        v_error = values - vs
        critic_loss = baseline_cost * 0.5 * tf.reduce_mean(tf.square(v_error))
        
        entropy = tf.reduce_mean(parametric_action_distribution.entropy(learner_policies[:-1]))
        entropy_loss = 0.002 * -entropy
        
        cvae_loss = -tf.reduce_mean(cvae_losses)

        rl_loss = actor_loss + critic_loss + entropy_loss
        total_loss = rl_loss + cvae_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return rl_loss, cvae_loss


@tf.function
def prediction(screen_state, inv_state, memory_state, carry_state):
    #tf.print("state.shape: ", state.shape)

    prediction = model(screen_state, inv_state, memory_state, carry_state, training=False)
    dist = tfd.Categorical(logits=prediction[0])
    action = int(dist.sample()[0])
    policy = prediction[0]

    memory_state = prediction[2]
    carry_state = prediction[3]

    return action, policy, memory_state, carry_state


@tf.function
def enque_data(env_ids, rewards, dones, screen_states, inv_states, policies, actions, memory_states, carry_states):
    queue.enqueue((env_ids, rewards, dones, screen_states, inv_states, policies, actions, memory_states, carry_states))


def Data_Thread(coord, i):
    env_ids = np.zeros((unroll_length + 1), dtype=np.int32)
    screen_states = np.zeros((unroll_length + 1, *screen_size), dtype=np.float32)
    inv_states = np.zeros((unroll_length + 1, 3), dtype=np.float32)
    actions = np.zeros((unroll_length + 1), dtype=np.int32)
    policies = np.zeros((unroll_length + 1, num_actions), dtype=np.float32)
    rewards = np.zeros((unroll_length + 1), dtype=np.float32)
    dones = np.zeros((unroll_length + 1), dtype=np.bool)
    memory_states = np.zeros((unroll_length + 1, 128), dtype=np.float32)
    carry_states = np.zeros((unroll_length + 1, 128), dtype=np.float32)

    memory_index = 0

    index = 0
    memory_state = np.zeros([1,128], dtype=np.float32)
    carry_state = np.zeros([1,128], dtype=np.float32)
    min_elapsed_time = 5.0

    reward_list = []

    while not coord.should_stop(): 
        start = time.time()

        message = socket_list[i].recv_pyobj()
        if memory_index == unroll_length:
            enque_data(env_ids, rewards, dones, screen_states, inv_states, policies, actions, memory_states, carry_states)

            env_ids[0] = env_ids[memory_index]
            screen_states[0] = screen_states[memory_index]
            inv_states[0] = inv_states[memory_index]
            actions[0] = actions[memory_index]
            policies[0] = policies[memory_index]
            rewards[0] = rewards[memory_index]
            dones[0] = dones[memory_index]
            memory_states[0] = memory_states[memory_index]
            carry_states[0] = carry_states[memory_index]

            memory_index = 1

        #screen_state = tf.constant(message["obs_screen"])
        #inv_state = tf.constant(message["obs_inv"])
        screen_state = tf.convert_to_tensor(message["obs_screen"], dtype=tf.float32)
        inv_state = tf.convert_to_tensor(message["obs_inv"], dtype=tf.float32)

        action, policy, new_memory_state, new_carry_state = prediction(screen_state, inv_state, 
                                                                       tf.convert_to_tensor(memory_state, dtype=tf.float32), 
                                                                       tf.convert_to_tensor(carry_state, dtype=tf.float32))

        env_ids[memory_index] = message["env_id"]
        screen_states[memory_index] = message["obs_screen"]
        inv_states[memory_index] = message["obs_inv"]
        actions[memory_index] = action
        policies[memory_index] = policy
        rewards[memory_index] = message["reward"]
        dones[memory_index] = message["done"]
        memory_states[memory_index] = memory_state
        carry_states[memory_index] = carry_state

        reward_list.append(message["reward"])

        memory_state = new_memory_state
        carry_state = new_carry_state

        socket_list[i].send_pyobj({"env_id": message["env_id"], "action": action})

        memory_index += 1
        index += 1
        if index % 2000 == 0:
            average_reward = sum(reward_list[-50:]) / len(reward_list[-50:])
            #print("state.numpy().shape: ", state.numpy().shape)
            mean, logvar = model.CVAE.encode(screen_state)
            z = model.CVAE.reparameterize(mean, logvar)
            reconstruced = model.CVAE.sample(z)
            reconstruced = np.array(reconstruced)
            reconstruced = cv2.resize(reconstruced[0], dsize=(320,224), interpolation=cv2.INTER_AREA)
            reconstruced = cv2.cvtColor(reconstruced, cv2.COLOR_BGR2RGB)
            #print("reconstruced.shape: ", reconstruced.shape)
            cv2.imwrite("images/state_" + str(index) + ".png", np.array(screen_state[0] * 255.0).astype(np.uint8))
            cv2.imwrite("images/reconstruced_" + str(index) + ".png", (reconstruced * 255.0).astype(np.uint8))
            #cv2.imshow('state[0]', np.array(state[0]))
            #cv2.imshow('reconstruced', reconstruced)
            #cv2.waitKey(1)

        end = time.time()
        elapsed_time = end - start

    if index == 100000000:
        coord.request_stop()


unroll_queues = []
unroll_queues.append(queue)

def dequeue(ctx):
    dequeue_outputs = tf.nest.map_structure(
        lambda *args: tf.stack(args), 
        *[unroll_queues[ctx].dequeue() for i in range(batch_size)]
      )

    # tf.data.Dataset treats list leafs as tensors, so we need to flatten and repack.
    return tf.nest.flatten(dequeue_outputs)


def dataset_fn(ctx):
    dataset = tf.data.Dataset.from_tensors(0).repeat(None)
    def _dequeue(_):
      return dequeue(ctx)

    return dataset.map(_dequeue, num_parallel_calls=1)


if arguments.gpu_use == True:
    device_name = '/device:GPU:0'
else:
    device_name = '/device:CPU:0'


dataset = dataset_fn(0)
it = iter(dataset)


@tf.function
def minimize(iterator):
    dequeue_data = next(iterator)

    # env_id reward done obs_screen obs_inv policy action memory_state carry_state
    # screen_states: 3
    # inv_states: 4
    # actions: 6
    # agent_policies: 5
    # rewards: 1
    # dones: 2
    # memory_states: 7
    # carry_states: 8
    rl_loss, cvae_loss = update(dequeue_data[3], dequeue_data[4], dequeue_data[6], dequeue_data[5], dequeue_data[1], 
                                dequeue_data[2], dequeue_data[7], dequeue_data[8])

    return (rl_loss, cvae_loss)


def Train_Thread(coord):
    training_step = 0

    while not coord.should_stop():
        #print("training_step: ", training_step)

        rl_loss, cvae_loss = minimize(it)

        with writer.as_default():
            tf.summary.scalar("rl_loss", rl_loss, step=training_step)
            tf.summary.scalar("cvae_loss", cvae_loss, step=training_step)
            writer.flush()

        if training_step % 1000 == 0:
            model.save_weights(arguments.exp_name + '/' + 'model/model_' + str(training_step))

        if training_step == 100000000:
            coord.request_stop()

        training_step += 1


coord = tf.train.Coordinator(clean_stop_exception_types=None)

thread_data_list = []
for i in range(arguments.env_num):
    thread_data = threading.Thread(target=Data_Thread, args=(coord,i))
    thread_data_list.append(thread_data)

thread_train = threading.Thread(target=Train_Thread, args=(coord,))
thread_train.start()

for thread_data in thread_data_list:
    thread_data.start()

for thread_data in thread_data_list:
    coord.join(thread_data)

coord.join(thread_train)
