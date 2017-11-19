"""
Created by Lorenzo Mambretti
on 11/18/2017
"""

import gym
import numpy as np
import tensorflow as tf
import random

env = gym.make('MountainCar-v0')
max_attempts = 20
epsilon = 0.1

dx = 20
dy = 200

# create the q-matrix with the right dimensions
n_actions = env.action_space.n
state_max, state_min = env.observation_space.high, env.observation_space.low
n_x = int(state_max[0]*dx - state_min[0]*dx)
n_y = int(state_max[1]*dy-state_min[1]*dy)
n_states = n_x * n_y
q_matrix = np.zeros((n_x, n_y,n_actions))

x = tf.placeholder(tf.float32, [None,3])
W = tf.Variable(tf.random_uniform([3,2]))
b = tf.Variable(tf.zeros([2]))

p = tf.tanh(tf.matmul(x,W)+b)

y = tf.placeholder(tf.float32, [None, 2])

#the learning rate of the Gradient Descent Optimizer
lr = 0.01

squared_deltas = tf.square(p - y)
loss = tf.reduce_mean(squared_deltas)
train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

def random_ride():

    n_iteration = 200

    inputs = np.zeros((0,3))
    outputs = np.zeros((0,2))
    
    
    for _ in range(25):
        env.reset()
        action = env.action_space.sample()
        for t in range(n_iteration):

            # read the observation and convert them
            observation, reward, done, info = env.step(action)
            observation[0] = np.floor(observation[0] * dx)/dx
            observation[1] = np.floor(observation[1] * dy)/dy

            if t < n_iteration - 1:
                inputs = np.append(inputs,[[observation[0],observation[1],action]], axis = 0)
           
            if t > 0:
                outputs = np.append(outputs,[[observation[0],observation[1]]],axis = 0)        
        
            action = env.action_space.sample()      

    return inputs,outputs

def train_nn(i,o):
    print("Start training probability network")
    
    for j in range(10001):
        aslice = random.randint(0,len(i)-501)
        batch_xs = i[aslice:aslice+500]
        batch_ys = o[aslice:aslice+500]
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        if j%1000 == 0:
            # Test trained model
            print("loss = ", sess.run(loss, feed_dict={x: batch_xs,
                                                  y: batch_ys}))

    print("Training completed")

def predicted_state(obs1,obs2,action):
    return sess.run(p, feed_dict={x: [[obs1, obs2, action]]})
             

def choose_action(q_matrix,observation):
    x = int(observation[0]*dx-state_min[0]*dx)
    y = int(observation[1]*dy-state_min[1]*dy)
    return np.argmax(q_matrix[x,y,:])

def train():

    n_iteration = 50
    
    for z in range(50):
        difference = 0
        for i in range(n_x):
            for j in range(n_y):
                for action in range(n_actions):
                    current_reward = 0
                    if i == n_x-1:
                        current_reward = 1
                    else: current_reward = q_matrix[i,j,action]
                    
                    obs1 = (i + dx * state_min[0]) / dx
                    obs2 = (j + dy * state_min[1]) / dy
                    probable_state  = predicted_state(obs1,obs2,action)
                    probable_x = int(probable_state[0,0] * dx-state_min[0] * dx)
                    probable_y = int(probable_state[0,1] * dy-state_min[1] * dy)
                    if probable_x < 0: probable_x = 0
                    if probable_y < 0: probable_y = 0
                    if probable_x >= n_x -1:
                        probable_x = n_x - 2
                    if probable_y >= n_y -1:
                        probable_y = n_y - 2
                    future_reward = np.max(q_matrix[probable_x, probable_y,:])
                    difference += abs(q_matrix[i,j,action]-(current_reward + future_reward))
                    q_matrix[i,j,action] = current_reward + future_reward
                    

        if difference < epsilon:
            print("Reinforcement Learning terminated after iteration",z)
            break
        
    return q_matrix

def run_game(q_matrix):
    observation = env.reset()
    action = env.action_space.sample()
    for t in range(200):
        observation, reward, done, info = env.step(action)
        observation[0] = np.floor(observation[0] * dx) / dx
        observation[1] = np.floor(observation[1] * dy) / dy
        env.render()
        
        action = choose_action(q_matrix,observation)
        print(observation,done,action)
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

i, o = random_ride()
train_nn(i,o)
q_matrix = train()
run_game(q_matrix)
