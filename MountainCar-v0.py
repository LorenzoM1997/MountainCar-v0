"""
Created by Lorenzo Mambretti
on 11/18/2017

My idea is to use a Neural Network in tensorflow to predict a past state (and action) given the current state
Once we have this prediction, we update the q-matrix updating the previous state of each state

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

def random_ride(x_size, y_size):

    n_iteration = 200

    inputs = np.zeros((0,x_size))
    outputs = np.zeros((0,y_size))
    
    for _ in range(100):
        env.reset()
        action = env.action_space.sample()
        for t in range(n_iteration):

            # read the observation and convert them
            observation, reward, done, info = env.step(action)
            observation[0] = np.floor(observation[0] * dx)/dx
            observation[1] = np.floor(observation[1] * dy)/dy

            if t < n_iteration - 1:
                if y_size == 3:
                    outputs = np.append(outputs,[[observation[0],observation[1],action]], axis = 0)
                else:
                    outputs = np.append(outputs,[[observation[0],observation[1]]], axis = 0)
            if t > 0:
                if x_size == 2:
                    inputs = np.append(inputs,[[observation[0],observation[1]]],axis = 0)
                else:
                    inputs = np.append(inputs,[[observation[0],observation[1],action]],axis = 0)         
        
            action = env.action_space.sample()

    return inputs, outputs

def correct(state):
    prob_x = int(state[0,0] * dx - state_min[0] * dx)
    prob_y = int(state[0,1] * dy - state_min[1] * dy)
    orig_x = prob_x
    orig_y = prob_y
    keep_going = True
    while (keep_going):
        keep_going = False
        if prob_x < 0:
            prob_x = 0
            keep_going = True
        if prob_y < 0:
            prob_y = 0
            keep_going = True
        if prob_x >= n_x -1:
            prob_x = n_x - 2
            keep_going = True
        if prob_y >= n_y -1:
            prob_y = n_y - 2
            keep_going = True
        if prob_x == orig_x:
            prob_x += (random.randint(0,1)*2)-1
            keep_going = True
        if prob_y == orig_y:
            prob_y += (random.randint(0,1)*2)-1
            keep_going = True
    return prob_x, prob_y

class nn():
    def __init__(self,x_size,y_size):
        self.x_size = x_size
        self.y_size = y_size
        self.x = tf.placeholder(tf.float32, [None,x_size])
        self.W = tf.Variable(tf.random_uniform([x_size,y_size]))
        self.b = tf.Variable(tf.zeros([y_size]))

        self.p = tf.tanh(tf.matmul(self.x,self.W)+self.b)

        self.y = tf.placeholder(tf.float32, [None, y_size])

        #the learning rate of the Gradient Descent Optimizer
        lr = 0.01
        
        squared_deltas = tf.square(self.p - self.y)
        self.loss = tf.reduce_mean(squared_deltas)
        self.train_step = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)
        self.sess = tf.InteractiveSession()

    def train(self):
        
        tf.global_variables_initializer().run()

        i,o = random_ride(self.x_size, self.y_size)

        print("Start training probability network")
    
        for j in range(10001):
            aslice = random.randint(0,len(i)-501)
            batch_xs = i[aslice:aslice+500]
            batch_ys = o[aslice:aslice+500]
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y: batch_ys})
            if j%10000 == 0:
                # Test trained model
                print("loss = ", self.sess.run(self.loss, feed_dict={self.x: i,
                                                  self.y: o}))
        print("Training completed")

    def predict_state(self,data):
        return self.sess.run(self.p, feed_dict={self.x: data})
             

def choose_action(q_matrix,observation):
    x = int(observation[0]*dx-state_min[0]*dx)
    y = int(observation[1]*dy-state_min[1]*dy)
    return np.argmax(q_matrix[x,y,:])

def train(past_nn, future_nn):
    
    print("Start RL training")
    n_iteration = 50
    
    for z in range(100):
        difference = 0
        for i in range(n_x):
            for j in range(n_y):
                    
                obs1 = (i + dx * state_min[0]) / dx
                obs2 = (j + dy * state_min[1]) / dy
                probable_state  = past_nn.predict_state([[obs1,obs2]])

                prob_x, prob_y = correct(probable_state)
                
                future_reward = np.max(q_matrix[i, j,:])
                for action in range(n_actions):

                    # update previous state
                    if i == n_x-1:
                        current_reward = 1
                    else: current_reward = 0
                    
                    difference += abs(q_matrix[prob_x,prob_y,action]-(current_reward + future_reward))
                    if action == 1: probability = probable_state[0,2]
                    else: probability = 1 - probable_state[0,2]
                    q_matrix[prob_x,prob_y,action] = current_reward + probability * future_reward

                    # update current state
                    prob_x, prob_y = correct(future_nn.predict_state([[i,j,action]]))

                    q_matrix[i,j,action] = current_reward + np.max(q_matrix[prob_x,prob_y])
                    

        if difference < (epsilon/n_states):
            print("Reinforcement Learning terminated after iteration",z)
            break

    print("Q-matrix optimization completed!")
    
    return q_matrix

def run_game(q_matrix):
    observation = env.reset()
    action = env.action_space.sample()
    for t in range(1000):
        observation, reward, done, info = env.step(action)
        observation[0] = np.floor(observation[0] * dx) / dx
        observation[1] = np.floor(observation[1] * dy) / dy
        env.render()
        
        action = choose_action(q_matrix,observation)

        """
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break"""

past_nn = nn(2,3)
past_nn.train()
future_nn = nn(3,2)
future_nn.train()
q_matrix = train(past_nn, future_nn)
run_game(q_matrix)
