
import random
import tensorflow.compat.v1 as tf   
import numpy as np
from agent import Agent
from q_network_tf1 import QNetwork


class DQNAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01, epsilon=1.0):
        self.state_dim = env.observation_space.shape
        self.action_size = env.action_space.n
        self.q_network = QNetwork(self.state_dim, self.action_size, learning_rate)

        self.gamma = discount_rate
        self.eps = epsilon
        self.learning_rate = learning_rate

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def get_action(self, state):
        q_state = self.q_network.get_q_state(self.sess, [state])
        action_greedy = np.argmax(q_state)
        action_random = np.random.randint(self.action_size)
        action = action_random if random.random() < self.eps else action_greedy
        return action


    def train(self, state, action, next_state, reward, done):
        q_next_state = self.q_network.get_q_state(self.sess, [next_state])
        q_state = self.q_network.get_q_state(self.sess, [state])
        q_next_state = (1-done) * q_next_state
        q_target = reward + (self.gamma * np.max(q_next_state) - np.max(q_state))
        self.q_network.update_model(self.sess, [state], [action], [q_target])

        if done: self.eps = max(0.1, 0.99 * self.eps)


