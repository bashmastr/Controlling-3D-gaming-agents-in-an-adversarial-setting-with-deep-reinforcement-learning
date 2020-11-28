
import random
import tensorflow.compat.v1 as tf   
import numpy as np
import retro
import time




class QNetwork():
    def __init__(self, state_dim, action_size):
        self.state_in = tf.placeholder(tf.float32, shape=[None, *state_dim])
        self.action_in = tf.placeholder(tf.int32, shape=[None])
        self.q_target_in = tf.placeholder(tf.float32, shape=[None])
        action_one_hot = tf.one_hot(self.action_in, depth=action_size)
        
        self.hidden1 = tf.layers.dense(self.state_in, 100, activation=tf.nn.relu)
        # print(self.hidden1)
        self.q_state = tf.layers.dense(self.hidden1, action_size, activation=None)
        # print(self.q_state)
        self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state, action_one_hot), axis=1)          
        
        self.loss = tf.reduce_mean(tf.square(self.q_state_action- self.q_target_in)) 
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def update_model(self, session, state, action, q_target):
        feed = {self.state_in:state, self.action_in:action, self.q_target_in:q_target}
        session.run(self.optimizer, feed_dict=feed)

    def get_q_state(self, session, state):
        q_state = session.run(self.q_state, feed_dict={self.state_in:state})
        return q_state


class DQNAgent():
    def __init__(self, env):
        self.state_dim = env.observation_space.shape
        self.action_size = env.action_space.n
        self.q_network = QNetwork(self.state_dim, self.action_size)

        self.gamma = 0.90
        self.eps = 1.0

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
        q_next_state = (1-done) * q_next_state
        q_target = reward + self.gamma + np.max(q_next_state)
        self.q_network.update_model(self.sess, [state], [action], [q_target])

        if done: self.eps = max(0.1, 0.99 * self.eps)

    # def __del__(self):
    #     self.sess.close()

if __name__ == "__main__":

    tf.disable_v2_behavior()
    game_name = "Airstriker-Genesis"
    env = retro.make(game_name)
    # print("Obeservation State: ", env.observation_space)
    # for i in range(15):
    #     print("Action space: ", env.action_space)

    agent = DQNAgent(env)
    # state = env.reset()
    num_episode = 5

    actions = {
        "0": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "1": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "2": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "3": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "4": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "5": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "6": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "7": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "8": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "9": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "10": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "11": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    }
    tf.train.list_variables('./saved_models')

    # for ep in range(num_episode):
    #     tf.train.list_variables('./saved_models')

    #     if ep == 0:
    #         tf.train.load_checkpoint('./saved_models')

    #     state = env.reset()
    #     total_reward = 0
    #     done = False
    #     while not done:
    #         actionDigit = agent.get_action(state)
    #         if actionDigit > 11 or actionDigit < 0:
    #             print("Not found")
    #             actionDigit = 10 
    #         action = actions[str(actionDigit)]
    #         # print("ActionDigit:", actionDigit)
    #         # print("Action:", action)
    #         # print("State:", state)
    #         next_state, reward, done, info = env.step(action)
    #         agent.train(state, actionDigit, next_state, reward, done)
    #         # env.render()
    #         total_reward += reward
    #         state = next_state
    #         print("next_state:", next_state)
    #         print("Reward:", reward)
    #         print("Done:", done)
    #         # time.sleep(10)

    #     if ep % 1 == 0:
    #         # print("Completed Training Cycle: " + str(epoch) + " out of " + str(self.num_of_epoch))
    #         # print("Current Loss: " + str(loss))

    #         saver = tf.train.Saver()
    #         saver.save(agent.sess, 'saved_models/testing')
    #         print("Model saved")

    #     # print("Episode: {}, total_rewards: {1.2f}".format(ep, total_reward))
    #     print("Episode: ", ep)
    #     print("total_reward: ", total_reward)
