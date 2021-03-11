
import random
# import tensorflow.compat.v1 as tf   
import numpy as np
import tensorflow as tf
import retro
import time

Air striker
Action size: 12
States size: 215040

CartPole
Action size: 2
States size: 16

Slimevolleygym
Action size: 3
States size: 12


class QNetwork():
    
    def __init__(self, state_dim, action_size, learning_rate):
        self.state_in = tf.keras.Input(shape=(state_dim), dtype=tf.dtypes.float32)
        self.action_in = tf.keras.Input(shape=(), dtype=tf.dtypes.int32)
        self.q_target_in = tf.keras.Input(shape=(), dtype=tf.dtypes.float32)
        self.action_one_hot = tf.one_hot(self.action_in, depth=action_size)
        
        # model = tf.keras.models.Sequential([
        #     tf.keras.layers.Conv2D(200, (3,3), activation='relu', input_shape=(self.state_in,))(self.state_in)
        #     tf.keras.layers.Flatten()
        #     tf.keras.layers.Dense(100, activation='relu')
        #     tf.keras.layers.Dense(action_size, activation="linear")
        # ])
        # self.model.compile(optimizer=tf.train.AdamOptimizer(1e-3), loss=loss_func(actions))


        # self.hidden1 = tf.keras.layers.Conv2D(200, (3,3), activation='relu', input_shape=(self.state_in,))(self.state_in)
        # self.hidden1 = tf.keras.layers.Flatten()(self.hidden1)
        self.hidden2 = tf.keras.layers.Dense(100, activation='relu',input_shape=(self.state_in,))(self.state_in)
        self.q_state_value = tf.keras.layers.Dense(action_size, activation="linear", input_shape=(self.hidden2,))(self.hidden2)
        self.model = tf.keras.models.Model(inputs=self.state_in , outputs=self.q_state_value)  
        self.model.compile(optimizer=tf.train.AdamOptimizer(learning_rate), loss=loss_function(self.action_in)) 
        
    def loss_function(self, actions):
        def nested_function(q_target, logits):
            print(q_target, logits)
            return tf.losses.mean_squared_error(tf.reduce_sum(tf.multiply(actions, logits), axis=-1), q_target)

        return nested_function

    def get_q_state(self, state):
        q_state_value = self.model.predict(state)
        q_state_action_value = tf.reduce_sum(tf.multiply(self.action_one_hot, q_state_value))
        return q_state_action_value

    def update_model(self, state, action, q_target):
        self.q_target_in = q_target
        self.action_in = action
        self.state_in = state
        self.model.fit(state, action, q_target)


    


class DQNAgent():
    def __init__(self, env):
        self.learning_rate = 0.001
        self.state_dim = env.observation_space.shape
        self.action_size = env.action_space.n
        self.q_network = QNetwork(self.state_dim, self.action_size, self.learning_rate)

        self.gamma = 0.90
        self.eps = 1.0

        


    def get_action(self, state):
        q_state_value = self.q_network.get_q_state([state])
        action_greedy = np.argmax(q_state_value)
        action_random = np.random.randint(self.action_size)
        action = action_random if random.random() < self.eps else action_greedy
        return action


    def train(self, state, action, next_state, reward, done):
        q_next_state_value = self.q_network.get_q_state([next_state])
        q_state_value = self.q_network.get_q_state([state])
        q_next_state_value = (1-done) * q_next_state_value
        q_target = reward + (self.gamma * np.max(q_next_state_value) - q_state_value)
        self.q_network.update_model([state], [action], [q_target])

        if done: self.eps = max(0.1, 0.99 * self.eps)



if __name__ == "__main__":



    game_name = "Airstriker-Genesis"
    env = retro.make(game_name)

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
    # tf.train.list_variables('./saved_models')

    # for ep in range(num_episode):
    #     tf.train.list_variables('./saved_models')

    #     if ep == 0:
    #         tf.train.load_checkpoint('./saved_models')

    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.get_action(state)
        actionDigit = agent.get_action(state)
        if actionDigit > 11 or actionDigit < 0:
            print("Not found")
            actionDigit = 10 
        action = actions[str(actionDigit)]
        # print("ActionDigit:", actionDigit)
        # print("Action:", action)
        # print("State:", state)
        next_state, reward, done, info = env.step(action)
        agent.train(state, actionDigit, next_state, reward, done)
        # env.render()
        total_reward += reward
        state = next_state
        print("next_state:", next_state)
        print("Reward:", reward)
        print("Done:", done)
        # time.sleep(10)

    # if ep % 1 == 0:
        # print("Completed Training Cycle: " + str(epoch) + " out of " + str(self.num_of_epoch))
        # print("Current Loss: " + str(loss))

        # saver = tf.train.Saver()
        # saver.save(agent.sess, 'saved_models/testing')
        # print("Model saved")

    # print("Episode: {}, total_rewards: {1.2f}".format(ep, total_reward))
    print("Episode: ", ep)
    print("total_reward: ", total_reward)
