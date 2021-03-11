import retro
import time
import random
import tensorflow.compat.v1 as tf   
import numpy as np
from dqn_agent_tf1 import DQNAgent



def main():
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

    for ep in range(num_episode):
        tf.train.list_variables('./saved_models')

        if ep == 0:
            tf.train.load_checkpoint('./saved_models')

        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            actionDigit = agent.get_action(state)
            if actionDigit > 11 or actionDigit < 0:
                print("Not found")
                actionDigit = 10 
            action = actions[str(actionDigit)]
            # print("ActionDigit:", actionDigit)
            print("Action:", action)
            # print("State:", state)

            next_state, reward, done, info = env.step(action)
            agent.train(state, actionDigit, next_state, reward, done)
            env.render()
            total_reward += reward
            state = next_state
            # print("next_state:", next_state)
            print("Reward:", reward)
            print("Done:", done)
            # time.sleep(10)

        if ep % 1 == 0:
            # print("Completed Training Cycle: " + str(epoch) + " out of " + str(self.num_of_epoch))
            # print("Current Loss: " + str(loss))

            saver = tf.train.Saver()
            saver.save(agent.sess, 'saved_models/testing')
            print("Model saved")

        # print("Episode: {}, total_rewards: {1.2f}".format(ep, total_reward))
        print("Episode: ", ep)
        print("total_reward: ", total_reward)


if __name__ == "__main__":
    main()