
import gym
import random
import numpy as np




class Agent():
    def __init__(self, env):
        self.is_discrete = \
            type(env.action_space) == gym.spaces.discrete.Discrete
        
        if self.is_discrete:
            self.action_size = env.action_space.n
            print("Action size:", self.action_size)
        else:
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape
            print("Action range:", self.action_low, self.action_high)
        
    def get_action(self, state):
        if self.is_discrete:
            action = random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low,
                    self.action_high,
                    self.action_shape)
        return action

if __name__ == "__main__":
    game_name = "CartPole-v1"
    # game_name = "MountainCar-v0"
    # game_name = "MountainCarContinuous-v0"
    # game_name = "Acrobot-v1"
    # game_name = "Pendulum-v0"
    # game_name = "FrozenLake-v0"
    

    #getting game environment
    env = gym.make(game_name)

    # print("Observation space:", env.observation_space)
    # print("Action space:", env.action_space)
    # type(env.action_space)

    #number of episode
    episode = 100

    #agent environment
    agent = Agent(env)

    #agent state
    state = env.reset()

    total_reward = 0
    for ep in range(episode):
        #selecting action
        action = agent.get_action(state)

        #performing action
        state, reward, done, info = env.step(action)

        #calculating total reward
        total_reward += reward
        
        # show game window
        env.render()    
        print("Episode: {}, Episode_rewards: {}, states: {}, info:{}, done:{}, total_reward:{}".format(ep, reward, state, info, done, total_reward))