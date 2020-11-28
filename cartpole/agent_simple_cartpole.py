import gym 
import random


class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        # print("Action Size: ", self.action_size)

    
    def get_action(self, state):
        
        pole_angles = state[2]
        action = 0 if pole_angles < 0 else 1
        return action



def main():
    game_name = "CartPole-v1"

    #getting game environment
    env = gym.make(game_name)

    # print("Observation space:", env.observation_space)
    # print("Action space:", env.action_space)

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

if __name__ == "__main__":
    main()
    