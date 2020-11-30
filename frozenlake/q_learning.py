
from q_agent import QAgent
import gym
from gym.envs.registration import register
# from IPython.display import clear_output


def main():
    try:
        register(
            id='FrozenLakeNoSlip-v0',
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name' : '4x4', 'is_slippery':False},
            max_episode_steps=100,
            reward_threshold=0.78, # optimum = .8196
        )
    except:
        pass

    game_name = "FrozenLake-v0"
    # game_name = "FrozenLakeNoSlip-v0"
    
    # gettig game environment
    env = gym.make(game_name)
    
    # print("Observation space:", env.observation_space)
    # print("Action space:", env.action_space)
    # type(env.action_space)
    
    # number of episode
    episode = 100

    # agent environment
    agent = QAgent(env, random_start=False)

    total_reward = 0
    for ep in range(episode):
        # agent state reset
        state = env.reset()
        # flag for terminating state
        done = False
        while not done:
            # selecting action 
            action = agent.get_action(state)
            # performing action
            next_state, reward, done, info = env.step(action)
            # training agent
            agent.train((state,action,next_state,reward,done))
            lst = [episode, reward, state, action, next_state, done, info, total_reward]
            # moving to next state
            state = next_state
            # calculating reward
            total_reward += reward
            agent.save_summary(lst)
            print("Episode: {}, Episode_rewards: {}, states: {}, action:{}, info:{}, done:{}, total_reward:{}".format(ep, reward, state, action, info, done, total_reward))# show game screen
            env.render()
    agent.save_q_table()

if __name__ == "__main__":

    main()
            