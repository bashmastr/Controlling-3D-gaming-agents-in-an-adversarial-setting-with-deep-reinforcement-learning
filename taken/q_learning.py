
from q_agent import QAgent

import retro
from gym.envs.registration import register




def main():
    

    game_name = "Airstriker-Genesis"
    
    # gettig game environment
    env = retro.make(game=game_name)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print(type(env.action_space))
    
    # number of episode
    episode = 10

    # agent environment
    agent = QAgent(env, random_start=False)

    total_reward = 0
    for ep in range(episode):
        # agent state reset
        state = env.reset()
        # print(state)
        state = agent.get_state(state)
        
        # flag for terminating state
        done = False
        while not done:
            # selecting action 
            action = agent.get_action(state)
            print(action)
            # performing action
            next_state, reward, done, info = env.step(action)
            # print(next_state)
            #getting state in digits
            next_state = agent.get_state(next_state)
            # training agent
            agent.train((state,action,next_state,reward,done))
            lst = [episode, reward, state, action, next_state, done, info, total_reward]
            # moving to next state
            state = next_state
            # calculating reward
            total_reward += reward
            # print("Episode: {}, Episode_rewards: {}, states: {}, action:{}, info:{}, done:{}, total_reward:{}".format(ep, reward, state, action, info, done, total_reward))
            # show game screen
            # env.render()
            agent.save_summary(lst)
        if ep % 1000 == 0:
            agent.save_q_table()
            



if __name__ == "__main__":
    main()