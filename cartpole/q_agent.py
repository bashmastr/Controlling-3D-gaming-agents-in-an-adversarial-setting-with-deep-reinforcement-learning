from agent import Agent
import gym
import random
import numpy as np
import time


class QAgent(Agent):
    """
    The CartPole environment gives us the position of the cart, its velocity, 
    the angle of the pole and the velocity at the tip of the pole as descriptors of the state. 
    However, all of these are continuous variables. To be able to solve this problem, 
    we need to discretize these states since otherwise, it would take forever to get values 
    for each of the possible combinations of each state, despite them being bounded. 
    """
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01, random_start=True):
        super().__init__(env)
        # checking state space is discrete or not
        self.is_discrete_state = type(env.observation_space) == gym.spaces.discrete.Discrete

        if self.is_discrete_state == True:
            self.state_size = env.observation_space.n
            print("State size:", self.state_size)
        else:    
            self.state_size =  2 ** env.observation_space.shape[0]
            print("States size:", self.state_size)
            self.states_dict = {
                "0": "0000",
                "1": "0001",
                "2": "0010",
                "3": "0011",
                "4": "0100",
                "5": "0101",
                "6": "0110",
                "7": "0111",
                "8": "1000",
                "9": "1001",
                "10": "1010",
                "11": "1011",
                "12": "1100",
                "13": "1101",
                "14": "1110",
                "15": "1111"
            }
            
        
        self.epsilon = 1.0
        # how much you give weigth to the later reward
        self.discount_rate = discount_rate
        # how much you want to give update
        self.learning_rate = learning_rate
        # creating model
        self.build_model(random_start)
        
    def build_model(self, random_start=True):
        # making q-table
        if random_start == True:
            self.q_table = -2*np.random.random([self.state_size, self.action_size])
            self.save_q_table()
        else:
            self.q_table = -2*np.random.random([self.state_size, self.action_size])
            self.load_q_table()


    def save_q_table(self):
        with open("q_learning_q_table.txt", "a+") as f:
            for i in self.q_table:
                for j in i:
                    f.write(str(j)+",")

            f.write("\n")

    def load_q_table(self):
        with open("q_learning_q_table.txt", "r") as f:
            lines_list = f.readlines()
            line = lines_list[len(lines_list)-1]
            line = line.split(",")
            count = 0
            for i in range(len(self.q_table)):
                for j in range(len(self.q_table[0])):
                    self.q_table[i][j] = float(line[count])
                    count += 1
    
    def save_summary(self, lst):
        time.time()
        with open("q_learning_logs.txt", "a+") as f:
            f.write(str(time.time())+" : ")
            for i in lst:
                f.write(str(i)+" , ")

            f.write("\n")
        
    def get_action(self, state):
        # getting q-states from current state
        q_state = self.q_table[state]
        # taking action greedy from q-state (exploit)
        action_greedy = np.argmax(q_state)
        # taking action random from q-state (explore)
        action_random = super().get_action(state)
        # returning action on the basis of epsilon value
        return action_random if random.random() < self.epsilon else action_greedy

    # discretization the state space
    def get_state(self, state):
        # checking states is discrete or not
        if not self.is_discrete_state:
            current_state = ""
            for i in range(len(state)):
                # if state value is greater then 0 we consider it 1 else consider it 0
                if state[i] > 0:
                    current_state += "1"
                else:
                    current_state += "0"
            # maping the current state to get the one digit that represent the state         
            for k in self.states_dict:
                if self.states_dict[k] == current_state:
                    return int(k)
            return 0
        else:
            return state


    
    def train(self, experience):
        state, action, next_state, reward, done = experience
        
        # what will be the reward if we move optimally from the next state
        q_next = self.q_table[next_state]
        # if done is true then the next state is terminating state, reward after the next state will be zero
        q_next = np.zeros([self.action_size]) if done else q_next
        # current reward we get from moving state to next state + discount_rate * maximum reward we get after next state
        q_target = reward + self.discount_rate * np.max(q_next)
        # getting difference between current q-value and target q-value
        q_update = q_target - self.q_table[state,action]
        # updating the q-value multiply with the learning rate
        self.q_table[state,action] += self.learning_rate * q_update
        
        # updating the epsilon after every episode 
        if done:
            self.epsilon = self.epsilon * 0.99