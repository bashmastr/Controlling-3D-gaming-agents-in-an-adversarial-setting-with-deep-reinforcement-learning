
from agent import Agent
import gym
import random
import numpy as np
import time

class QAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01, random_start=True):
        super().__init__(env)
        self.state_size = env.observation_space.n
        print("State size:", self.state_size)
        
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
            self.q_table = 1e-4*np.random.random([self.state_size, self.action_size])
            self.save_q_table()
        else:
            self.q_table = 1e-4*np.random.random([self.state_size, self.action_size])
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