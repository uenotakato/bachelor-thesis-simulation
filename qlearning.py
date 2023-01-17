import tqdm
from common.utils import plot_total_reward
from common.utils import greedy_probs
from common.gridworld import GridWorld
import numpy as np
from collections import defaultdict
import os
import sys
import copy
# for importing the parent dirs
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.99 #割引率
        self.alpha = 0.2 #学習率
        self.epsilon = 0.1 #貪欲方策であるかそうでないか
        self.action_size = 4
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 1000)

    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        self.pi[state] = greedy_probs(self.Q, state, epsilon=0)  # 目的方策
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)  # 行動方策

np.random.seed(100)

env = GridWorld()
agent = QLearningAgent()

episodes = 500

total_reward = 0
reward_history = []
acum_reward = []
iter = 100


for i in tqdm.tqdm(range(iter)):
    for episode in range(episodes):
        state = env.reset()
        episode_reward = []
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            episode_reward.append(reward)
            agent.update(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
        reward_history.append(np.sum(episode_reward))
    acum_reward.append(copy.deepcopy(reward_history))

# 試行結果の平均を取る
acum_reward = np.array(reward_history).reshape(iter, episodes)
acum_reward = acum_reward.mean(0)

env.render_q(agent.Q)
