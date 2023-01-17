import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
from collections import defaultdict, deque
import numpy as np
from common.gridworld import GridWorld
from common.utils import greedy_probs
from common.utils import plot_total_reward
import tqdm
import copy


class SarsaAgent:
    def __init__(self):
        self.gamma = 0.99 #割引率
        self.alpha = 0.2 #学習率
        self.epsilon = 0.001 #貪欲方策にするかそうでないか
        self.action_size = 4
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        next_q = 0 if done else self.Q[next_state, next_action]

        target = reward + self.gamma * next_q
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha
        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)

np.random.seed(100)

env = GridWorld()
agent = SarsaAgent()

episodes = 500

total_reward = 0
reward_history = []
acum_reward = []
iter = 100


for i in tqdm.tqdm(range(iter)):
    for episode in range(episodes):
        state = env.reset()
        episode_reward = []
        agent.reset()
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            episode_reward.append(reward)
            agent.update(state, action, reward, done)
            if done:
                agent.update(next_state, None, None, None)
                break
            state = next_state
        reward_history.append(np.sum(episode_reward))
    acum_reward.append(copy.deepcopy(reward_history))



acum_reward = np.array(reward_history).reshape(iter, episodes)
acum_reward = acum_reward.mean(0)

#env.render_q(agent.Q)
