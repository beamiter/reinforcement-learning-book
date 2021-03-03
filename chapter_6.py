import chapter_5 as c5
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
env = gym.make('MountainCar-v0')
env = env.unwrapped
print('observation: {}'.format(env.observation_space))
print('action: {}'.format(env.action_space))
print('position: {}'.format((env.min_position, env.max_position)))
print('speed: {}'.format((-env.max_speed, env.max_speed)))
print('target: {}'.format(env.goal_position))

positions, velocities = [], []
observation = env.reset()
n = 0
while True:
    positions.append(observation[0])
    velocities.append(observation[1])
    next_observation, reward, done, _ = env.step(2)
    if done or n > 200:
        break
    observation = next_observation
    n += 1

if next_observation[0] > 0.5:
    print('success')
else:
    print('fail')

# fig, ax = plt.subplots()
# ax.plot(positions, label='positon')
# ax.plot(velocities, label='velocity')
# ax.legend()
# fig.show()
# plt.show()


class TileCoder:
    def __init__(self, layers, features):
        self.layers = layers
        self.features = features
        self.codebook = {}

    def get_feature(self, codeword):
        if codeword in self.codebook:
            return self.codebook[codeword]
        count = len(self.codebook)
        if count >= self.features:
            return hash(codeword) % self.features
        else:
            self.codebook[codeword] = count
            return count

    def __call__(self, floats=(), ints=()):
        dim = len(floats)
        scaled_floats = tuple(f * self.layers * self.layers for f in floats)
        features = []
        for layer in range(self.layers):
            codeword = (layer, ) + tuple(
                int((f + (1 + dim * i) * layer) / self.layers)
                for i, f in enumerate(scaled_floats)) + ints
            feature = self.get_feature(codeword)
            features.append(feature)
        return features


class SARSAAgent:
    def __init__(self,
                 env,
                 layers=9,
                 features=1893,
                 gamma=1.,
                 learning_rate=0.03,
                 epsilon=0.001):
        self.action_n = env.action_space.n
        self.obs_low = env.observation_space.low
        self.obs_scale = env.observation_space.high - env.observation_space.low
        self.encoder = TileCoder(layers, features)
        self.w = np.zeros(features)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def encode(self, observation, action):
        states = tuple((observation - self.obs_low) / self.obs_scale)
        actions = (action, )
        return self.encoder(states, actions)

    def get_q(self, observation, action):
        features = self.encode(observation, action)
        return self.w[features].sum()

    def decide(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        else:
            qs = [
                self.get_q(observation, action)
                for action in range(self.action_n)
            ]
            return np.argmax(qs)

    def learn(self, observation, action, reward, next_observation, done,
              next_action):
        u = reward + (1. - done) * self.gamma * self.get_q(
            next_observation, next_action)
        td_error = u - self.get_q(observation, action)
        features = self.encode(observation, action)
        self.w[features] += (self.learning_rate * td_error)


agent = SARSAAgent(env)

episodes = 300
episode_rewards = []
for episode in range(episodes):
    episode_reward = c5.play_sarsa(env, agent, train=True)
    episode_rewards.append(episode_reward)
plt.plot(episode_rewards)
plt.show()


class SARSALambdaAgent(SARSAAgent):
    def __init__(self,
                 env,
                 layers=8,
                 features=1893,
                 gamma=1.,
                 learning_rate=0.03,
                 epsilon=0.001,
                 lambd=0.9):
        super().__init__(env=env,
                         layers=layers,
                         features=features,
                         gamma=gamma,
                         learning_rate=learning_rate,
                         epsilon=epsilon)
        self.lambd = lambd
        self.z = np.zeros(features)

    def learn(self, observation, action, reward, next_observation, done,
              next_action):
        u = reward
        if not done:
            u += (self.gamma * self.get_q(next_observation, next_action))
            self.z *= (self.gamma * self.lambd)
            features = self.encode(observation, action)
            self.z[features] = 1.
        td_error = u - self.get_q(observation, action)
        self.w += (self.learning_rate * td_error * self.z)
        if done:
            self.z = np.zeros_like(self.z)


agent = SARSALambdaAgent(env)
episodes = 300
episode_rewards = []
for episode in range(episodes):
    episode_reward = c5.play_sarsa(env, agent, train=True)
    episode_rewards.append(episode_plt)
reward.plot(episode_rewards)
plt.show()


class DQNReplayer():
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=[
                                       'observation', 'action', 'reward',
                                       'next_observation', 'done'
                                   ])
        self.i = 0
        self.count = 0
        self.capicity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + i) % self.capicity
        self.count = min(self.count + 1, self.capicity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field])
                for field in self.memory.columns)
