import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import glorot_uniform
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

# episodes = 300
# episode_rewards = []
# for episode in range(episodes):
# episode_reward = c5.play_sarsa(env, agent, train=True)
# episode_rewards.append(episode_reward)
# plt.plot(episode_rewards)
# plt.show()


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


# agent = SARSALambdaAgent(env)
# episodes = 300
# episode_rewards = []
# for episode in range(episodes):
# episode_reward = c5.play_sarsa(env, agent, train=True)
# episode_rewards.append(episode_reward)
# plt.plot(episode_rewards)
# plt.show()


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
        self.i = (self.i + 1) % self.capicity
        self.count = min(self.count + 1, self.capicity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field])
                for field in self.memory.columns)


class DQNAgent:
    def __init__(self,
                 env,
                 net_kwargs={},
                 gamma=0.99,
                 epsilon=0.001,
                 replayer_capacity=10000,
                 batch_size=64):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity)

        self.evaluate_net = self.build_network(input_size=observation_dim,
                                               output_size=self.action_n,
                                               **net_kwargs)
        self.target_net = self.build_network(input_size=observation_dim,
                                             output_size=self.action_n,
                                             **net_kwargs)

        self.target_net.set_weights(self.evaluate_net.get_weights())

    def build_network(self,
                      input_size,
                      hidden_sizes,
                      output_size,
                      activation=tf.nn.relu,
                      output_activation=None,
                      learning_rate=0.01):
        model = keras.Sequential()
        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs = dict(input_shape=(input_size, )) if not layer else {}
            model.add(
                keras.layers.Dense(units=hidden_size,
                                   activation=activation,
                                   kernel_initializer=glorot_uniform(seed=0),
                                   **kwargs))
        model.add(
            keras.layers.Dense(units=output_size,
                               activation=activation,
                               kernel_initializer=glorot_uniform(seed=0)))
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        model.compile(loss="mse", optimizer=optimizer)
        return model

    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation,
                            done)

        observations, actions, rewards, next_observation, dones = \
            self.replayer.sample(self.batch_size)

        next_qs = self.target_net.predict(next_observation)
        next_max_qs = next_qs.max(axis=-1)
        us = rewards + self.gamma * (1. - dones) * next_max_qs
        targets = self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)

        if done:
            self.target_net.set_weights(self.evaluate_net.get_weights())

    def decide(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        qs = self.evaluate_net.predict(observation[np.newaxis])
        return np.argmax(qs)


net_kwargs = {
    'hidden_sizes': [
        64,
    ],
    'learning_rate': 0.01
}
agent = DQNAgent(env, net_kwargs=net_kwargs)

episodes = 300
episode_rewards = []
for episode in range(episodes):
    episode_reward = c5.play_qlearning(env, agent, train=True)
    episode_rewards.append(episode_reward)
plt.plot(episode_rewards)
plt.show()


class DoubleDQNAgent(DQNAgent):
    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation,
                            done)
        observations, actions, rewards, next_observations, dones = \
            self.replayer.sample(self.batch_size)
        next_eval_qs = self.evaluate_net.predict(next_observations)
        next_actions = next_eval_qs.argmax(axis=-1)
        next_qs = self.target_net.predict(next_observations)
        next_max_qs = next_qs[np.arange(next_qs.shape[0]), next_actions]
        us = rewards + self.gamma * next_max_qs * (1. - dones)
        targets = self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)

        if done:
            self.target_net.set_weights(self.evaluate_net.get_weights())


net_kwargs = {
    'hidden_sizes': [
        64,
    ],
    'learning_rate': 0.01
}
agent = DoubleDQNAgent(env, net_kwargs=net_kwargs)

episodes = 300
episode_rewards = []
for episode in range(episodes):
    episode_reward = c5.play_qlearning(env, agent, train=True)
    episode_rewards.append(episode_reward)

plt.plot(episode_rewards)
plt.show()
