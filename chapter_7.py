#!/usr/bin/env python3

import chapter_1
from chapter_1 import play_montecarlo
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

env = gym.make('CartPole-v0')


class VPGAgent:
    def __init__(self, env, policy_kwargs, baseline_kwargs=None, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma

        self.trajectory = []

        self.policy_net = self.build_network(
            output_size=self.action_n,
            output_activation=tf.nn.softmax,
            loss=keras.losses.categorical_crossentropy,
            **policy_kwargs)
        if baseline_kwargs:
            self.baseline_net = self.build_networdk(output_size=1,
                                                    **baseline_kwargs)

    def build_network(self,
                      hidden_sizes,
                      output_size,
                      activation=tf.nn.relu,
                      output_activation=None,
                      loss=keras.losses.mse,
                      learning_rate=0.01):
        model = keras.Sequential()
        for hidden_size in hidden_sizes:
            model.add(
                keras.layers.Dense(units=hidden_size, activation=activation))
        model.add(
            keras.layers.Dense(units=output_size,
                               activation=output_activation))
        optimizer = keras.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def decide(self, observation):
        probs = self.policy_net.predict(observation[np.newaxis])[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, reward, done):
        self.trajectory.append((observation, action, reward))

        if done:
            df = pd.DataFrame(self.trajectory,
                              columns=['observation', 'action', 'reward'])
            df['discount'] = self.gamma**df.index.to_series()
            df['discounted_reward'] = df['discount'] * df['reward']
            df['discounted_return'] = df['discounted_reward'][::-1].cumsum()
            df['psi'] = df['discounted_return']

            x = np.stack(df['observation'])
            if hasattr(self, 'baseline_net'):
                df['baseline'] = self.baseline_net.predict(x)
                df['psi'] -= (df['baseline']) * (df['discount'])
                df['return'] = df['discounted_return'] / df['discount']
                y = df['return'].values[:, np.newaxis]
                self.baseline_net.fit(x, y, verbose=0)

            y = np.eye(self.action_n)[df['action']] * \
                df['psi'].values[:, np.newaxis]
            self.policy_net.fit(x, y, verbose=0)

            self.trajectory = []


policy_kwargs = {
    'hidden_sizes': [
        10,
    ],
    'activation': tf.nn.relu,
    'learning_rate': 0.01
}
agent = VPGAgent(env, policy_kwargs=policy_kwargs)

# policy_kwargs = {'hidden_sizes': [10,], 'activation': tf.nn.relu,
#                  'learning_rate': 0.01}
# baseline_kwargs = {'hidden_sizes': [10,], 'activation': tf.nn.relu,
#                    'learning_rate': 0.01}
# agent = VPGAgent(env, policy_kwargs=policy_kwargs, baseline_kwargs=baseline_kwargs)

# episodes = 500
# episode_rewards = []
# for episode in range(episodes):
# episode_reward = play_montecarlo(env, agent, train=True)
# episode_rewards.append(episode_reward)
# plt.plot(episode_rewards)
# plt.show()


class OffPolicyVPGAgent(VPGAgent):
    def __init__(self, env, policy_kwargs, baseline_kwargs=None, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma

        self.trajectory = []

        def dot(y_true, y_pred):
            return -tf.reduce_sum(y_true * y_pred, axis=-1)

        self.policy_net = self.build_network(output_size=self.action_n,
                                             output_activation=tf.nn.softmax,
                                             loss=dot,
                                             **policy_kwargs)
        if baseline_kwargs:
            self.baseline_net = self.build_network(output_size=1,
                                                   **baseline_kwargs)

    def learn(self, observation, action, behavior, reward, done):
        self.trajectory.append((obseration, action, behavior, reward))

        if done:
            df = pd.DataFrame(
                self.trajectory,
                columns=['observation', 'action', 'behavior', 'reward'])
            df['discount'] = self.gamma**df.index.to_series()
            df['discounted_reward'] = df['discount'] * df['reward']
            df['discounted_return'] = \
                df['discounted_reward'][::-1].cumsum()
            df['psi'] = df['discounted_return']

            x = np.stack(df['observation'])
            if hasattr(self, 'baseline_net'):
                df['baseline'] = self.baseline_net.predict(x)
                df['psi'] -= df['baseline'] * df['discount']
                df['return'] = df['discounted_return'] / df['discount']
                y = df['return'].values[:, np.newaxis]
                self.baseline_net.fit(x, y, verbose=0)

            y = np.eye(self.action_n)[df['action']] * \
                (df['psi'] / df['behavior']).values[:, np.newaxis]
            self.policy_net.fit(x, y, verbose=0)
            self.trajectory = []


policy_kwargs = {
    'hidden_sizes': [
        10,
    ],
    'activation': tf.nn.relu,
    'learning_rate': 0.01
}
agent = OffPolicyVPGAgent(env, policy_kwargs=policy_kwargs)


class RandomAgent:
    def __init__(self, env):
        self.action_n = env.action_space.n

    def decide(self, observation):
        action = np.random.choice(self.action_n)
        behavior = 1. / self.action_n
        return action, behavior


behavior_agent = RandomAgent(env)

episodes = 1500
episode_rewards = []
for episode in range(episodes):
    obseration = env.reset()
    episode_reward = 0.
    while True:
        action, behavior = behavior_agent.decide(obseration)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        agent.learn(obseration, action, behavior, reward, done)
        if done:
            break
        observation = next_observation

    episode_reward = play_montecarlo(env, agent)
    episode_rewards.append(episode_reward)

plt.plot(episode_rewards)
plt.show()
