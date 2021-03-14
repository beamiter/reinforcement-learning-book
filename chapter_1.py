#!/usr/bin/env python3

import gym

env = gym.make('MountainCar-v0')

def play_montecarlo(env, agent, render=False, train=False):
    episode_reward = 0.
    observation = env.reset()
    while True:
        if render:
            env.render(
            )
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, done)
        if done:
            break
        observation = next_observation
    return episode_reward
