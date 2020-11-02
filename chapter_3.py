import gym
import numpy as np


def play_policy(env, policy, render=False):
    total_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = np.random.choice(env.action_space.n, p=policy[observation])
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def v2q(env, v, s=None, gamma=1.0):
    if s is not None:
        q = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                q[a] += prob * (reward + gamma * v[next_state] * (1. - done))
    else:
        q = np.zeros((env.nS, env.nA))
        for s in range(env.nS):
            q[s] = v2q(env, v, s, gamma)
    return q


def evaluate_policy(env, policy, gamma=1., tolerant=1e-6):
    v = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            vs = np.sum(policy[s] * v2q(env, v, s, gamma))
            delta = max(delta, abs(v[s] - vs))
            v[s] = vs
        if delta < tolerant:
            break
    return v


def imporve_policy(env, v, policy, gamma=1.):
    optimal = True
    for s in range(env.nS):
        q = v2q(env, v, s, gamma)
        a = np.argmax(q)
        if policy[s][a] != 1.0:
            optimal = False
            policy[s] = 0.
            policy[s][a] = 1.
    return optimal


def iterate_policy(env, gamma=1., tolerant=1e-6):
    policy = np.ones((env.nS, env.nA)) / env.nA
    while True:
        v = evaluate_policy(env, policy, gamma, tolerant)
        if imporve_policy(env, v, policy):
            break
    return policy, v


def iterate_value(env, gamma=1., tolerant=1e-6):
    v = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            vmax = max(v2q(env, v, s, gamma))
            delta = max(delta, abs(v[s] - vmax))
            v[s] = vmax
        if delta < tolerant:
            break

    policy = np.zeros((env.nS, env.nA)) / env.nA
    for s in range(env.nS):
        a = np.argmax(v2q(env, v, s, gamma))
        policy[s][a] = 1.
    return policy, v


if __name__ == '__main__':
    print('enter this main junction')
    env = gym.make('FrozenLake-v0')
    env = env.unwrapped
    print(env.observation_space, env.action_space)
    print(env.P[14][2])
    random_policy = np.ones((env.nS, env.nA)) / env.nA
    episode_rewards = [play_policy(env, random_policy) for i in range(100)]
    print('mean_reward: {}'.format(np.mean(episode_rewards)))
    v_rand = evaluate_policy(env, random_policy)
    print(v_rand.reshape(4, 4))
    q_rand = v2q(env, v_rand)
    print(q_rand)
    policy = random_policy.copy()
    optimal = imporve_policy(env, v_rand, policy)
    if optimal:
        print('no imporve, policy: ')
    else:
        print('has imporve, policy: ')
    print(policy)
    policy_pi, v_pi = iterate_policy(env)
    print('v_pi: ')
    print(v_pi.reshape(4, 4))
    print('policy_pi: ')
    print(np.argmax(policy_pi, axis=1).reshape(4, 4))
    policy_pi, v_pi = iterate_value(env)
    print('v_pi: ')
    print(v_pi.reshape(4, 4))
    print('policy_pi: ')
    print(np.argmax(policy_pi, axis=1).reshape(4, 4))
    episode_rewards = [play_policy(env, policy_pi) for _ in range(100)]
    print('value iteration mean reward: {}'.format(np.mean(episode_rewards)))

