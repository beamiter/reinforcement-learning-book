import sympy
from sympy import symbols
import gym
import numpy as np
import scipy


def bellman_func():
    v_hungry, v_full = symbols('v_hungry v_full')
    g_hungry_eat, g_hungry_none, g_full_eat, g_full_none = \
        symbols('g_hungry_eat g_hungry_none g_full_eat g_full_none')
    alpha, beta, x, y, gamma = symbols('alpha beta x y gamma')
    system = sympy.Matrix((
        (1, 0, x - 1, -x, 0, 0, 0),
        (0, 1, 0, 0, -y, y - 1, 0),
        (-gamma, 0, 1, 0, 0, 0, -2),
        ((alpha - 1) * gamma, -alpha * gamma, 0, 1, 0, 0, 4 * alpha - 3),
        (-beta * gamma, (beta - 1) * gamma, 0, 0, 1, 0, -4 * beta + 2),
        (0, -gamma, 0, 0, 0, 1, 1)
    ))
    result = sympy.solve_linear_system(system, v_hungry, v_full, g_hungry_eat, g_hungry_none,
                                       g_full_eat, g_full_none)
    print(result)


def bellman_func_optimal():
    v_hungry, v_full = symbols('v_hungry v_full')
    g_hungry_eat, g_hungry_none, g_full_eat, g_full_none = \
        symbols('g_hungry_eat g_hungry_none g_full_eat g_full_none')
    alpha, beta, gamma = symbols('alpha beta gamma')
    xy_tuple = ((0, 0), (1, 0), (0, 1), (1, 1))
    for x, y in xy_tuple:
        system = sympy.Matrix((
            (1, 0, x - 1, -x, 0, 0, 0),
            (0, 1, 0, 0, -y, y - 1, 0),
            (-gamma, 0, 1, 0, 0, 0, -2),
            ((alpha - 1) * gamma, -alpha * gamma, 0, 1, 0, 0, 4 * alpha - 3),
            (-beta * gamma, (beta - 1) * gamma, 0, 0, 1, 0, -4 * beta + 2),
            (0, -gamma, 0, 0, 0, 1, 1)
        ))
        result = sympy.solve_linear_system(system, v_hungry, v_full, g_hungry_eat, g_hungry_none,
                                           g_full_eat, g_full_none)
        msgx = 'v(hungry) = q(hungry, {} eat)'.format('' if x else 'no')
        msgy = 'v(full) = q(full, {} eat)'.format('' if y else 'no')
        print('==== {}, {} ==== x = {}, y = {} ===='.format(msgx, msgy, x, y))
        print(result)


def play_once(env, policy):
    total_reward = 0
    state = env.reset()
    while True:
        loc = np.unravel_index(state, env.shape)
        print('state: {}, loc: {}'.format(state, loc), end=' ')
        action = np.random.choice(env.nA, p=policy[state])
        state, reward, done, _ = env.step(action)
        print('action: {}, reward: {}'.format(action, reward))
        total_reward += reward
        if done:
            break
    return total_reward


def evaluate_bellman(env, policy, gamma=1.):
    a, b = np.eye(env.nS), np.zeros(env.nS)
    for state in range(env.nS - 1):
        for action in range(env.nA):
            pi = policy[state][action]
            for p, next_state, reward, done in env.P[state][action]:
                a[state, next_state] -= pi * gamma * p
                b[state] += pi * reward * p
    v = np.linalg.solve(a, b)
    q = np.zeros((env.nS, env.nA))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for p, next_state, reward, done in env.P[state][action]:
                q[state][action] += (reward + gamma + v[next_state]) * p
    return v, q


def optimal_bellman(env, gamma=1.0):
    p = np.zeros((env.nS, env.nA, env.nS))
    r = np.zeros((env.nS, env.nA))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for prob, next_state, reward, done in env.P[state][action]:
                p[state, action, next_state] += prob
                r[state][action] += reward * prob
    c = np.ones(env.nS)
    a_ub = gamma * p.reshape(-1, env.nS) - \
           np.repeat(np.eye(env.nS), env.nA, axis=0)
    b_ub = -r.reshape(-1)
    bounds = [(None, None), ] * env.nS
    res = scipy.optimize.linprog(c, a_ub, b_ub, bounds=bounds,
                                 method='interior-point')
    v = res.x
    q = r + gamma * np.dot(p, v)
    return v, q


if __name__ == '__main__':
    sympy.init_printing()
    # bellman_func()
    # bellman_func_optimal()
    env = gym.make('CliffWalking-v0')
    print(env.observation_space, env.action_space, env.nS, env.nA, env.shape)
    actions = np.ones(env.shape, dtype=int)
    print(actions.shape)
    actions[-1, :] = 0
    actions[:, -1] = 2
    optimal_policy = np.eye(4)[actions.reshape(-1)]
    print(optimal_policy.shape)
    total_reward = play_once(env, optimal_policy)
    print('total_reward: {}'.format(total_reward))
    policy = np.random.uniform(size=(env.nS, env.nA))
    print(policy.shape)
    policy = policy / np.sum(policy, axis=1)[:, np.newaxis]
    print(policy.shape)
    state_values, action_values = evaluate_bellman(env, policy)
    optimal_state_value, optimal_action_value = evaluate_bellman(env, policy)
    print(state_values.shape, action_values.shape)
    print("**********************")
    # print(optimal_state_value, optimal_action_value)
    optimal_state_values, optimal_action_values = optimal_bellman(env)
    print(optimal_state_values, '\n', optimal_action_values)
    optimal_actions = optimal_action_values.argmax(axis=1)
    print(optimal_actions)
