import gym
import matplotlib.pyplot as plt
import numpy as np
env = gym.make("Blackjack-v0")
observation = env.reset()
print("observation: {}".format(observation))

while True:
    print("{}, {}".format(env.player, env.dealer))
    action = np.random.choice(env.action_space.n)
    print("{}".format(action))
    observation, reward, done, _ = env.step(action)
    print("{}, {}, {}".format(observation, reward, done))
    if done:
        break

def ob2state(observation):
    return (observation[0], observation[1], int(observation[2]))

def evaluate_action_monte_carlo(env, policy, episode_num=500000):
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        state_actions = []
        observation = env.reset()
        while True:
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n, p=policy[state])
            state_actions.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                break
        g = reward
        for state, action in state_actions:
            c[state][action] += 1
            q[state][action] += (g - q[state][action]) / c[state][action]
    return q

# policy = np.zeros((22, 11, 2, 2))
# policy[20:, :, :, 0] = 1
# policy[:20, :, :, 1] = 1
# q = evaluate_action_monte_carlo(env, policy)
# v = (q * policy).sum(axis=-1)

def plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    titles = ['without ace', 'with ace']
    have_aces = [0, 1]
    extent = [12, 22, 1, 11]
    for title, have_ace, axis in zip(titles, have_aces, axes):
        dat = data[extent[0]:extent[1], extent[2]:extent[3], have_ace].T
        axis.imshow(dat, extent=extent, origin='lower')
        axis.set_xlabel('player sum')
        axis.set_ylabel('dealer showing')
        axis.set_title(title)
    plt.show()

# plot(v)

def monte_carlo_with_exploring_start(env, episode_num=500000):
    policy = np.zeros((22, 11, 2, 2))
    policy[:, :, :, 1] = 1.
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        state = (np.random.randint(12, 22),
                 np.random.randint(1, 11),
                 np.random.randint(2))
        action = np.random.randint(2)
        env.reset()
        if state[2:]:
            env.player = [1, state[0] - 11]
        else:
            if state[0] == 21:
                env.player = [10, 9, 2]
            else:
                env.player = [10, state[0] - 10]
        env.dealer[0] = state[1]
        state_actions = []
        while True:
            state_actions.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                break
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n, p=policy[state])
        g = reward
        for state, action in state_actions:
            c[state][action] += 1.
            q[state][action] += (g - q[state][action]) / c[state][action]
            a = q[state].argmax()
            policy[state] = 0.
            policy[state][a] = 1.
        return policy, q

# policy, q = monte_carlo_with_exploring_start(env)
# v = q.max(axis=-1)
# plot(policy.argmax(-1));
# plot(v)

def monte_carlo_with_soft(env, episode_num=500000, epsilon=0.1):
    policy = np.ones((22, 11, 2, 2)) * 0.5
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        state_actions = []
        observation = env.reset()
        while True:
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n, p=policy[state])
            state_actions.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                break
        g = reward
        for state, action in state_actions:
            c[state][action] += 1.
            q[state][action] += (g - q[state][action]) / c[state][action]
            a = q[state].argmax()
            policy[state] = epsilon / 2.
            policy[state][a] += (1. - epsilon)
    return policy, q

#policy, q = monte_carlo_with_soft(env)
#v = q.max(axis=-1)
#plot(policy.argmax(-1))
#plot(v)

def evaluate_monte_carlo_importance_sampling(env, policy, behavior_policy,
                                             episode_num=500000):
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        state_actions = []
        observation = env.reset()
        while True:
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n,
                                      p=behavior_policy[state])
            state_actions.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                break
        g = reward
        rho = 1.0
        for state, action in reversed(state_actions):
            c[state][action] += rho
            q[state][action] += (rho / c[state][action] * (g - q[state][action]))
            rho *= (policy[state][action] / behavior_policy[state][action])
            if rho == 0:
                break
    return q

#policy = np.zeros((22, 11, 2, 2))
#policy[20:, :, :, 0] = 1
#policy[:20, :, :, 1] = 1
#behavior_policy = np.ones_like(policy) * 0.5
#q = evaluate_monte_carlo_importance_sampling(env, policy, behavior_policy)
#v = (q * policy).sum(axis=-1)
#plot(v)

def monte_carlo_importance_resample(env, episode_num=500000):
    policy = np.zeros((22, 11, 2, 2))
    policy[:, :, :, 0] = 1.
    behavior_policy = np.ones_like(policy) * 0.5
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        state_actions = []
        observation = env.reset()
        while True:
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n,
                                      p=behavior_policy[state])
            state_actions.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                break
        g = reward
        rho = 1.
        for state, action in reversed(state_actions):
            c[state][action] += rho
            q[state][action] += (rho / c[state][action] * (g - q[state][action]))
            a = q[state].argmax()
            policy[state] = 0.
            policy[state][a] = 1.
            if a != action:
                break
            rho /= behavior_policy[state][action]
    return policy, q

policy, q = monte_carlo_importance_resample(env)
v = q.max(axis=-1)
plot(policy.argmax(-1));
plot(v)
