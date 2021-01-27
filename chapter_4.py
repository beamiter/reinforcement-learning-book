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

policy = np.zeros((22, 11, 2, 2))
policy[20:, :, :, 0] = 1
policy[:20, :, :, 1] = 1
q = evaluate_action_monte_carlo(env, policy)
v = (q * policy).sum(axis=-1)

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

plot(v)
