import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('Taxi-v3')
state = env.reset()
taxirow, taxicol, passloc, destidx = env.unwrapped.decode(state)
print("taxi pos: {}".format((taxirow, taxicol)))
print("pass loc: {}".format(env.unwrapped.locs[passloc]))
print("dest loc: {}".format(env.unwrapped.locs[destidx]))
env.render()
env.step(1)


class SARSAAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=0.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state):
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done, next_action):
        u = reward + self.gamma * \
            self.q[next_state, next_action] * (1. - done)
        td_error = u - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error


agent = SARSAAgent(env)


def play_sarsa(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        next_action = agent.decide(next_observation)
        if train:
            agent.learn(observation, action, reward, next_observation, done,
                        next_action)
        if done:
            break
        observation, action = next_observation, next_action
    return episode_reward


episodes = 5000
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_sarsa(env, agent, train=True)
    episode_rewards.append(episode_reward)
# plt.plot(episode_rewards)
# plt.show()

agent.epsilon = 0.
episode_rewards = [play_sarsa(env, agent) for _ in range(100)]
print('average episode reward: {} / {} = {}'.format(sum(episode_rewards),
                                                    len(episode_rewards),
                                                    np.mean(episode_rewards)))


class ExpectedSARSAAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=0.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q = np.zeros((env.observation_space.n, env.action_space.n))
        self.action_n = env.action_space.n

    def decide(self, state):
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done):
        v = (self.q[next_state].sum() * self.epsilon +
             self.q[next_state].max() * (1. - self.epsilon))
        u = reward + self.gamma * v * (1. - done)
        td_error = u - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error


agent = ExpectedSARSAAgent(env)


def play_qlearning(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, next_observation, done)
        if done:
            break
        observation = next_observation
    return episode_reward


episodes = 5000
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_qlearning(env, agent, train=True)
    episode_rewards.append(episode_reward)
# plt.plot(episode_rewards)
# plt.show()

agent.epsilon = 0.
episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
print("average reward: {} / {} = {}".format(sum(episode_rewards),
                                            len(episode_rewards),
                                            np.mean(episode_rewards)))


class QLearningAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=0.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state):
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randinit(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done):
        u = reward + self.gamma * self.q[next_state].max() * (1. - done)
        td_error = u - self.q[state][action]
        self.q[state][action] += self.learning_rate * td_error


agent = QLearningAgent(env)


class DoubleQlearningAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=0.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.action_space.n
        self.q0 = np.zeros((env.observation_space.n, env.action_space.n))
        self.q1 = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state):
        if np.random.uniform() > self.epsilon:
            action = (self.q0 + self.q1)[state].argmax()
        else:
            action = np.random.randinit(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done):
        if np.random.randint(2):
            self.q0, self.q1 = self.q1, self.q0
        a = self.q0[next_state].argmax()
        u = reward + self.gamma + self.q1[next_state, a] * (1. - done)
        td_error = u - self.q0[state, action]
        self.q0[state, action] += self.learning_rate * td_error


agent = DoubleQlearningAgent(env)


class SARSALambdaAgent(SARSAAgent):
    def __init__(self,
                 env,
                 lambd=0.5,
                 beta=1.,
                 gamma=0.9,
                 learning_rate=0.1,
                 epsilon=.01):
        super().__init__(env,
                         gamma=gamma,
                         learning_rate=learning_rate,
                         epsilon=epsilon)
        self.lambd = lambd
        self.beta = beta
        self.e = np.zeros((env.observation_space.n, env.action_space.n))

    def learn(self, state, action, reward, next_state, done, next_action):
        self.e *= (self.lambd * self.gamma)
        self.e[state][action] = 1. + self.beta * self.e[state][action]

        u = reward + self.gamma * \
            self.q[next_state, next_action] * (1. - done)
        td_error = u - self.q[state, action]
        self.q += self.learning_rate * self.e * td_error

        if done:
            self.e *= 0.


agent = SARSALambdaAgent(env)
