import gym
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from chapter_5 import play_qlearning
from chapter_6 import DQNReplayer


class QActorCriticAgent:
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.discount = 1.
        self.actor_net = self.build_network(
            output_size=self.action_n,
            output_activation=tf.nn.softmax,
            loss=keras.losses.categorical_crossentropy,
            **actor_kwargs)
        self.critic_net = self.build_network(output_size=self.action_n,
                                             **critic_kwargs)

    def build_network(self,
                      hidden_sizes,
                      output_size,
                      input_size=None,
                      activation=tf.nn.relu,
                      output_activation=None,
                      loss=keras.losses.mse,
                      learning_rate=0.01):
        model = keras.Sequential()
        for idx, hidden_size in enumerate(hidden_sizes):
            kwargs = {}
            if idx == 0 and input_size is not None:
                kwargs['input_shape'] = (input_size, )
            model.add(
                keras.layers.Dense(
                    units=hidden_size,
                    activation=activation,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(
                        seed=0),
                    **kwargs))
            optimizer = tf.keras.optimizers.Adam(learning_rate)
            model.compile(optimizer=optimizer, loss=loss)
            return model

    def decide(self, observation):
        probs = self.actor_net.predict(observation[np.newaxis])[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self,
              observation,
              action,
              reward,
              next_observation,
              done,
              next_action=None):
        x = observation[np.newaxis]
        u = self.critic_net.predict(x)
        q = u[0, action]
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)[0, action]
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor, 1e-6, 1.))
            loss_tensor = -self.discount * q * logpi_tensor
        grad_tensors = tape.gradient(loss_tensor, self.actor_net.variables)
        self.actor_net.optimizer.apply_gradients(
            zip(grad_tensors, self.actor_net.variables))

        u[0, action] = reward
        if not done:
            q = self.critic_net.predict(
                next_observation[np.newaxis])[0, next_action]
            u[0, action] += self.gamma * q
        self.critic_net.fit(x, u, verbose=0)

        if done:
            self.discount = 1.
        else:
            self.discount *= self.gamma


actor_kwargs = {
    'hidden_sizes': [
        100,
    ],
    'learning_rate': 0.0002
}
critic_kwargs = {
    'hidden_sizes': [
        100,
    ],
    'learning_rate': 0.0005
}
env = gym.make('Acrobot-v1')

agent = QActorCriticAgent(env,
                          actor_kwargs=actor_kwargs,
                          critic_kwargs=critic_kwargs)

# episodes = 300
# episode_rewards = []
# for episode in range(episodes):
# episode_reward = play_qlearning(env, agent, train=True)
# episode_rewards.append(episode_reward)
# plt.plot(episode_rewards)
# plt.show()


class AdvantageActorCriticAgent:
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.discount = 1.

        self.actor_net = self.build_network(
            output_size=self.action_n,
            output_activation=tf.nn.softmax,
            loss=keras.losses.categorical_crossentropy,
            **actor_kwargs)
        self.critic_net = self.build_network(output_size=1, **critic_kwargs)

    def build_network(self,
                      hidden_sizes,
                      output_size,
                      activation=tf.nn.relu,
                      output_activation=None,
                      loss=tf.keras.losses.mse,
                      learning_rate=0.01):
        pass

    def decide(self, observation):
        pass

    def learn(self, observation, action, reward, next_observation, done):
        x = observation[np.newaxis]
        u = reward + (1. - done) * self.gamma * self.critic_net.predict(
            next_observation[np.newaxis])
        td_error = u - self.critic_net.predict(x)

        x_tensor = tf.convert_to_tensor(observation[np.newaxis],
                                        dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)[0, action]
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor, 1e-6, 1.))
            loss_tensor = -self.discount * td_error * logpi_tensor
            grad_tensors = tape.gradient(loss_tensor, self.actor_net.variables)
            self.actor_net.optimizer.apply_gradients(
                zip(grad_tensors, self.actor_net.variables))

        self.critic_net.fit(x, u, verbose=0)
        if done:

            self.discount = 1.
        else:
            self.discount *= self.gamma


actor_kwargs = {
    'hidden_sizes': [
        100,
    ],
    'learning_rate': 0.0001
}
critic_kwargs = {
    'hidden_sizes': [
        100,
    ],
    'learning_rate': 0.0002
}
agent = AdvantageActorCriticAgent(env,
                                  actor_kwargs=actor_kwargs,
                                  critic_kwargs=critic_kwargs)


class ElibilityTraceActorCriticAgent(QActorCriticAgent):
    def __init__(self,
                 env,
                 actor_kwargs,
                 critic_kwargs,
                 gamma=0.99,
                 actor_lambda=0.9,
                 critic_lambda=0.9):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.actor_lambda = actor_lambda
        self.critic_lambda = critic_lambda
        self.gamma = gamma
        self.discount = 1.

        self.actor_net = self.build_network(input_size=observation_dim,
                                            output_size=self.action_n,
                                            output_activation=tf.nn.softmax,
                                            **actor_kwargs)
        self.critic_net = self.build_network(input_size=observation_dim,
                                             output_size=1,
                                             **critic_kwargs)
        self.actor_traces = [
            np.zeros_like(weight) for weight in self.actor_net.get_weights()
        ]
        self.critic_trace = [
            np.zeros_like(weight) for weight in self.critic_net.get_weights()
        ]

    def learn(self, observation, action, reward, next_observation, done):
        q = self.critic_net.predict(observation[np.newaxis])[0, 0]
        u = reward + (1. - done) * self.gamma * \
            self.critic_net.predict(next_observation[np.newaxis])[0, 0]
        td_error = u - q

        x_tensor = tf.convert_to_tensor(observation[np.newaxis],
                                        dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor, 1e-6, 1.))
            logpi_pick_tensor = logpi_tensor[0, action]
        grad_tensors = tape.gradient(logpi_pick_tensor,
                                     self.actor_net.variables)
        self.actor_traces = [
            self.gamma * self.actor_lambda * trace +
            self.discount * grad.numpy()
            for trace, grad in zip(self.actor_traces, grad_tensors)
        ]
        actor_grads = [
            tf.convert_to_tensor(-td_error * trace, dtype=tf.float32)
            for trace in self.actor_traces
        ]
        actor_grads_and_vars = tuple(zip(actor_grads,
                                         self.actor_net.variables))
        self.actor_net.optimizer.apply_gradients(actor_grads_and_vars)

        with tf.GradientTape() as tape:
            v_tensor = self.critic_net(x_tensor)
        grad_tensors = tape.gradient(v_tensor, self.critic_net.variables)
        self.critic_traces = [
            self.gamma * self.critic_lambda * trace +
            self.discount * grad.numpy()
            for trace, grad in zip(self.critic_traces, grad_tensors)
        ]
        critic_grads = [
            tf.convert_to_tensort(-td_error * trace, dtype=tf.float32)
            for trace in self.critic_traces
        ]
        critic_grads_and_vars = tuple(
            zip(critic_grads, self.critic_net.variables))
        self.critic_net.optimizer.apply_gradienst(critic_grads_and_vars)

        if done:
            self.actior_traces = [
                np.zeros_like(weight)
                for weight in self.actor_net.get_weights()
            ]
            self.critic_traces = [
                np.zeros_like(weight)
                for weight in self.critic_net.get_weights()
            ]
            self.discount = 1.
        else:
            self.discount *= self.gamma


actor_kwargs = {
    'hidden_sizes': [
        100,
    ],
    'learning_rate': 0.001
}
critic_kwargs = {
    'hidden_sizes': [
        100,
    ],
    'learning_rate': 0.001
}
agent = ElibilityTraceActorCriticAgent(env,
                                       actor_kwargs=actor_kwargs,
                                       critic_kwargs=critic_kwargs)


class PPOReplayer:
    def __init__(self):
        self.memory = pd.DataFrame()

    def store(self, df):
        self.memory = pd.concat([self.memory, df], ignore_index=True)

    def sample(self, size):
        indices = np.random.choice(self.memory.shape[0], size=size)
        return (np.stack(self.memory.loc[indices, field])
                for field in self.memory.columns)


class PPOAgent(QActorCriticAgent):
    def __init__(self,
                 env,
                 actor_kwargs,
                 critic_kwargs,
                 clip_ratio=0.1,
                 gamma=0.99,
                 lambd=0.99,
                 min_trajectory_length=1000,
                 batches=1,
                 batch_size=64):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.lambd = lambd
        self.min_trajectory_length = min_trajectory_length
        self.batches = batches
        self.batch_size = batch_size

        self.trajectory = []
        self.replayer = PPOReplayer()

        def ppo_loss(y_true, y_pred):
            p = y_pred
            p_old = y_true[:, :self.action_n]
            advantage = y_true[:, self.action_n:]
            surrogate_advantage = (p / p_old) * advantage
            clip_times_advantage = clip_ratio * advantage
            max_surrogate_advantage = advantage + tf.where(
                advantage > 0., clip_times_advantage, -clip_times_advantage)
            clipped_surrogate_advantage = tf.minimum(surrogate_advantage,
                                                     max_surrogate_advantage)
            return -tf.reduce_mean(clipped_surrogate_advantage, axis=-1)

        self.actor_net = self.build_network(output_size=self.action_n,
                                            output_activation=tf.nn.softmax,
                                            loss=ppo_loss,
                                            **actor_kwargs)
        self.critic_net = self.build_network(output_size=1, **critic_kwargs)

    def learn(self, observation, action, reward, done):
        self.trajectory.append((observation, action, reward))

        if done:
            df = pd.DataFrame(self.trajectory,
                              columns=['observation', 'action', 'reward'])
            observations = np.stack(df['observation'])
            df['v'] = self.critic_net.predict(observations)
            pis = self.actor_net.predict(observations)
            df['pi'] = [a.flatten() for a in np.split(pis, pis.shape[0])]

            df['next_v'] = df['v'].shift(-1).fillna(0.)
            df['u'] = df['reward'] + self.gamma * df['next_v']
            df['delta'] = df['u'] - df['v']
            df['return'] = df['reward']
            df['advantage'] = df['delta']
            for i in df.index[-2::-1]:
                df.loc[i, 'return'] += self.gamma * df.loc[i + 1, 'return']
                df.loc[i, 'advantage'] += self.gamma * self.lambd * \
                    df.loc[i + 1, 'advantage']
            fields = ['observation', 'action', 'pi', 'advantage', 'return']
            self.replayer.store(df[fields])
            self.trajectory = []

            if len(self.trajectory.memory) > self.min_trajectory_length:
                for batch in range(self.batches):
                    observations, actions, pis, advantages, returns = \
                        self.replayer.sample(size=self.batch_size)
                    ext_advantages = np.zeors_like(pis)
                    ext_advantages[range(self.batch_size), actions] = \
                        advantages
                    actor_targets = np.hstack([pis, ext_advantages])
                    self.actor_net.fit(observations, actor_targets, verbose=0)
                    self.critic_net.fit(observations, returns, verbose=0)

                self.replayer = PPOReplayer()


class SACAgent:
    def __init__(self,
                 env,
                 actor_kwargs,
                 critic_kwargs,
                 replayer_capacity=10000,
                 gamma=0.99,
                 alpha=0.99,
                 batches=1,
                 batch_size=64,
                 net_learning_rate=0.995):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.alpah = alpha
        self.net_learning_rate = net_learning_rate
        self.batches = batches
        self.batch_size = batch_size

        self.replayer = DQNReplayer(replayer_capacity)

        def sac_loss(y_true, y_pred):
            qs = alpha * tf.math.xlogy(y_pred, y_pred) - y_pred * y_true
            return tf.reduce_sum(qs, axis=-1)

        self.actor_net = self.build_network(input_size=observation_dim,
                                            output_size=self.action_n,
                                            output_activate=tf.nn.softmax,
                                            loss=sac_loss,
                                            **actor_kwargs)
        self.q0_net = self.build_network(input_size=observation_dim,
                                         output_size=self.action_n,
                                         **critic_kwargs)

        self.q0_net = self.build_network(input_size=observation_dim,
                                         output_size=self.action_n,
                                         **critic_kwargs)
        self.q1_net = self.build_network(input_size=observation_dim,
                                         output_size=self.action_n,
                                         **critic_kwargs)
        self.v_evaluate_net = self.build_network(input_size=observation_dim,
                                                 output_size=1,
                                                 **critic_kwargs)
        self.v_target_net = self.build_network(input_size=observation_dim,
                                               output_size=1,
                                               **critic_kwargs)
        self.update_target_net(self.v_target_net, self.v_evaluate_net)

    def build_network(self,
                      input_size,
                      hidden_sizes,
                      output_size,
                      activation=tf.nn.relu,
                      output_activation=None,
                      loss=keras.losses.mse,
                      learning_rate=0.01):
        model = keras.Sequential()
        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs = {
                'input_shape': {
                    input_size,
                }
            } if layer == 0 else {}
            model.add(
                keras.layers.Dense(units=hidden_size,
                                   activation=activation,
                                   **kwargs))
        model.add(
            keras.layers.Dense(units=output_size,
                               activation=output_activation))
        optimizer = keras.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def update_target_net(self, target_net, evaluate_net, learning_rate=1.):
        target_weights = target_net.get_weights()
        evaluate_weights = evaluate_net.get_weights()
        average_weights = [(1. - learning_rate) * t + learning_rate * e
                           for t, e in zip(target_weights, evaluate_weights)]
        target_net.set_weights(average_weights)

    def decide(self, observation):
        probs = self.actor_net.predict(observation[np.newaxis])[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation,
                            done)

        if done:
            for batch in range(self.batches):
                observations, actions, rewards, next_observations,
                dones = self.replayer.sample(self.batch_size)
                pis = self.actor_net.predict(observation)
                q0s = self.q0_net.predict(observation)
                q1s = self.q1_net.predict(observation)

                self.actor_net.fit(observation, q0s, verbose=0)

                q01s = np.minimum(q0s, q1s)
                entropic_q01s = q01s - self.alpha * np.log(pis)
                v_targets = (pis * entropic_q01s).mean(axis=1)
                self.v_evaluate_net.fit(observations, v_targets, verbose=0)

                next_vs = self.v_target_net.predict(next_observations)
                q_targets = rewards + self.gamma * (1. - done) * next_vs[:, 0]
                q0s[range(self.batch_size), actions] = q_targets
                q1s[range(self.batch_size), actions] = q_targets
                self.q0_net.fit(observations, q0s, verbose=0)
                self.q1_net.fit(observations, q1s, verbose=0)

                self.update_target_net(self.v_target_net, self.v_evaluate_net,
                                       self.net_learning_rate)
