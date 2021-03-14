import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from chapter_5 import play_qlearning


class QActorCriticAgent:
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.discount = 1.
        self.action_net = self.build_network(
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
        probs = self.action_net.predict(observation[np.newaxis])[0]
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
        grad_tensors = tape.gradient(loss_tensor, self.action_net.variables)
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
            grad_tensors = tape.gradient(loss_tensor,
                                         self.action_net.variables)
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
