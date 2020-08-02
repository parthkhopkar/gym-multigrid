from replay_memory import ReplayMemory
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten

class PPOModel(tf.keras.Model):
    def __init__(self, num_inputs, num_outputs, hidden_sizes):
        super(PPOModel, self).__init__()
        self.actor_network = keras.Sequential(name='actor_network')

        for i, size in enumerate(hidden_sizes):
            self.actor_network.add(Dense(size, activation='relu', name=f'actor_dense_{i}'))
        self.actor_network.add(Dense(num_outputs))

        self.critic_network = keras.Sequential(name='critic_network')

        for i, size in enumerate(hidden_sizes):
            self.critic_network.add(Dense(size, activation='relu', name=f'critic_dense_{i}'))
        self.critic_network.add(Dense(1))

    def call(self, x):
        value = self.critic_network(x)
        mu = self.actor_network(x)
        return mu, value

class PPOAgent():
    def __init__(self, num_inputs, num_outputs, hidden_sizes, lr, buffer_size=128, training_batch_size=16):
        super(PPOAgent, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_sizes = hidden_sizes
        self.LR = lr
        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.95
        self.CLIP_PARAM = 0.2
        self.CRITIC_DISCOUNT = 0.5
        self.ENTROPY_BETA = 0.001
        self.PPO_EPOCHS = 4
        self.BUFFER_SIZE = buffer_size
        self.TRAINING_BATCH_SIZE = training_batch_size
        self.replay_memory = ReplayMemory()
        self.model = PPOModel(self.num_inputs, self.num_outputs, self.hidden_sizes)
        self.model.build(input_shape=(None, self.num_inputs))
        self.model.summary()
        # self.critic_network = self._build_critic_network()
        # self.actor_network = self._build_actor_network()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.LR)

        # Some checks
        if self.BUFFER_SIZE % self.TRAINING_BATCH_SIZE != 0:
            print("WARNING: Buffer size is not divisible by batch size. All collected trajectories will not be used during training")

    def _build_critic_network(self):
        critic_network = keras.Sequential(name='critic_network')
        critic_network.add(Input(shape=(self.num_inputs, )))
        for i, size in enumerate(self.hidden_sizes):
            critic_network.add(Dense(size, activation='relu', name=f'critic_dense_{i}'))
        critic_network.add(Dense(1))

        # TODO Compile critic
        # critic_network.compile()

        critic_network.summary()

        return critic_network

    def _build_actor_network(self):
        actor_network = keras.Sequential(name='actor_network')
        actor_network.add(Input(shape=(self.num_inputs, )))
        for i, size in enumerate(self.hidden_sizes):
            actor_network.add(Dense(size, activation='relu', name=f'actor_dense_{i}'))
        actor_network.add(Dense(self.num_outputs))

        # TODO Compile actor
        # actor_network.compile()

        actor_network.summary()

        return actor_network

    def get_action(self, obs):
        # TODO Return most likely action if not training

        # Flatten observation
        obs = np.array(obs)

        obs = obs.reshape((1, self.num_inputs))
        assert obs.shape[1] == self.num_inputs
        policy_logits, value = self.model(obs)
        pi = tfp.distributions.Categorical(logits=policy_logits)
        action = pi.sample()

        log_prob = pi.log_prob(action)
        self.replay_memory.batch_log_probs.append(log_prob)
        self.replay_memory.batch_v.append(value)

        return action

    def store_transition(self, obs, action, reward, next_obs, done):
        obs = np.array(obs).reshape((-1, self.num_inputs))
        next_obs = np.array(next_obs).reshape((-1, self.num_inputs))
        self.replay_memory.store(obs, action, reward, next_obs, done)

    def train(self):
        # Get GAE returns
        self.calculate_gae()

        # Get randomized minibatches
        for i in range(self.PPO_EPOCHS):
            for states, actions, old_log_probs, gae_returns, values in self.replay_memory.get_batch(self.TRAINING_BATCH_SIZE):

                assert len(states) == len(actions) == len(old_log_probs) == len(gae_returns) == len(values)

                advantages = gae_returns - values
                advantages = keras.utils.normalize(advantages)  # Normalize advantages
                
                # for state, action, old_log_prob, gae_return, advantage in zip(states, actions, log_probs, gae_returns, advantages):

                # TODO Add entropy exploration to loss
                states = tf.convert_to_tensor(states)
                with tf.GradientTape() as tape:
                    policy_logits, values = self.model(states)
                    pi = tfp.distributions.Categorical(logits=policy_logits)
                    entropy = tf.reduce_mean(pi.entropy())
                    new_log_probs = pi.log_prob(actions)

                    ratio = tf.math.exp(new_log_probs - old_log_probs)

                    surr1 = ratio * advantages
                    surr2 = tf.clip_by_value(ratio, 1.0 - self.CLIP_PARAM, 1.0 + self.CLIP_PARAM) * advantages

                    actor_loss = - tf.reduce_mean(tf.minimum(surr1, surr2))
                    critic_loss = tf.reduce_mean(tf.math.square(gae_returns - values))

                    loss = self.CRITIC_DISCOUNT * critic_loss + actor_loss - self.ENTROPY_BETA * entropy

                trainable_variables = self.model.trainable_variables
                gradients = tape.gradient(loss, trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, trainable_variables))

            print(f'PPO_epoch:{i} | Actor loss: {actor_loss} | Critic loss: {critic_loss} | Loss: {loss}')



    def calculate_gae(self):
        """Generates GAE type rewards and pushes them into memory object
        GAE algorithm: 
            delta = r + gamma * V(s') * mask - V(s)  |aka advantage
            gae = delta + gamma * lambda * mask * gae |moving average smoothing
            return(s,a) = gae + V(s)  |add value of state back to it.
        """
        gae = 0
        mask = 0
        for i in reversed(range(len(self.replay_memory))):
            mask = 0 if self.replay_memory.batch_done[i] else 1
            v = self.replay_memory.batch_v[i]

            delta = self.replay_memory.batch_r[i] + (self.GAMMA * self.get_value(self.replay_memory.batch_s_[i]) * mask) - v
            gae = delta + (self.GAMMA * self.GAE_LAMBDA * mask * gae)

            self.replay_memory.batch_gae_return.insert(0, gae+v)
        # self.replay_memory.batch_gae_return.reverse()

    def get_value(self, state):
        """A forward pass through the network to get the value of state
        """
        _, value = self.model.predict_on_batch(state)
        return value


