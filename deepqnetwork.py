import tensorflow as tf
import keras
from keras import layers
import random
from collections import deque
import numpy as np


class DQN:
    def __init__(
        self,
        input_shape,
        actions,
        starting_epsilon=1,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        max_experience_size=10000,
        batch_size=32,
        discount_rate=0.95,
    ):
        self.epsilon = starting_epsilon
        self.eps_min = epsilon_min
        self.eps_decay = epsilon_decay
        self.inputshape = input_shape
        self.batchsize = batch_size
        self.gamma = discount_rate
        self.replay_memory = deque(maxlen=max_experience_size)

        self.actions = actions

        self.model = self._build_model()
        
        tf.keras.config.disable_interactive_logging()

    def _build_model(self):
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        model = keras.Sequential(
            [
                keras.Input(shape = self.inputshape),
                layers.Conv2D(
                    16, kernel_size=(8, 8), strides=(4, 4), activation="relu"
                ),
                layers.Conv2D(
                    32, kernel_size=(4, 4), strides=(2, 2), activation="relu"
                ),
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.Dense(len(self.actions), activation="linear"),
            ]
        )

        model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])

        return model

    def choose_action(self, state):
        p = random.random()

        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

        if p < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.actions[np.argmax(self.model.predict(state))]

    def save_experience_replay(self, state, action, reward, state_prime, terminal):
        self.replay_memory.append((state, action, reward, state_prime, terminal))

        self._train_replays()

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)

    def _train_replays(self):
        if len(self.replay_memory) < self.batchsize:
            return
        
        minibatch = random.sample(self.replay_memory, self.batchsize)
                
        states = np.squeeze(np.array([memory[0] for memory in minibatch]))
        state_primes = np.squeeze(np.array([memory[3] for memory in minibatch]))
        
        y = self.model.predict_on_batch(states)
        Q_prime = self.gamma * np.max(self.model.predict_on_batch(state_primes), axis=1)
        
        for i in range(self.batchsize):
            action = minibatch[i][1]
            reward = minibatch[i][2]
            terminal = minibatch[i][4]
            
            y[i, action] = reward + terminal * Q_prime[i]
        
        self.model.train_on_batch(states, y)
