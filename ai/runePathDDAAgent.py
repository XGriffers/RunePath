import logging
import requests
import numpy as np
import json
import math
import os
from tensorflow import keras


Model = keras.models.Model
Sequential = keras.models.Sequential
Input = keras.layers.Input
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
Dense = keras.layers.Dense
Concatenate = keras.layers.Concatenate
Adam = keras.optimizers.Adam
EarlyStopping = keras.callbacks.EarlyStopping
ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau



class DDAAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001))
        return model
    
    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + 0.99 * np.amax(self.model.predict(next_state))
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def act(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values)

