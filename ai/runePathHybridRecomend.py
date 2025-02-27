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



class HybridRecommender:
    def __init__(self, num_players, num_quests, num_features):
        self.num_players = num_players
        self.num_quests = num_quests
        self.num_features = num_features

        player_input = Input(shape=(1,))
        quest_input = Input(shape=(1,))
        quest_difficulty = Input(shape=(1,))
        quest_rewards = Input(shape=(1,))
        combat_level_input = Input(shape=(1,))

        player_embedding = Embedding(num_players, 50)(player_input)
        quest_embedding = Embedding(num_quests, 50)(quest_input)

        player_flatten = Flatten()(player_embedding)
        quest_flatten = Flatten()(quest_embedding)

        concat = Concatenate()([player_flatten, quest_flatten, quest_difficulty, quest_rewards, combat_level_input])
        dense1 = Dense(100, activation='relu')(concat)
        dense2 = Dense(50, activation='relu')(dense1)
        output = Dense(1)(dense2)

        self.model = Model(inputs=[player_input, quest_input, quest_difficulty, quest_rewards, combat_level_input], outputs=output)
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='mse')

    def train(self, player_ids, quest_ids, difficulties, rewards, combat_level, ratings, epochs=100, batch_size=1024):
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.00001)
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        history = self.model.fit(
            [player_ids, quest_ids, difficulties, rewards, combat_level],
            ratings,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[reduce_lr, early_stopping],
            validation_split=0.2
        )
        return history

    def predict(self, player_ids, quest_ids, difficulties, rewards, combat_level):
        return self.model.predict([player_ids, quest_ids, difficulties, rewards, combat_level])