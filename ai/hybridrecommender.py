from tensorflow import keras
import logging

Model = keras.models.Model
Input = keras.layers.Input
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
Dense = keras.layers.Dense
Concatenate = keras.layers.Concatenate
Adam = keras.optimizers.Adam
EarlyStopping = keras.callbacks.EarlyStopping
ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRecommender:
    def __init__(self, num_players, num_quests, num_skills, num_features):
        self.num_players = num_players
        self.num_quests = num_quests
        self.num_skills = num_skills
        self.num_features = num_features

        try:
            self.model = self._build_model()
            logger.info("Successfully built recommender model")
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            self.model = None
            raise
    
    def _build_model(self):

        player_input = Input(shape=(1,))
        quest_input = Input(shape=(1,))
        skill_input = Input(shape=(1,))
        quest_difficulty = Input(shape=(1,))
        quest_rewards = Input(shape=(1,))
        combat_level_input = Input(shape=(1,))

        player_embedding = Embedding(self.num_players, 50)(player_input)
        quest_embedding = Embedding(self.num_quests, 50)(quest_input)
        skill_embedding = Embedding(self.num_skills, 50)(skill_input)

        player_flatten = Flatten()(player_embedding)
        quest_flatten = Flatten()(quest_embedding)
        skill_flatten = Flatten()(skill_embedding)

        concat = Concatenate()([player_flatten, quest_flatten, skill_flatten, quest_difficulty, quest_rewards, combat_level_input])
        dense1 = Dense(100, activation='relu')(concat)
        dense2 = Dense(50, activation='relu')(dense1)
        output = Dense(1)(dense2)

        model_instance = Model(inputs=[player_input, quest_input, skill_input, quest_difficulty, quest_rewards, combat_level_input], outputs=output)
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        model_instance.compile(optimizer=optimizer, loss='mse')
        return model_instance


    def train(self, player_ids, quest_ids, skill_ids, difficulties, rewards, combat_level, ratings, epochs=100, batch_size=1024):
        if self.model is None:
            logger.error("Model not Init; Cannot train. Have you run initialize_ai first?")
            raise ValueError("Model Not Init")
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.00001)
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        history = self.model.fit(
            [player_ids, quest_ids, skill_ids, difficulties, rewards, combat_level],
            ratings,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[reduce_lr, early_stopping],
            validation_split=0.2
        )
        return history

    def predict(self, player_ids, quest_ids, skill_ids, difficulties, rewards, combat_level):
        if self.model is None:
            logger.error("Model not initialized; cannot predict")
            raise ValueError("Model not initialized")
        return self.model.predict([player_ids, quest_ids, skill_ids, difficulties, rewards, combat_level], verbose = 0)