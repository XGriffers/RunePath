import logging
import requests
import numpy as np
import json
import math
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunePathAI:
    def __init__(self):
        self.quest_graph = {}
        self.player_data = {}
        self.recommender = None
        self.dda_agent = None
        

    def fetch_player_data(self, username):
        profile_url = f"https://apps.runescape.com/runemetrics/profile/profile?user={username}"
        quests_url = f"https://apps.runescape.com/runemetrics/quests?user={username}"
       



        profile_response = requests.get(profile_url)
        quests_response = requests.get(quests_url)
       

        if profile_response.status_code != 200 or quests_response.status_code != 200:
            if  profile_response != 200:
                logger.error(f"Failed to fetch player data from RuneMetrics API: {profile_response.status_code}") 
                raise Exception("Failed to fetch player data from RuneMetrics API")
        
        

        profile_data = profile_response.json()
        quests_data = quests_response.json()

        SKILL_NAMES = {
            0: "Attack", 1: "Defence", 2: "Strength", 3: "Constitution", 4: "Ranged", 5: "Prayer",
            6: "Magic", 7: "Cooking", 8: "Woodcutting", 9: "Fletching", 10: "Fishing", 11: "Firemaking",
            12: "Crafting", 13: "Smithing", 14: "Mining", 15: "Herblore", 16: "Agility", 17: "Thieving",
            18: "Slayer", 19: "Farming", 20: "Runecrafting", 21: "Hunter", 22: "Construction",
            23: "Summoning", 24: "Dungeoneering", 25: "Divination", 26: "Invention", 27: "Archaeology",
            28: "Necromancy"
        }  

        player_data = {
        "is_member": any(skill["id"] > 20 for skill in profile_data.get("skillvalues", [])),
        "skills": {},
        "completed_quests": []
            }
        
        for skill in profile_data.get("skillvalues", []):
            skill_id = skill["id"]
            skill_name = SKILL_NAMES.get(skill_id, f"Unknown Skill {skill_id}")
            player_data["skills"][skill_name] = {
                "id": skill_id,
                "level": skill["level"],
                "xp": skill["xp"],
                "rank": skill["rank"],
                "progress": self.calculate_progress_to_level(skill["xp"], skill["level"])
            }
        
        for quest in quests_data.get("quests", []):
            if quest["status"] == "COMPLETED":
                player_data["completed_quests"].append(quest["title"])
        
        
        with open(f"{username}_player_data.json", "w") as f:
            json.dump(player_data, f, indent=4)
        logger.info(f"Saved player data to {username}_player_data.json")

        return player_data
    
    def xp_for_level(self, level):
        return sum(math.floor(l + 300 * 2 ** (l / 7)) for l in range(1, level))

    def xp_to_level(self, xp):
    # RuneScape's official XP table
        xp_table = [0]
        max = 120
        for level in range(1, max):  # RuneScape max level is 120 (or 200M XP)
            xp_table.append(xp_table[-1] + math.floor(level + 300 * 2 ** (level / 7)))

        # Find the level corresponding to the given XP
        for level, xp_threshold in enumerate(xp_table):
            if xp < xp_threshold:
                return level
        return max  # Cap at max level

    def calculate_progress_to_level(self, current_xp, current_level):
        xp_for_current_level = self.xp_for_level(current_level)
        xp_for_next_level = self.xp_for_level(current_level + 1)
    
        if current_xp < xp_for_current_level:
            return 0  # Prevent negative progress
    
        xp_needed = xp_for_next_level - xp_for_current_level
        xp_gained = current_xp - xp_for_current_level
        progress = (xp_gained / xp_needed) * 100
    
        return round(progress, 2)  # Return progress as a percentage (0-100)



    
   
    def calculate_combat_level(skills, player_data):
        calculate_combat_level = 0
        
        attack = skills.get(0, 1)
        defence = skills.get(1, 1)
        strength = skills.get(2, 1)
        hitpoints = skills.get(3, 10)
        ranged = skills.get(4, 1)
        prayer = skills.get(5, 1)
        magic = skills.get(6, 1)
    
        base = 0.25 * (defence + hitpoints + math.floor(prayer/2))
        melee = 0.325 * (attack + strength)
        range_magic = 0.325 * (math.floor(3*ranged/2) + magic)
        player_data["combat_level"] = calculate_combat_level(player_data["skills"])                                     
    
        return math.floor(base + max(melee, range_magic))

    def load_quest_data(self, quest_data):
        self.quest_graph = {quest['name']: quest for quest in quest_data}
        logger.info(f"Loaded {len(quest_data)} quests into the graph")

    def fetch_quest_data(self, username):
        quests_url = f"https://apps.runescape.com/runemetrics/quests?user={username}"
        response = requests.get(quests_url)
        if response.status_code != 200:
            raise Exception("Failed to fetch quest data from RuneMetrics API")
        
        quests_data = response.json()
        quest_graph = {}

        for quest in quests_data.get("quests", []):
            quest_title = quest["title"]
            quest_graph[quest_title] = {
                "status": quest["status"],
                "difficulty": quest["difficulty"],
                "is_members_only": quest["members"],
                "quest_points": quest["questPoints"],
                "user_eligible": quest["userEligible"],
                "completed": quest["status"] == "COMPLETED",
                "subquests": []
            }
        
            # Identify potential parent quests and subquests
            if ":" in quest_title:
                parent_quest, subquest = quest_title.split(":", 1)
                parent_quest = parent_quest.strip()
                subquest = subquest.strip()
                if parent_quest in quest_graph:
                    quest_graph[parent_quest]["subquests"].append(subquest)
                else:
                    quest_graph[parent_quest] = {
                        "status": "NOT_STARTED",
                        "difficulty": quest["difficulty"],
                        "is_members_only": quest["members"],
                        "quest_points": 0,
                        "user_eligible": quest["userEligible"],
                        "completed": False,
                        "subquests": [subquest]
                    }


        quest_graph[quest_title]["skill_requirements"] = quest.get("skillRequirements", {})
        self.quest_graph = quest_graph
        logger.info(f"Loaded {len(quest_graph)} quests from RuneMetrics API")


    def update_player_data(self, player_data):
        self.player_data = player_data
        logger.info(f"Updated player data: {player_data}")

    def initialize_recommender(self, num_players, num_quests, num_features):
        self.recommender = HybridRecommender(num_players, num_quests, num_features)

    def initialize_dda_agent(self, state_size, action_size):
        self.dda_agent = DDAAgent(state_size, action_size)

    def train_recommender(self, player_ids, quest_ids, difficulties, rewards, ratings, epochs=100, batch_size=256):
        if self.recommender:
            history = self.recommender.train(player_ids, quest_ids, difficulties, rewards, ratings, epochs, batch_size)
            return history
        else:
            logger.error("Recommender not initialized")


    def update_membership(self, is_member):
        self.player_data['is_member'] = is_member
        logger.info(f"Updated membership status: {'Member' if is_member else 'Free-to-play'}")

    def can_do_quest(self, quest_name):
        quest = self.quest_graph[quest_name]
        if quest.get('is_members_only', False) and not self.player_data['is_member']:
            return False
        for prereq in quest.get('prerequisites', []):
            if prereq not in self.player_data['completed_quests']:
                return False
        for skill, level in quest.get('skill_requirements', {}).items():
            if self.player_data['skills'].get(skill, 0) < level:
                return False
    
        # Check subquests
        if quest.get('prerequisites'):
            for prereq in quest['prequisites']:
                if prereq not in self.player_data['completed_quests']:
                    return True  # Can do the quest if any subquest is incomplete
            return False  # All subquests are completed
    
        return True


    def suggest_quests(self, player_id, num_recommendations=5):
        if self.recommender:
            eligible_quests = [quest for quest, data in self.quest_graph.items()
                            if data['user_eligible'] and not data['completed'] and data['status'] == 'STARTED' or data['status'] == 'NOT_STARTED']
            if not eligible_quests:
                skill_gaps = self.calculate_skill_gaps()
                return [f"Train {skill} to level {self.player_data['skills'].get(skill, 0) + gap}" for skill, gap in skill_gaps.items()][:num_recommendations]

            quest_ids = [list(self.quest_graph.keys()).index(quest) for quest in eligible_quests]
            difficulties = [self.quest_graph[quest]['difficulty'] for quest in eligible_quests]
            rewards = [self.quest_graph[quest]['quest_points'] for quest in eligible_quests]

            player_ids = np.array([player_id] * len(eligible_quests))
            quest_ids = np.array(quest_ids)
            difficulties = np.array(difficulties)
            rewards = np.array(rewards)

            predictions = self.recommender.predict(player_ids, quest_ids, difficulties, rewards)
            top_quests = sorted(zip(eligible_quests, predictions.flatten()), 
                    key=lambda x: (self.quest_graph[x[0]]['status'] == 'STARTED', x[1]),
                    reverse=True)

            suggested = []
            for quest, _ in top_quests:
                if self.quest_graph[quest].get('subquests'):
                    for subquest in self.quest_graph[quest]['subquests']:
                        full_subquest_name = f"{quest}: {subquest}"
                        if full_subquest_name not in self.player_data['completed_quests']:
                            suggested.append(full_subquest_name)
                            break
                    else:  # If all subquests are completed
                        suggested.append(quest)
                else:
                    suggested.append(quest)
                if len(suggested) == num_recommendations:
                    break
            return suggested
        else:
            logger.error("Recommender not initialized")
            return []


        
    def calculate_skill_gaps(self):
        skill_gaps = {}
        for data in self.quest_graph.values():
            if not data['user_eligible']:
                for skill, required_level in data.get('skill_requirements', {}).items():
                    current_level = self.player_data['skills'].get(skill, 0)
                    if current_level < required_level:
                        skill_gaps[skill] = max(skill_gaps.get(skill, 0), required_level - current_level)
        return skill_gaps



class HybridRecommender:
    def __init__(self, num_players, num_quests, num_features):
        self.num_players = num_players
        self.num_quests = num_quests
        self.num_features = num_features

        player_input = Input(shape=(1,))

        quest_input = Input(shape=(1,))
        quest_difficulty = Input(shape=(1,))
        quest_rewards = Input(shape=(1,))
        combat_level = Input(shape=(1,))
        

        player_embedding = Embedding(num_players, 50)(player_input)
        quest_embedding = Embedding(num_quests, 50)(quest_input)

        player_flatten = Flatten()(player_embedding)
        quest_flatten = Flatten()(quest_embedding)

        concat = Concatenate()([player_flatten, quest_flatten, quest_difficulty, quest_rewards, combat_level])
        dense1 = Dense(100, activation='relu')(concat)
        dense2 = Dense(50, activation='relu')(dense1)
        output = Dense(1)(dense2)

        self.model = Model(inputs=[player_input, quest_input, quest_difficulty, quest_rewards], outputs=output)
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='mse')

    def train(self, player_ids, quest_ids, difficulties, rewards, combat_level, ratings, epochs = 100, batch_size = 1024):
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.00001)
            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            if self.recommender:
                # Calculate or retrieve combat levels for the players
                combat_level = np.array([self.calculate_combat_level(self.player_data['skills']) for _ in player_ids])
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


