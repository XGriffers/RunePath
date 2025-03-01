from tensorflow import keras
import logging
import requests
import numpy as np
import json
import math
import os
from pymongo import MongoClient
from gridfs import GridFS
import gridfs.errors
from hybridrecommender import HybridRecommender
from DDAAgent import DDAAgent
from dotenv import load_dotenv

Model = keras.models.Model
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunePathAI:

    def test_mongo_connection(self):
        try:
            self.db.command("ping")
            logger.info("MongoDB connection successful")
            return True
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            return False

    SKILL_NAMES = {
            0: "Attack", 1: "Defence", 2: "Strength", 3: "Constitution", 4: "Ranged", 5: "Prayer",
            6: "Magic", 7: "Cooking", 8: "Woodcutting", 9: "Fletching", 10: "Fishing", 11: "Firemaking",
            12: "Crafting", 13: "Smithing", 14: "Mining", 15: "Herblore", 16: "Agility", 17: "Thieving",
            18: "Slayer", 19: "Farming", 20: "Runecrafting", 21: "Hunter", 22: "Construction",
            23: "Summoning", 24: "Dungeoneering", 25: "Divination", 26: "Invention", 27: "Archaeology",
            28: "Necromancy"
        }
    def __init__(self):
        self.mongo_uri = os.getenv("MONGODB_URI")
        if not self.mongo_uri:
            raise ValueError("MONGODB_URI not found in .env file")
        self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=30000, connectTimeoutMS=30000)
        self.db = self.client['runepath']
        self.fs = GridFS(self.db)
        self.quest_graph = {}
        self.player_data = {}
        self.recommender = None
        self.dda_agent = None
        self.num_players = 1000
        self.num_features = 50

        if not self.test_mongo_connection():
            raise ValueError("Failed to connect to MongoDB")

    def _save_model_to_gridfs(self, model, filename):
        """Save a TensorFlow model to MongoDB GridFS."""
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as temp_file:
            model.save(temp_file.name)
            with open(temp_file.name, 'rb') as f:
                self.fs.put(f, filename=filename)
        os.unlink(temp_file.name)  # Clean up temporary file
        logger.info(f"Saved model to MongoDB GridFS: {filename}")

    def _load_model_from_gridfs(self, filename):
        """Load a TensorFlow model from MongoDB GridFS."""
        grid_out = self.fs.get_last_version(filename=filename)
        if not grid_out:
            raise FileNotFoundError(f"Model {filename} not found in GridFS")
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as temp_file:
            temp_file.write(grid_out.read())
            model = keras.models.load_model(temp_file.name)
        os.unlink(temp_file.name)  # Clean up temporary file
        return model
        
        
    
    def generate_xp_table(self):
        """Generate and save the RuneScape XP table to MongoDB."""
        levels_data = {}
        for level in range(1, 128):
            exact = self.exact_xp(level)
            xp_to_next = self.exact_xp(level + 1) - exact if level < 127 else 0
        
            levels_data[str(level)] = {
                "total_xp_exact": exact,
                "xp_to_next_level": xp_to_next
            }
        
        # Store in MongoDB, replacing any existing XP table
        self.db.xp_tables.update_one(
            {"type": "runescape"},
            {"$set": {"table": levels_data}},
            upsert=True
        )
        logger.info("XP table generated and saved to MongoDB")
    def exact_xp(self, level):
        if level <= 1:
            return 0
        total = 0
        for n in range(1, level):
            total += math.floor((n + 300 * (2 ** (n / 7.0))) / 4)
        return total

    def progress_to_next_level(self, current_xp, current_level):
        current_xp_scaled = current_xp / 10.0
        current_level_xp = self.exact_xp(current_level)
        next_level_xp = self.exact_xp(current_level + 1)
        
        if current_xp_scaled < current_level_xp:
            return 0.0
        if current_xp_scaled >= next_level_xp:
            return 100.0
        
        xp_needed = next_level_xp - current_level_xp
        xp_gained = current_xp_scaled - current_level_xp
        percentage = (xp_gained / xp_needed) * 100
        return round(percentage, 2)

    def fetch_player_data(self, username):
        """Fetch player data and quests from RuneMetrics API, store in MongoDB."""
        profile_url = f"https://apps.runescape.com/runemetrics/profile/profile?user={username}"
        quests_url = f"https://apps.runescape.com/runemetrics/quests?user={username}"

        try:
            profile_response = requests.get(profile_url)
            quests_response = requests.get(quests_url)

            if profile_response.status_code != 200 or quests_response.status_code != 200:
                if profile_response.status_code != 200:
                    logger.error(f"Failed to fetch player profile for {username}: {profile_response.status_code}")
                if quests_response.status_code != 200:
                    logger.error(f"Failed to fetch quest data for {username}: {quests_response.status_code}")
                # Fallback to MongoDB or default data if API fails
                user = self.db.users.find_one({"username": username})
                if user:
                    self.player_data = user
                    return user
                else:
                    default_data = {
                        "username": username,
                        "is_member": False,
                        "skills": {s: {"Id": self.SKILL_NAMES.get(s, -1), "Level": 1, "Total_xp": 0, "Rank": -1, "Progress": 0.0} for s in self.SKILL_NAMES.values()},
                        "completed_quests": []
                    }
                    self.db.users.insert_one(default_data)
                    self.player_data = default_data
                    return default_data

            profile_data = profile_response.json()
            quests_data = quests_response.json()

            # Check for incomplete profile (OSRS or untracked)
            if 'error' in profile_data or 'rank' not in profile_data.get('skillvalues', [{}])[0]:
                logger.warning(f"Player {username} not found or profile incomplete in RuneMetrics. Using default profile.")
                player_data = {
                    "username": username,
                    "is_member": False,
                    "skills": {s: {"Id": self.SKILL_NAMES.get(s, -1), "Level": 1, "Total_xp": 0, "Rank": -1, "Progress": 0.0} for s in self.SKILL_NAMES.values()},
                    "completed_quests": []
                }
            else:
                # Better F2P vs. Members check
                members_skills_xp = sum(skill["xp"] / 10 for skill in profile_data.get("skillvalues", []) if skill["id"] > 20 and skill["xp"] > 1000)  # API XP is *10
                members_quests = sum(1 for q in quests_data.get("quests", []) if q["status"] == "COMPLETED" and q.get("members", False))
                is_member = members_skills_xp > 10000 or members_quests > 5  # Adjustable thresholds

                # Build player data
                player_data = {
                    "username": username,  # Use username instead of name for consistency
                    "is_member": is_member,
                    "skills": {},
                    "completed_quests": [q["title"] for q in quests_data.get("quests", []) if q["status"] == "COMPLETED"]
                }

                # Process skills
                for skill in profile_data.get("skillvalues", []):
                    skill_id = skill["id"]
                    skill_name = self.SKILL_NAMES.get(skill_id, f"Unknown Skill {skill_id}")
                    total_xp = float(skill["xp"]) / 10  # Convert API XP to real XP
                    level = skill["level"]
                    rank = skill["rank"] if isinstance(skill["rank"], int) else int(skill["rank"].replace(",", "")) if skill["rank"] != "-1" else -1
                    progress = self.progress_to_next_level(total_xp, level)
                    player_data["skills"][skill_name] = {
                        "Id": skill_id,
                        "Level": level,
                        "Total_xp": total_xp,
                        "Rank": rank,
                        "Progress": progress
                    }

            # Store in MongoDB instead of JSON
            self.db.users.update_one(
                {"username": username},
                {"$set": player_data},
                upsert=True
            )
            self.player_data = player_data
            logger.info(f"Saved player data for {username} to MongoDB")
            return player_data

        except requests.RequestException as e:
            logger.error(f"Failed to fetch player data for {username} from RuneMetrics: {e}")
            # Fallback to MongoDB or default data
            user = self.db.users.find_one({"username": username})
            if user:
                self.player_data = user
                return user
            else:
                default_data = {
                    "username": username,
                    "is_member": False,
                    "skills": {s: {"Id": self.SKILL_NAMES.get(s, -1), "Level": 1, "Total_xp": 0, "Rank": -1, "Progress": 0.0} for s in self.SKILL_NAMES.values()},
                    "completed_quests": []
                }
                self.db.users.insert_one(default_data)
                self.player_data = default_data
                logger.info(f"Saved default player data for {username} to MongoDB")
                return default_data


    def load_quest_data(self, quest_data):
        self.quest_graph = {quest['name']: quest for quest in quest_data}
        logger.info(f"Loaded {len(quest_data)} quests into the graph")

    def fetch_quest_data(self, username):
        """Fetch quest data from RuneMetrics API and store in MongoDB."""
        quests_url = f"https://apps.runescape.com/runemetrics/quests?user={username}"
        response = requests.get(quests_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch quest data for {username} from RuneMetrics API: {response.status_code}")
        
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

        # Store in MongoDB instead of self.quest_graph
        for quest_title, quest_data in quest_graph.items():
            self.db.quests.update_one(
                {"title": quest_title, "username": username},  # Link to username for user-specific eligibility
                {"$set": quest_data},
                upsert=True
            )
        
        # Fetch from MongoDB for this instance (user-specific)
        self.quests = list(self.db.quests.find({"username": username}))
        logger.info(f"Loaded {len(self.quests)} quests from RuneMetrics API and saved to MongoDB for {username}")
        return self.quests


    def update_player_data(self, player_data):
        """Update player data in the instance"""
        username = player_data.get("name", "unknown")
        self.player_data[username] = player_data

    def initialize_recommender(self, num_players, num_quests, num_skills, num_features):
        """Initialize the recommender, loading pre-trained model if available, otherwise create new."""
        try:
            loaded_model = self._load_model_from_gridfs("pretrained_recommender_model.h5")
            self.recommender = HybridRecommender(num_players, num_quests, num_skills, num_features)
            self.recommender.model = loaded_model
            logger.info("Loaded pre-trained recommender model from MongoDB GridFS")
        except (FileNotFoundError, gridfs.errors.NoFile):
            # If model not found, initialize a new recommender
            num_skills = len(self.SKILL_NAMES)  # Ensure this matches your HybridRecommender requirements
            self.recommender = HybridRecommender(num_players, num_quests, num_skills, num_features)
            logger.info("Initialized new recommender model as pre-trained model not found in GridFS")

    def initialize_dda_agent(self, state_size, action_size):
        self.dda_agent = DDAAgent(state_size, action_size)


    def pre_train_recommender(self, epochs=100, batch_size=1024):
        """Pre-train the recommender using MongoDB quest data."""
        if not self.recommender:
            num_skills = len(self.SKILL_NAMES)
            num_quests = self.db.quests.count_documents({})
            self.recommender = HybridRecommender(self.num_players, num_quests, num_skills, self.num_features)

        # Use quests from MongoDB for pre-training
        quests = list(self.db.quests.find({}))
        player_ids = []
        quest_ids = []
        skill_ids = []
        difficulties = []
        rewards = []
        combat_level = []
        ratings = []

        # Dummy data for quests (neutral ratings, using all quests from MongoDB)
        for i, quest in enumerate(quests):
            player_ids.append(0)  # Single player
            quest_ids.append(i)
            skill_ids.append(0)  # No skill focus for quests here
            difficulties.append(quest.get("difficulty", 1))
            rewards.append(quest.get("quest_points", 0))
            combat_level.append(50)  # Average combat level for pre-training
            ratings.append(0.5)  # Neutral rating

        # Use HybridRecommender's train method
        history = self.recommender.train(
            np.array(player_ids), np.array(quest_ids), np.array(skill_ids),
            np.array(difficulties), np.array(rewards), np.array(combat_level),
            np.array(ratings), epochs, batch_size
        )
        self._save_model_to_gridfs(self.recommender.model, "pretrained_recommender_model.h5")
        logger.info("Pre-trained recommender with MongoDB quest data and saved to MongoDB GridFS")
        return history
    

    def train_recommender(self, username, epochs=100, batch_size=256):
        """Train the recommender using player and quest data from MongoDB."""
        if not self.recommender:
            logger.error("Recommender not initialized")
            return None
        
        player_data = self.fetch_player_data(username)  # Use MongoDB to fetch player data
        if not player_data:
            logger.error(f"No player data found for {username}")
            return None
        
        completed_quests = player_data.get("completed_quests", [])
        combat_level = player_data.get("combat_level", 0)
        
        # Fetch quests from MongoDB for this user
        quest_titles = [q["title"] for q in self.db.quests.find({"username": username})]
        player_ids = np.array([0] * len(quest_titles))
        quest_ids = np.array([i for i in range(len(quest_titles))])
        skill_ids = np.array([0] * len(quest_titles))  # No skill focus for quests
        difficulties = np.array([self.db.quests.find_one({"title": q, "username": username}).get("difficulty", 1) for q in quest_titles])
        rewards = np.array([self.db.quests.find_one({"title": q, "username": username}).get("quest_points", 0) for q in quest_titles])
        combat_level_array = np.array([combat_level] * len(quest_titles))
        ratings = np.array([
            1.0 if q in completed_quests else 
            0.5 if self.db.quests.find_one({"title": q, "username": username}).get("user_eligible", False) and \
            self.db.quests.find_one({"title": q, "username": username}).get("status", "NOT_STARTED") != "COMPLETED" else 
            0.0 for q in quest_titles
        ])

            # Clone the pre-trained model to avoid overwriting
        pre_trained_model = self._load_model_from_gridfs("pretrained_recommender_model.h5")
        self.recommender = HybridRecommender(self.num_players, len(quest_titles), len(self.SKILL_NAMES), self.num_features)
        self.recommender.model = keras.models.clone_model(pre_trained_model)  # Clone architecture only
        self.recommender.model.set_weights(pre_trained_model.get_weights())  # Copy weights

        # Compile the cloned model (required after cloning)
        self.recommender.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            loss='mse'
        )

        # Train the cloned model with player-specific data
        history = self.recommender.train(
                player_ids, quest_ids, skill_ids, difficulties, rewards, combat_level_array, ratings, epochs, batch_size
            )

        # Save the player-specific model with a unique name
        self._save_model_to_gridfs(self.recommender.model, f"recommender_model_{username}.h5")
        logger.info(f"Trained recommender for {username} and saved to MongoDB GridFS")
        return history
    
    def get_player_state(self, username):  # Added for DDA feedback
        player = self.player_data.get(username, {})
        skills = player.get("skills", {})
        state = np.array([skills.get(skill, {}).get("Progress", 0.0) for skill in self.SKILL_NAMES.values()] + 
                         [player.get("combat_level", 0)])
        return state.reshape(1, -1)
    
    def train_with_feedback(self, username, action, reward):  # Moved from DDAAgent
        if not self.dda_agent:
            logger.error("DDA agent not initialized")
            return
        state = self.get_player_state(username)
        next_state = self.get_player_state(username)  # Update after action in real app
        self.dda_agent.train(state, action, reward, next_state, False)

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


    def suggest_quests(self, player_id, player_data, num_recommendations=5, progress_threshold=90.0):
        if not self.recommender:
            logger.error("Recommender not initialized")
            return []
        
        player_key = player_data["username"] if isinstance(player_data, dict) else player_data
        if isinstance(player_key, str) and player_key not in self.player_data:  # Only fetch if not already loaded
            player_data = self.fetch_player_data(player_key)
            
        elif not isinstance(player_data, dict):  # Ensure player_data is a dict if passed directly
            player_data = self.fetch_player_data(player_key)
        
        if not player_data or not isinstance(player_data, dict):
            logger.error("No valid player data loaded")
            return ["No player data available"]

        skills = player_data.get("skills", {})
        combat_level = player_data.get("combat_level", 0)
        is_member = player_data.get("is_member", False)
        base_url = "https://runescape.wiki/w/{}_training"

        # Prioritize near-level skills
        near_level_skills = [
            (skill, data["Progress"], data["Level"])
            for skill, data in skills.items()
            if data["Progress"] >= progress_threshold and data["Level"] < (120 if skill in ["Dungeoneering", "Invention", "Archaeology", "Necromancy"] else 99)
        ]
        SKILL_IDS = {name: idx for idx, name in enumerate(skills.keys())}
        
        if near_level_skills:
            near_level_skills.sort(key=lambda x: x[1], reverse=True)

            skill_recommendations = []
            
            for skill, progress, level in near_level_skills:
                wiki_skill = skill if skill != "Constitution" else "Hitpoints"  # Wiki uses "Hitpoints"
                url = base_url.format(wiki_skill)
                try:
                    response = requests.head(url, timeout=5)
                    link = url if response.status_code == 200 else "https://runescape.wiki/w/Skills"
                except requests.RequestException:
                    link = "https://runescape.wiki/w/Skills"  # Fallback on error
                
                player_ids = np.array([player_id]).reshape(1, 1)  # Shape: (1, 1)
                quest_ids = np.array([0]).reshape(1, 1)           # Shape: (1, 1)
                skill_ids = np.array([SKILL_IDS.get(skill, 0)]).reshape(1, 1)  # Shape: (1, 1)
                difficulties = np.array([1]).reshape(1, 1)        # Shape: (1, 1)
                rewards = np.array([1.0]).reshape(1, 1)           # Dummy reward, since we’re not using XP/hour
                combat_level_array = np.array([combat_level]).reshape(1, 1)  # Shape: (1, 1)
                pred = self.recommender.predict(player_ids, quest_ids, skill_ids, difficulties, rewards, combat_level_array)
                method = f"Visit {link} for {skill} training details"
                skill_recommendations.append((f"Train {skill} to level {level + 1} ({progress:.2f}%) - {method}", pred[0][0]))
            
            skill_recommendations.sort(key=lambda x: x[1], reverse=True)
            suggested = [rec[0] for rec in skill_recommendations[:num_recommendations]]
            remaining_slots = num_recommendations - len(suggested)
            if remaining_slots <= 0:
                return suggested
        else:
            suggested = []
            remaining_slots = num_recommendations

        # Quest recommendations
        eligible_quests = [q["title"] for q in self.db.quests.find({"username": player_key, "user_eligible": True, "completed": False}) if q["title"] in list(self.db.quests.find({"username": player_key}))]
        if not eligible_quests and not suggested:
            skill_gaps = self.calculate_skill_gaps(player_key)
            return [f"Train {skill} to level {skills.get(skill, {}).get('Level', 0) + gap}" for skill, gap in skill_gaps.items()][:num_recommendations]

        quest_ids = np.array([list(self.db.quests.find({"username": player_key})).index(q["title"]) for q in self.db.quests.find({"username": player_key, "user_eligible": True, "completed": False}) if q["title"] in eligible_quests]).reshape(-1, 1)  # Shape: (N, 1)
        player_ids = np.array([player_id] * len(eligible_quests)).reshape(-1, 1)  # Shape: (N, 1)
        skill_ids = np.array([0] * len(eligible_quests)).reshape(-1, 1)  # Shape: (N, 1)
        difficulties = np.array([q.get("difficulty", 1) for q in self.db.quests.find({"username": player_key, "user_eligible": True, "completed": False}) if q["title"] in eligible_quests]).reshape(-1, 1)
        rewards = np.array([q.get("quest_points", 0) for q in self.db.quests.find({"username": player_key, "user_eligible": True, "completed": False}) if q["title"] in eligible_quests]).reshape(-1, 1)
        combat_level_array = np.array([combat_level] * len(eligible_quests)).reshape(-1, 1)  # Shape: (N, 1)

        predictions = self.recommender.predict(player_ids, quest_ids, skill_ids, difficulties, rewards, combat_level_array)
        top_quests = sorted(zip(eligible_quests, predictions.flatten()), key=lambda x: x[1], reverse=True)[:remaining_slots]
        for quest, _ in top_quests:
            suggested.append(quest)
            if len(suggested) == num_recommendations:
                break
        
        # Merge Melee skills if present
        melee_skills = {"Attack", "Strength", "Defence"}
        melee_recs = [r for r in suggested if any(ms in r for ms in melee_skills)]
        other_recs = [r for r in suggested if not any(ms in r for ms in melee_skills)]
        if melee_recs:
            highest_melee = max(melee_recs, key=lambda x: float(x.split("(")[1].split("%")[0]))
            melee_skill = next(ms for ms in melee_skills if ms in highest_melee)
            wiki_skill = melee_skill if melee_skill != "Constitution" else "Hitpoints"
            url = base_url.format(wiki_skill)
            try:
                response = requests.head(url, timeout=5)
                link = url if response.status_code == 200 else "https://runescape.wiki/w/Skills"
            except requests.RequestException:
                link = "https://runescape.wiki/w/Skills"
            melee_rec = highest_melee.replace(melee_skill, "Melee").replace(f"for {melee_skill}", f"for Melee").replace(url, link)
            suggested = other_recs + [melee_rec]
        else:
            suggested = other_recs

        return suggested[:num_recommendations]

        
    def calculate_skill_gaps(self, username):  # Ensure this is an instance method
        skill_gaps = {}
        player_data = self.fetch_player_data(username)
        for quest in self.db.quests.find({"username": username}):
            if not quest.get("user_eligible", False):
                for skill, required_level in quest.get("skill_requirements", {}).items():
                    current_level = player_data["skills"].get(skill, {}).get("Level", 0)
                    if current_level < required_level:
                        skill_gaps[skill] = max(skill_gaps.get(skill, 0), required_level - current_level)
        return skill_gaps
    
def __del__(self):
    """Close MongoDB connection on object deletion, safely handle shutdown."""
    try:
        if hasattr(self, 'client') and self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    except Exception as e:
        logger.warning(f"Error closing MongoDB connection during shutdown: {e}")


