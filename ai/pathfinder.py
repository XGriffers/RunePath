from tensorflow import keras
import logging
import requests
import numpy as np
import json
import math
import os
from hybridrecommender import HybridRecommender
from DDAAgent import DDAAgent

Model = keras.models.Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunePathAI:
    SKILL_NAMES = {
            0: "Attack", 1: "Defence", 2: "Strength", 3: "Constitution", 4: "Ranged", 5: "Prayer",
            6: "Magic", 7: "Cooking", 8: "Woodcutting", 9: "Fletching", 10: "Fishing", 11: "Firemaking",
            12: "Crafting", 13: "Smithing", 14: "Mining", 15: "Herblore", 16: "Agility", 17: "Thieving",
            18: "Slayer", 19: "Farming", 20: "Runecrafting", 21: "Hunter", 22: "Construction",
            23: "Summoning", 24: "Dungeoneering", 25: "Divination", 26: "Invention", 27: "Archaeology",
            28: "Necromancy"
        }
    def __init__(self):
        self.quest_graph = {}
        self.player_data = {}
        self.recommender = None
        self.dda_agent = None
        self.num_players = 1000
        self.num_features = 50
        
        
    
    def generate_xp_table(self):
        levels_data = {}
        for level in range(1, 128):
            exact = self.exact_xp(level)
            xp_to_next = self.exact_xp(level + 1) - exact if level < 127 else 0
    
            levels_data[str(level)] = {
                "total_xp_exact": exact,
                "xp_to_next_level": xp_to_next
            }
            
            # Write to JSON file
        with open('xp_levels.json', 'w') as f:
            json.dump(levels_data, f, indent=4)

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
        """Fetch player data and quests from RuneMetrics API."""
        profile_url = f"https://apps.runescape.com/runemetrics/profile/profile?user={username}"
        quests_url = f"https://apps.runescape.com/runemetrics/quests?user={username}"
        
        try:
            profile_response = requests.get(profile_url)
            quests_response = requests.get(quests_url)

            if profile_response.status_code != 200 or quests_response.status_code != 200:
                if profile_response.status_code != 200:
                    logger.error(f"Failed to fetch player profile: {profile_response.status_code}")
                if quests_response.status_code != 200:
                    logger.error(f"Failed to fetch quest data: {quests_response.status_code}")
                raise Exception("Failed to fetch player data from RuneMetrics API")

            profile_data = profile_response.json()
            quests_data = quests_response.json()

            # Check for incomplete profile (OSRS or untracked)
            if 'error' in profile_data or 'rank' not in profile_data.get('skillvalues', [{}])[0]:
                logger.warning(f"Player {username} not found or profile incomplete in RuneMetrics. Using default profile.")
                player_data = {
                    "name": username,
                    "combat_level": 3,
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
                    "name": profile_data.get("name", username),
                    "combat_level": int(profile_data.get("combatlevel", 0)),
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
                    rank = int(skill["rank"].replace(",", "")) if skill["rank"] != "-1" else -1
                    progress = self.progress_to_next_level(total_xp, level)
                    player_data["skills"][skill_name] = {
                        "Id": skill_id,
                        "Level": level,
                        "Total_xp": total_xp,
                        "Rank": rank,
                        "Progress": progress
                    }

            # Store and save
            self.player_data[username] = player_data
            with open(f"{username}_player_data.json", "w") as f:
                json.dump(player_data, f, indent=4)
                logger.info(f"Saved player data to {username}_player_data.json")
            
            return player_data

        except requests.RequestException as e:
            logger.error(f"Failed to fetch player data for {username}: {e}")
            # Fallback to default profile on error
            default_data = {
                "name": username,
                "combat_level": 3,
                "is_member": False,
                "skills": {s: {"Id": self.SKILL_NAMES.get(s, -1), "Level": 1, "Total_xp": 0, "Rank": -1, "Progress": 0.0} for s in self.SKILL_NAMES.values()},
                "completed_quests": []
            }
            self.player_data[username] = default_data
            with open(f"{username}_player_data.json", "w") as f:
                json.dump(default_data, f, indent=4)
                logger.info(f"Saved default player data to {username}_player_data.json")
            return default_data


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
        """Update player data in the instance"""
        username = player_data.get("name", "unknown")
        self.player_data[username] = player_data

    def initialize_recommender(self, num_players, num_quests, num_features):
        model_path = "pretrained_recommender_model.h5"
        num_skills = 28
        if os.path.exists(model_path):
            self.recommender = keras.models.load_model(model_path)
            logger.info("Loaded pre-trained recommender model")
        else:
            self.recommender = HybridRecommender(num_players, num_quests, num_skills, num_features)
            logger.info("Initialized new recommender model")

    def initialize_dda_agent(self, state_size, action_size):
        self.dda_agent = DDAAgent(state_size, action_size)


    def pre_train_recommender(self, epochs=100, batch_size=1024):
        if not self.recommender:
            num_skills = len(self.SKILL_NAMES)
            self.recommender = HybridRecommender(self.num_players, len(self.quest_graph), num_skills, self.num_features)

        training_methods = self.scrape_wiki_training_data()
        player_ids = []
        quest_ids = []
        skill_ids = []
        difficulties = []
        rewards = []
        combat_level = []
        ratings = []

        # Dummy data for quests (neutral ratings for now)
        for i, quest in enumerate(self.quest_graph.keys()):
            player_ids.append(0)  # Single player
            quest_ids.append(i)
            skill_ids.append(0)  # No skill focus for quests here
            difficulties.append(self.quest_graph[quest]["difficulty"])
            rewards.append(self.quest_graph[quest]["quest_points"])
            combat_level.append(50)  # Average combat level for pre-training
            ratings.append(0.5)  # Neutral rating

        # Add skill training methods
        SKILL_IDS = {name: idx for idx, name in self.SKILL_NAMES.items()}
        for skill, levels in training_methods.items():
            skill_id = SKILL_IDS.get(skill, 0)
            for level, data in levels.items():
                player_ids.append(0)
                quest_ids.append(0)  # No quest focus
                skill_ids.append(skill_id)
                difficulties.append(1)  # Placeholder; adjust based on method complexity
                rewards.append(data["xp_hour"] / 1000)  # Normalize XP/hour to a reward scale
                combat_level.append(50)  # Average combat level
                ratings.append(1.0)  # Positive rating for known good methods

        history = self.recommender.train(
            np.array(player_ids), np.array(quest_ids), np.array(skill_ids),
            np.array(difficulties), np.array(rewards), np.array(combat_level),
            np.array(ratings), epochs, batch_size
        )
        self.recommender.model.save("pretrained_recommender_model.h5")
        logger.info("Pre-trained recommender with Wiki data")
        return history
    

    def train_recommender(self, username, epochs=100, batch_size=256):
        if not self.recommender:
            logger.error("Recommender not initialized")
            return None
        player_data = self.player_data.get(username, {})
        if not player_data:
            logger.error(f"No player data found for {username}")
            return None
        completed_quests = player_data.get("completed_quests", [])
        combat_level = player_data.get("combat_level", 0)
        quest_titles = list(self.quest_graph.keys())
        player_ids = np.array([0] * len(quest_titles))
        quest_ids = np.array([i for i in range(len(quest_titles))])
        skill_ids = np.array([0] * len(quest_titles))  # No skill focus for quests
        difficulties = np.array([self.quest_graph[q]["difficulty"] for q in quest_titles])
        rewards = np.array([self.quest_graph[q]["quest_points"] for q in quest_titles])
        combat_level = np.array([combat_level] * len(quest_titles))
        ratings = np.array([
            1.0 if q in completed_quests else 
            0.5 if self.quest_graph[q]["user_eligible"] and self.quest_graph[q]["status"] != "COMPLETED" else 
            0.0 for q in quest_titles
        ])
        history = self.recommender.train(player_ids, quest_ids, skill_ids, difficulties, rewards, combat_level, ratings, epochs, batch_size)
        self.recommender.model.save("recommender_model.h5")
        logger.info(f"Trained recommender for {username} and saved model")
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


    def suggest_quests(self, player_id, player_data, num_recommendations=5, username=None, progress_threshold=90.0):
        if not self.recommender:
            logger.error("Recommender not initialized")
            return []
        
        player_key = username if username else list(self.player_data.keys())[0]
        skills = self.player_data.get(player_key, {}).get("skills", {})
        combat_level = self.player_data.get(player_key, {}).get("combat_level", 0)
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
        eligible_quests = [q for q, data in self.quest_graph.items() if data['user_eligible'] and not data['completed'] and (data['status'] == 'STARTED' or data['status'] == 'NOT_STARTED')]
        if not eligible_quests and not suggested:
            skill_gaps = self.calculate_skill_gaps()
            return [f"Train {skill} to level {skills.get(skill, {}).get('Level', 0) + gap}" for skill, gap in skill_gaps.items()][:num_recommendations]

        quest_ids = np.array([list(self.quest_graph.keys()).index(q) for q in eligible_quests]).reshape(-1, 1)  # Shape: (N, 1)
        player_ids = np.array([player_id] * len(eligible_quests)).reshape(-1, 1)  # Shape: (N, 1)
        skill_ids = np.array([0] * len(eligible_quests)).reshape(-1, 1)  # Shape: (N, 1)
        difficulties = np.array([self.quest_graph[q]["difficulty"] for q in eligible_quests]).reshape(-1, 1)  # Shape: (N, 1)
        rewards = np.array([self.quest_graph[q]["quest_points"] for q in eligible_quests]).reshape(-1, 1)  # Shape: (N, 1)
        combat_level_array = np.array([combat_level] * len(eligible_quests)).reshape(-1, 1)  # Shape: (N, 1)

        predictions = self.recommender.predict(player_ids, quest_ids, skill_ids, difficulties, rewards, combat_level_array)
        top_quests = sorted(zip(eligible_quests, predictions.flatten()), key=lambda x: (self.quest_graph[x[0]]['status'] == 'STARTED', x[1]), reverse=True)

        for quest, _ in top_quests[:remaining_slots]:
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

        
    def calculate_skill_gaps(self):
        skill_gaps = {}
        for data in self.quest_graph.values():
            if not data['user_eligible']:
                for skill, required_level in data.get('skill_requirements', {}).items():
                    current_level = self.player_data['skills'].get(skill, 0)
                    if current_level < required_level:
                        skill_gaps[skill] = max(skill_gaps.get(skill, 0), required_level - current_level)
        return skill_gaps




