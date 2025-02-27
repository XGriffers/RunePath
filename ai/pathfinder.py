import logging
import requests
import numpy as np
import json
import math
import os
from runePathHybridRecomend import HybridRecommender
from runePathDDAAgent import DDAAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunePathAI:
    def __init__(self):
        self.quest_graph = {}
        self.player_data = {}  # Add this to store player data
        self.recommender = None
        self.dda_agent = None

    def exact_xp(self, level):
        """Calculate exact XP required for a given RuneScape level"""
        if level <= 1:
            return 0
        total = 0
        for n in range(1, level):
            total += math.floor((n + 300 * (2 ** (n / 7.0))) / 4)
        return total

    def progress_to_next_level(self, current_xp, current_level):
        """Calculate percentage progress to next level using RuneScape XP"""
        current_xp_scaled = current_xp / 10.0
        current_level_xp = self.exact_xp(current_level)
        next_level_xp = self.exact_xp(current_level + 1)
        
        #print(f"Debug: {current_level} - Raw XP: {current_xp}, Scaled XP: {current_xp_scaled}, "
              #f"Current Level XP: {current_level_xp}, Next Level XP: {next_level_xp}")
        
        if current_xp_scaled < current_level_xp:
            return 0.0
        if current_xp_scaled >= next_level_xp:
            return 100.0
        
        xp_needed = next_level_xp - current_level_xp
        xp_gained = current_xp_scaled - current_level_xp
        percentage = (xp_gained / xp_needed) * 100
        return round(percentage, 2)

    def fetch_player_data(self, username):
        profile_url = f"https://apps.runescape.com/runemetrics/profile/profile?user={username}"
        quests_url = f"https://apps.runescape.com/runemetrics/quests?user={username}"

        profile_response = requests.get(profile_url)
        quests_response = requests.get(quests_url)

        if profile_response.status_code != 200 or quests_response.status_code != 200:
            if profile_response.status_code != 200:
                logger.error(f"Failed to fetch player data: {profile_response.status_code}")
            if quests_response.status_code != 200:
                logger.error(f"Failed to fetch quest data: {quests_response.status_code}")
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
            "completed_quests": [],
            "combat_level": int(profile_data.get("combatlevel", "0"))
        }
        
        for skill in profile_data.get("skillvalues", []):
            skill_id = skill["id"]
            skill_name = SKILL_NAMES.get(skill_id, f"Unknown Skill {skill_id}")
            total_xp = skill["xp"]
            level = skill["level"]
            player_data["skills"][skill_name] = {
                "Id": skill_id,
                "Level": level,
                "Total_xp": total_xp,
                "Rank": skill["rank"],
                "Progress": self.progress_to_next_level(total_xp, level)
            }
        
        for quest in quests_data.get("quests", []):
            if quest["status"] == "COMPLETED":
                player_data["completed_quests"].append(quest["title"])
        
        self.player_data[username] = player_data  # Store in instance
        
        
        with open(f"{username}_player_data.json", "w") as f:
            json.dump(player_data, f, indent=4)
            logger.info(f"Saved player data to {username}_player_data.json")
        
        logger.info(f"Updated player data: {player_data}")
        return player_data


    def generate_xp_table(self):
        levels_data = {}
        for level in range(1, 128):
            exact = self.exact_xp(level)
            approx = self.approx_xp(level)
            xp_to_next = self.exact_xp(level + 1) - exact if level < 127 else 0
    
            levels_data[str(level)] = {
                "total_xp_exact": exact,
                "total_xp_approx": approx,
                "xp_to_next_level": xp_to_next,
                "error": abs(exact - approx)
            }
            
            # Write to JSON file
        with open('xp_levels.json', 'w') as f:
            json.dump(levels_data, f, indent=4)
   

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
        self.recommender = HybridRecommender(num_players, num_quests, num_features)

    def initialize_dda_agent(self, state_size, action_size):
        self.dda_agent = DDAAgent(state_size, action_size)

    def train_recommender(self, player_ids, quest_ids, difficulties, rewards, ratings, epochs=100, batch_size=256):
        if self.recommender:
            # Use the current player's combat level; assumes player data is loaded
            combat_level = np.array([self.player_data.get(list(self.player_data.keys())[0], {}).get("combat_level", 0)] * len(player_ids))
            history = self.recommender.train(player_ids, quest_ids, difficulties, rewards, combat_level, ratings, epochs, batch_size)
            return history
        else:
            logger.error("Recommender not initialized")
            return None

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


    def suggest_quests(self, player_id, num_recommendations=5, username=None, progress_threshold=90.0):
        if self.recommender:
            # Get the player's skills (use username or first loaded player)
            player_key = username if username else list(self.player_data.keys())[0]
            skills = self.player_data.get(player_key, {}).get("skills", {})

         # Check for skills close to leveling
            near_level_skills = [
                (skill, data["Progress"])
                for skill, data in skills.items()
                if data["Progress"] >= progress_threshold and data["Level"] < 126  # Cap at max level
            ]
        
            if near_level_skills:
                # Sort by progress (highest first) and take top recommendations
                near_level_skills.sort(key=lambda x: x[1], reverse=True)
                skill_recommendations = [
                    f"Train {skill} to level {skills[skill]['Level'] + 1} ({progress:.2f}%)"
                    for skill, progress in near_level_skills[:num_recommendations]
                ]
                # Pad with quests if fewer than num_recommendations
                remaining_slots = num_recommendations - len(skill_recommendations)
                if remaining_slots <= 0:
                    return skill_recommendations

            else:
                skill_recommendations = []
                remaining_slots = num_recommendations

            # Proceed to quest recommendations if needed
            eligible_quests = [
                quest for quest, data in self.quest_graph.items()
                if data['user_eligible'] and not data['completed'] and (data['status'] == 'STARTED' or data['status'] == 'NOT_STARTED')
            ]
            if not eligible_quests and not skill_recommendations:
                skill_gaps = self.calculate_skill_gaps()
                return [f"Train {skill} to level {skills.get(skill, {}).get('Level', 0) + gap}"
                        for skill, gap in skill_gaps.items()][:num_recommendations]

            quest_ids = [list(self.quest_graph.keys()).index(quest) for quest in eligible_quests]
            difficulties = [self.quest_graph[quest]['difficulty'] for quest in eligible_quests]
            rewards = [self.quest_graph[quest]['quest_points'] for quest in eligible_quests]

            player_ids = np.array([player_id] * len(eligible_quests))
            quest_ids = np.array(quest_ids)
            difficulties = np.array(difficulties)
            rewards = np.array(rewards)
            combat_level = np.array([self.player_data.get(player_key, {}).get("combat_level", 0)] * len(eligible_quests))

            predictions = self.recommender.predict(player_ids, quest_ids, difficulties, rewards, combat_level)
            top_quests = sorted(zip(eligible_quests, predictions.flatten()), 
                                key=lambda x: (self.quest_graph[x[0]]['status'] == 'STARTED', x[1]), 
                                reverse=True)

            suggested = skill_recommendations[:]
            for quest, _ in top_quests[:remaining_slots]:
                if self.quest_graph[quest].get('subquests'):
                    for subquest in self.quest_graph[quest]['subquests']:
                        full_subquest_name = f"{quest}: {subquest}"
                        if full_subquest_name not in self.player_data.get(player_key, {}).get('completed_quests', []):
                            suggested.append(full_subquest_name)
                            break
                    else:
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




