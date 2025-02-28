import cmd
import logging
import json
from pathfinder import RunePathAI
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class RunePathCLI(cmd.Cmd):
    intro = "Welcome to the RunePath AI CLI. Type 'help' for a list of commands."
    prompt = "(RunePath) "

    def __init__(self):
        super().__init__()
        self.ai = RunePathAI()
        self.current_player = None

    def do_scrape_training_methods(self, arg):
        """Scrape and save training methods from Wiki: scrape_training_methods"""
        self.ai.scrape_and_save_training_methods()
        print("Training methods scraped and saved to training_methods.json")


    def do_load_quests(self, arg):
        """Load quest data from RuneMetrics API: load_quests <username>"""
        try:
            self.ai.fetch_quest_data(arg)
            print(f"Loaded quests for player: {arg}")
        except Exception as e:
            print(f"Error loading quest data: {str(e)}")


    def do_load_player(self, arg):
        """Load player data from RuneMetrics API: load_player <username>"""
        try:
            player_data = self.ai.fetch_player_data(arg)
            self.ai.update_player_data(player_data)
            self.current_player = arg  # Set the current player
            print("Player data loaded successfully.")
            logger.info(f"Loaded player data for: {arg}")
        except Exception as e:
            print(f"Error loading player data: {str(e)}")

    def do_initialize_ai(self, arg):
        """Initialize the AI recommender and DDA agent"""
        num_players = 1000  # Example value, adjust as needed
        num_quests = len(self.ai.quest_graph)
        num_features = 50  # Example value, adjust as needed
        self.ai.initialize_recommender(num_players, num_quests, num_features)
        
        state_size = 10  # Example value, adjust as needed
        action_size = 5  # Example value, adjust as needed
        self.ai.initialize_dda_agent(state_size, action_size)
        print("AI components initialized")

    def do_toggle_membership(self, arg):
        """Toggle membership status: toggle_membership"""
        current_status = self.ai.player_data['is_member']
        new_status = not current_status
        self.ai.update_membership(new_status)
        print(f"Membership status changed to: {'Member' if new_status else 'Free-to-play'}")

    def do_pre_train_recommender(self, arg):
        """Pre-train the recommender with Wiki data: pre_train_recommender <epochs> <batch_size>"""
        args = arg.split()
        epochs = int(args[0]) if len(args) > 0 else 100
        batch_size = int(args[1]) if len(args) > 1 else 1024
        self.ai.pre_train_recommender(epochs, batch_size)
        print(f"Recommender pre-trained with {epochs} epochs and batch size {batch_size}")

    def do_train_recommender(self, arg):
        """Train the recommender system: train_recommender <epochs> <batch_size>"""
        if not self.current_player:
            print("Error: No player data loaded. Use 'load_player <username>' first.")
            return
    
        args = arg.split()
        epochs = int(args[0]) if len(args) > 0 else 100
        batch_size = int(args[1]) if len(args) > 1 else 1024
    
        self.ai.train_recommender(self.current_player, epochs, batch_size)
        print(f"Recommender system trained with {epochs} epochs and batch size {batch_size}")

    def do_suggest_quests(self, arg):
        """Suggest quests using the AI recommender"""
        player_id = 0  # Assuming a single player for simplicity
        suggested_quests = self.ai.suggest_quests(player_id)
        print("Suggested quests:")
        for quest in suggested_quests:
            print(f"- {quest}")

    def do_generate_xp_table(self, arg):
        """Create and save the RuneScape XP table to a JSON file"""
        try:
            self.ai.generate_xp_table()
            print("XP table created and saved successfully.")
        except Exception as e:
            print(f"Error creating XP table: {str(e)}")


    
    def do_quit(self, arg):
        """Exit the RunePath AI CLI"""
        print("Thank you for using RunePath AI. Goodbye!")
        return True

if __name__ == '__main__':
    RunePathCLI().cmdloop()
