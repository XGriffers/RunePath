import cmd
import logging
from pathfinder import RunePathAI
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger = logging.getLogger("ai")

class RunePathCLI(cmd.Cmd):
    intro = "Welcome to the RunePath AI CLI. Type 'help' for a list of commands."
    prompt = "(RunePath) "

    def __init__(self):
        super().__init__()
        self.ai = RunePathAI()
        self.current_player = None
        # Generate XP table on init, stored in MongoDB
        self.ai.generate_xp_table()

    def do_load_quests(self, arg):
        """Load quest data from RuneMetrics API and MongoDB: load_quests <username>"""
        try:
            username = arg.strip()
            self.ai.fetch_quest_data(username)
            print(f"Loaded quests for player: {username}")
        except Exception as e:
            print(f"Error loading quest data: {str(e)}")

    def do_load_player(self, arg):
        """Load player data from MongoDB/RuneMetrics API: load_player <username>"""
        try:
            username = arg.strip()
            self.current_player = self.ai.fetch_player_data(username)
            print("Player data loaded successfully.")
            logger.info(f"Loaded player data for: {username}")
        except Exception as e:
            print(f"Error loading player data: {str(e)}")

    def do_initialize_ai(self, arg):
        """Initialize AI components: initialize_ai <username>"""
        username = arg.strip()
        if not username:
            print("Error: Username required for initialize_ai. Use 'initialize_ai <username>'.")
            return

        # Load player data if not already loaded
        if not self.current_player:
            try:
                self.current_player = self.ai.fetch_player_data(username)
                print(f"Loaded player data for {username} for initialization.")
            except Exception as e:
                print(f"Error loading player data for initialization: {str(e)}")
                return

        # Ensure quests are loaded
        if not hasattr(self, 'quests') or not self.quests:
            try:
                self.quests = self.ai.fetch_quest_data(username)
                print(f"Loaded quests for {username} for initialization.")
            except Exception as e:
                print(f"Error loading quests for initialization: {str(e)}")
                return

        # Get counts for initialization
        num_players = 1  # Assuming single player for now
        num_quests = len(self.quests) if self.quests else 0
        num_skills = len(self.current_player.get('skills', [])) if self.current_player else 0  # Safely access skills
        num_features = 50  # Or calculate based on your data structure
        
        if num_quests == 0 or num_skills == 0:
            print(f"Error: No quests or skills loaded for {username}. Ensure load_quests and player data include valid data.")
            return

        self.ai.initialize_recommender(num_players, num_quests, num_skills, num_features)
        print(f"AI components initialized for {username}")

    def do_pre_train_recommender(self, arg):
        """Pre-train the recommender with MongoDB data: pre_train_recommender <epochs> <batch_size>"""
        args = arg.split()
        try:
            epochs = int(args[0]) if len(args) > 0 else 100
            batch_size = int(args[1]) if len(args) > 1 else 1024
            self.ai.pre_train_recommender(epochs, batch_size)
            print(f"Recommender pre-trained with {epochs} epochs and batch size {batch_size}")
        except ValueError as e:
            print(f"Error: Invalid arguments. Use 'pre_train_recommender <epochs> <batch_size>' with integer values. Error: {str(e)}")
        except Exception as e:
            print(f"Error pre-training recommender: {str(e)}")

    def do_train_recommender(self, arg):
        """Train the recommender system: train_recommender <epochs> <batch_size>"""
        if not self.current_player:
            print("Error: No player data loaded. Use 'load_player <username>' first.")
            return
    
        args = arg.split()
        epochs = int(args[0]) if len(args) > 0 else 100
        batch_size = int(args[1]) if len(args) > 1 else 1024
    
        try:
            self.ai.train_recommender(self.current_player["username"], epochs, batch_size)
            print(f"Recommender system trained with {epochs} epochs and batch size {batch_size}")
        except Exception as e:
            print(f"Error training recommender: {str(e)}")

    def do_suggest_quests(self, arg):
        """Suggest quests and training methods for the loaded player."""
        if not self.current_player:
            print("Error: No player data loaded. Use 'load_player <username>' first.")
            return
        if not self.ai or not self.ai.recommender:
            print("Error: AI not initialized. Use 'initialize_ai <username>' first.")
            return
        player_id = 0  # Assuming single player for simplicity
        try:
            # Handle if self.current_player is a username (string) or player data (dict)
            player_data = self.current_player if isinstance(self.current_player, dict) else self.ai.fetch_player_data(self.current_player)
            suggested_quests = self.ai.suggest_quests(player_id, player_data)  # Removed username parameter
            print("Suggested quests:")
            for quest in suggested_quests:
                print(f"- {quest}")
        except Exception as e:
            logger.error(f"Error suggesting quests: {e}")
            print(f"Error: {e}")

    def do_generate_xp_table(self, arg):
        """Generate and save the RuneScape XP table to MongoDB"""
        try:
            self.ai.generate_xp_table()
            print("XP table created and saved to MongoDB successfully.")
        except Exception as e:
            print(f"Error creating XP table: {str(e)}")

    def do_quit(self, arg):
        """Exit the RunePath AI CLI"""
        print("Thank you for using RunePath AI. Goodbye!")
        return True

if __name__ == '__main__':
    import sys
    cli = RunePathCLI()
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            cli.onecmd(arg)
    else:
        cli.cmdloop()
