import cmd
import logging
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

    def do_train_recommender(self, arg):
        """Train the recommender system: train_recommender <epochs> <batch_size>"""
        args = arg.split()
        epochs = int(args[0]) if len(args) > 0 else 100
        batch_size = int(args[1]) if len(args) > 1 else 1024
    
        player_ids = np.random.randint(0, 1000, 10000)
        quest_ids = np.random.randint(0, len(self.ai.quest_graph), 10000)
        difficulties = np.random.randint(1, 10, 10000)
        rewards = np.random.randint(100, 10000, 10000)
        ratings = np.random.rand(10000)
        self.ai.train_recommender(player_ids, quest_ids, difficulties, rewards, ratings, epochs, batch_size)
        print(f"Recommender system trained with {epochs} epochs and batch size {batch_size}")


    def do_suggest_quests(self, arg):
        """Suggest quests using the AI recommender"""
        player_id = 0  # Assuming a single player for simplicity
        suggested_quests = self.ai.suggest_quests(player_id)
        print("Suggested quests:")
        for quest in suggested_quests:
            print(f"- {quest}")

    
    def do_quit(self, arg):
        """Exit the RunePath AI CLI"""
        print("Thank you for using RunePath AI. Goodbye!")
        return True

if __name__ == '__main__':
    RunePathCLI().cmdloop()
