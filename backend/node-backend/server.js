const express = require('express');
const mongoose = require('mongoose');
const { exec } = require('child_process');
require('dotenv').config(); // Load .env for sensitive data

const app = express();
const port = process.env.PORT || 3000;

// MongoDB Atlas URI from .env
const uri = process.env.MONGODB_URI;

if (!uri) {
  console.error('MONGODB_URI is not defined in .env file');
  process.exit(1);
}

// Python paths (configurable via .env or defaults for local testing)
const PYTHON_PATH = process.env.PYTHON_PATH || 'C:\\Users\\nikki\\.conda\\envs\\tf\\python.exe';
const CLI_PATH = process.env.CLI_PATH || 'E:\\RunePath\\ai\\runepath_cli.py';

console.log(`Using Python path: ${PYTHON_PATH}`);
console.log(`Using CLI path: ${CLI_PATH}`);

// Connect to MongoDB Atlas
mongoose.connect(uri, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
  .then(() => console.log('MongoDB Atlas connected'))
  .catch(err => console.error('MongoDB Atlas connection error:', err));

// Define schemas
const userSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  isMember: { type: Boolean, default: false }
});
const User = mongoose.model('User', userSchema);

const questSchema = new mongoose.Schema({
  title: { type: String, required: true, unique: true },
  description: String,
  requirements: String,
  location: String,
  difficulty: Number,
  completed: { type: Boolean, default: false },
  memberOnly: { type: Boolean, default: false }
});
const Quest = mongoose.model('Quest', questSchema);

const progressSchema = new mongoose.Schema({
  timestamp: { type: Date, default: Date.now },
  user: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  quest: { type: mongoose.Schema.Types.ObjectId, ref: 'Quest' },
  skillName: String,
  currentLevel: Number,
  targetLevel: Number,
  progress: Number,
  completed: { type: Boolean, default: false }
});
const Progress = mongoose.model('Progress', progressSchema);

app.use(express.json());

// Endpoint to load player data
app.get('/api/load-player/:username', async (req, res) => {
  const username = req.params.username;
  try {
    let user = await User.findOne({ username });
    if (!user) {
      user = new User({ username, isMember: false });
      await user.save();
      console.log(`Created new user: ${username}`);
    }

    exec(`${PYTHON_PATH} ${CLI_PATH} "load_player ${username}"`, { timeout: 30000 }, (error, stdout, stderr) => {
      if (error) {
        console.error('Python CLI load player error:', error);
        return res.status(500).json({ code: "INTERNAL_SERVER_ERROR", message: error.message || 'Failed to load player' });
      }
      if (stderr) console.error('Python CLI load player stderr:', stderr);
      console.log(`Player data loaded for ${username}:`, stdout);
      res.json({ message: 'Player data loaded successfully', output: stdout });
    });
  } catch (error) {
    console.error('Server error during load player:', error);
    res.status(500).json({ code: "INTERNAL_SERVER_ERROR", message: error.message || 'Server error during load player' });
  }
});

// Endpoint to load quests for player
app.get('/api/load-quests/:username', async (req, res) => {
  const username = req.params.username;
  try {
    const user = await User.findOne({ username });
    if (!user) {
      return res.status(404).json({ code: "NOT_FOUND", message: 'User not found' });
    }

    exec(`${PYTHON_PATH} ${CLI_PATH} "load_quests ${username}"`, { timeout: 30000 }, (error, stdout, stderr) => {
      if (error) {
        console.error('Python CLI load quests error:', error);
        return res.status(500).json({ code: "INTERNAL_SERVER_ERROR", message: error.message || 'Failed to load quests' });
      }
      if (stderr) console.error('Python CLI load quests stderr:', stderr);
      console.log(`Quests loaded for ${username}:`, stdout);
      res.json({ message: 'Quests loaded successfully', output: stdout });
    });
  } catch (error) {
    console.error('Server error during load quests:', error);
    res.status(500).json({ code: "INTERNAL_SERVER_ERROR", message: error.message || 'Server error during load quests' });
  }
});

// Endpoint to initialize AI
app.get('/api/initialize-ai/:username', async (req, res) => {
    const username = req.params.username;
    try {
      const user = await User.findOne({ username });
      if (!user) {
        return res.status(404).json({ code: "NOT_FOUND", message: 'User not found' });
      }
  
      exec(`${PYTHON_PATH} ${CLI_PATH} "initialize_ai ${username}"`, { timeout: 30000 }, (error, stdout, stderr) => {
        if (error) {
          console.error('Python CLI initialize AI error:', error);
          return res.status(500).json({ code: "INTERNAL_SERVER_ERROR", message: error.message || 'Failed to initialize AI' });
        }
        if (stderr) console.error('Python CLI initialize AI stderr:', stderr);
        console.log(`AI initialized for ${username}:`, stdout);
        res.json({ message: 'AI initialized successfully', output: stdout });
      });
    } catch (error) {
      console.error('Server error during initialize AI:', error);
      res.status(500).json({ code: "INTERNAL_SERVER_ERROR", message: error.message || 'Server error during initialize AI' });
    }
  });

// Endpoint to pre-train AI recommender
app.get('/api/pre-train-ai/:username', async (req, res) => {
  const username = req.params.username;
  const epochs = 100; // Default epochs
  const batchSize = 1024; // Default batch size
  try {
    const user = await User.findOne({ username });
    if (!user) {
      return res.status(404).json({ code: "NOT_FOUND", message: 'User not found' });
    }

    exec(`${PYTHON_PATH} ${CLI_PATH} "load_player ${username}" "load_quests ${username}" "initialize_ai" "pre_train_recommender ${epochs} ${batchSize}"`, { timeout: 30000 }, (error, stdout, stderr) => {
      if (error) {
        console.error('Python CLI pre-training error:', error);
        return res.status(500).json({ code: "INTERNAL_SERVER_ERROR", message: error.message || 'AI pre-training failed' });
      }
      if (stderr) console.error('Python CLI pre-training stderr:', stderr);
      console.log(`AI pre-trained for ${username}:`, stdout);
      res.json({ message: 'AI pre-trained successfully', output: stdout });
    });
  } catch (error) {
    console.error('Server error during pre-training:', error);
    res.status(500).json({ code: "INTERNAL_SERVER_ERROR", message: error.message || 'Server error during pre-training' });
  }
});

// Endpoint to train AI recommender
app.get('/api/train-ai/:username', async (req, res) => {
  const username = req.params.username;
  const epochs = 100; // Default epochs
  const batchSize = 1024; // Default batch size
  try {
    const user = await User.findOne({ username });
    if (!user) {
      return res.status(404).json({ code: "NOT_FOUND", message: 'User not found' });
    }

    exec(`${PYTHON_PATH} ${CLI_PATH} "load_player ${username}" "load_quests ${username}" "initialize_ai" "train_recommender ${epochs} ${batchSize}"`, { timeout: 30000 }, (error, stdout, stderr) => {
      if (error) {
        console.error('Python CLI training error:', error);
        return res.status(500).json({ code: "INTERNAL_SERVER_ERROR", message: error.message || 'AI training failed' });
      }
      if (stderr) console.error('Python CLI training stderr:', stderr);
      console.log(`AI trained for ${username}:`, stdout);
      res.json({ message: 'AI trained successfully', output: stdout });
    });
  } catch (error) {
    console.error('Server error during training:', error);
    res.status(500).json({ code: "INTERNAL_SERVER_ERROR", message: error.message || 'Server error during training' });
  }
});

// Recommendation endpoint with user and progress persistence
app.get('/api/recommendations/:username', async (req, res) => {
  const username = req.params.username;
  try {
    let user = await User.findOne({ username });
    if (!user) {
      user = new User({ username, isMember: false });
      await user.save();
      console.log(`Created new user: ${username}`);
    }

    exec(`${PYTHON_PATH} ${CLI_PATH} "load_player ${username}" "load_quests ${username}" "initialize_ai" "suggest_quests ${username}"`, { timeout: 30000 }, (error, stdout, stderr) => {
      if (error) {
        console.error('Python CLI recommendation error:', error);
        return res.status(500).json({ code: "INTERNAL_SERVER_ERROR", message: error.message || 'Recommendations failed' });
      }
      if (stderr) console.error('Python CLI recommendation stderr:', stderr);
      const recommendations = stdout.split('\n')
        .filter(line => line.startsWith('- ') && line.trim())
        .map(line => line.trim().replace('- ', ''));
      console.log(`Recommendations for ${username}:`, recommendations);

      recommendations.forEach(async (questTitle) => {
        let quest = await Quest.findOne({ title: questTitle });
        if (!quest) {
          quest = new Quest({
            title: questTitle,
            description: '', // Add actual data if available from Python CLI
            requirements: '',
            location: '',
            difficulty: 1,
            memberOnly: false
          });
          await quest.save();
        }
        const progress = new Progress({
          user: user._id,
          quest: quest._id,
          completed: false
        });
        await progress.save();
      });

      res.json(recommendations);
    });
  } catch (error) {
    console.error('Server error during recommendations:', error);
    res.status(500).json({ code: "INTERNAL_SERVER_ERROR", message: error.message || 'Server error during recommendations' });
  }
});

// Start the server
app.listen(port, () => console.log(`Server running on port ${port}`));