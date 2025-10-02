Mario Level Generator 🎮

An AI project that evolves Super Mario Bros. levels using genetic algorithms. Built with Python for search and optimization and Unity for simulation and playtesting, this system explores how different level encodings, fitness functions, and evolutionary operators can produce playable, challenging, and creative Mario levels.

✨ Features

Two Level Representations

Grid Encoding – treats the level as a fixed tile grid (X, ?, B, pipes, etc.)

Design Element Encoding (DE) – uses domain knowledge with higher-level constructs (platforms, enemies, coins, pipes).

Customizable Evolutionary Algorithm

Selection: implemented elitist, roulette-wheel, and tournament strategies

Crossover: single-point, uniform, and variable-point methods explored

Mutation: configurable rates for tile or element perturbations

Fitness Functions

Metrics include solvability, meaningful jumps, leniency, linearity, decoration %, and path length

writeup

Extended fitness weighting to balance fun, novelty, and challenge

Parallel Execution

Fitness evaluation runs in parallel with Python’s multiprocessing

Unity Integration

Generates .txt levels that can be imported and played in the Unity Mario simulator

🧠 How It Works

Initialization – Start with random populations of levels (grid tiles or design elements).

Evaluation – Score each level using metrics (e.g., solvability, path %, difficulty).

Selection – Choose parents via elitist, roulette-wheel, or tournament strategies.

Crossover & Mutation – Combine and perturb genomes to create new levels.

Iteration – Repeat across generations until convergence or threshold fitness is reached.

Export & Play – Convert top levels into .txt and load into Unity for playtesting.

📂 Tech Stack

Python – Evolutionary algorithm, metrics, level encoding

Unity – Mario simulator for visualizing and playing generated levels

NumPy & SciPy – Probability distributions, numerical routines

🚀 Getting Started
Prerequisites

Python 3.10+ with required libraries:

pip install numpy scipy


Unity3D (any version that supports the provided Mario simulator)

Running the Algorithm
# Run the GA with Grid encoding
python ga.py

# Switch to Design Element encoding
# (edit ga.py: Individual = Individual_DE)
python ga.py

Play Levels in Unity
python copy_level.py levels/last.txt


Then open Unity → load GameWorldScene → press ▶ to play.

📊 Example Results

Generations: Convergence typically achieved after ~100 generations

Output Levels: Balanced distribution of enemies, power-ups, gaps, and jumps

Evaluation Metrics: Solvability > 90%, varied jump difficulty, increased novelty score across generations

(Insert screenshots/gifs of generated levels here!)

🔮 Future Work

Multi-objective optimization (FI-2POP, novelty search)

writeup

Smarter initialization using real Mario level seeds

Adaptive difficulty based on player performance

👩‍💻 Author

Created by Mary Tarevern – CS @ UC Santa Cruz.

GitHub

LinkedIn
