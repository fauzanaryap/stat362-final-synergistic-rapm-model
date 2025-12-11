# stat362-final-synergistic-rapm-model

**INSTRUCTIONS**

In order to replicate my model, you will need to go through the Jupyter Notebooks in the \src folder in this order:

0. Run the pyproject.toml to install all libraries
1. stint_processing.ipynb
 * This notebook walks through gathering all the play-by-play data across the last 10 years of the NBA
 * Run each cell sequentially and wait patiently for it to pull the data and process the play-by-play dfs into stint-level dataframes
 * You need to run this step! Or else there will be no target variable.
2. stat_collection.ipynb
 * This notebook will gather all of the X features from nba_api
 * Run each cell sequentially and wait patiently for it to pull the data
 * _This step is not necessary if looking at the last 10 years of NBA data, as it has already been uploaded for your convenience_
3. data_cleaning.ipynb
 * This notebook cleans the stint data to actionable S-RAPM
 * Also cleans X features
 * Follow the instructions in the notebook and run through it sequentially
4. model.ipynb
 * Where all the magic happens!
 * Run through the notebook sequentially and follow instructions

***

Fall Quarter 2025, STAT 362, Fauzan Aryaputra

**Project Description**:

This project models pairwise offensive player synergy in the NBA, extending my honors thesis on Synergistic RAPM (S-RAPM). The goal is to predict the counterfactual synergy between two players using only their individual statistics, i.e., estimate how well two players would perform together even if they have never shared the court.

This project tries to answer a simple but difficult question:

__*Can we use a player’s individual statistics to predict how well two players would perform together, even if they have never played together before?*__

**The dataset consists of**:

* Player-season box score and advanced stats (datasets/X_pair.csv)
* Pairwise offensive S-RAPM labels computed from lineup data and possessions played per pair (datasets/Y_pair.csv)

The final training data contains thousands of player–player pairs per season, each represented by two sets of player features and a synergy target.

The final model is a symmetry-preserving Deep Sets–inspired network consisting of:

**Shared Player Encoder**
  * Produces a 32-dim continuous embedding and an 8-dim soft archetype distribution for each player
  * Captures both fine-grained style and coarse role structure

**Symmetric Interaction Block**
  * Combines the two players with order-invariant operations:
    * φ(e_A) + φ(e_B)
    * |φ(e_A) − φ(e_B)|
    * φ(e_A) ⊙ φ(e_B)
    * a_A ⊙ a_B, |a_A − a_B|
  * Explicitly models similarity, complementarity, redundancy, and role interaction

**Pair-Level Synergy Head**
  * A small MLP predicting offensive S-RAPM
  * Trained with minutes-weighted MSE loss to handle label noise

The entire architecture enforces Synergy(A, B) = Synergy(B, A) and is evaluated using Leave-One-Player-Out (LOPO) cross-validation to ensure generalization to unseen players.

Despite my best efforts, I was not able to make a model that is able to explain most of the variation in the data. The best test R^2 achieved was 0.078. Training plots and other visualizations of model performance can be found in the images folder.
