# Federated Learning Simulation Framework with Flower

Documentation: <br>

- <b>sim.py</b> <br>
  Main script, taking following arguments:
  - (-m / --model): Model-Choice / Choices: fnn, cnn, tcn, lstm, bilstm, xgb
  - (-H / --hpo): HPO Boolean / Default: False
  - (-k / --kfolds): Kfolds Boolean / Default: False
  - (-e / --eval): Centralized Evaluation Boolean / Default: False
  - (-d / --data): Data Choice / Choices: w, pv
- <b>client_utils.py</b> <br>
  Client-specific classes and functions, such as:
  - class FlowerClient: Client class for neural network models
  - class Xgbclient: Client class for xgb model
  - def get_client_fn: Returns specific client object
- <b>utils.py</b> <br>
  Utilization functions for performing simulation:
  - class CustomFedAvg: Server Strategy for neural nets
  - def run_simulation: Returns history of flower simulation (for neural nets and xgb different strategies)
  - def get_evaluate_fn: Returns centralized evaluation results
  - def get_partitions: Returns partitions (path to dir, list of filenames and model name required). Partitions are split in train and val (train_end and val_start date in config)
  - def get_kfolds_partitions: If kfolds, the partitions of each client are split by n_splits (in config)
  - def weighted_average: Aggregate and returns distributed metrics
  - def loss_per_client_epoch_fn: Returns loss and metrics per client and epoch and round (which is passed to simulation history)
  - def get_hyperparameters: Returns hyperparameters, dependend on model and HPO boolean)
  - def create_or_load_study: Return study (create new one, if no one exists)
  - def load_study: Return study if exists, else None
- <b>preprocessing.py</b> <br>
  Preprocessing steps:
  - def make_windows: Returns X_train, y_train, X_test, y_test in shape (, output_dim, feature_dim)
  - def make_flat_windows: Returns X_train, y_train, X_test, y_test in shape (, output_dim * feature_dim) / XGB needs 2dim data
  - def germansolarfarm/ def europewindfarm: Data-specific adjustments
- <b>models.py</b> <br>
  Exclusively neural network models.
  - def get_model: Returns the required neural network model, with passed hyperparameters
- <b>hpo_analysis.ipynb</b> <br>
  Analyzing HPO performance
- <b>evaluation.ipynb</b> <br>
  Analyzing FL performance for 1 specific FL simulation (for example best hpo trial)
- <b>start_screens.sh</b> <br>
  Starting parallel simulations in screen sessions. Which model and which data is passed in the config variable (may to allow file execution with sudo)
  Screens are shut down automatically, when process is done.
- <b>close_screens.sh</b> <br>
  Closing all active screen sessions.
- <b>train_best.sh</b> <br>
  When HPO performed, train all models simultaneously with best HPO trials.
