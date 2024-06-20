# nba_pistons_report
**Compiling a report is the primary goal of this project. It is located in the [`REPORT.md`](REPORT.md) file.**
This is a project for organizing data collection, data processing, and machine learning tasks related to NBA player statistics, specifically to determine valuable players among the DETROIT PISTONS.

## Usage

To use this project, clone the repository and set up the necessary dependencies.
Create an environment (Ctrl+Shift+P on VSCODE) using the requirements.txt.
You can then run the scripts in the `main_ipynb.ipynb` for easy use or directly in the `src` directory for data collection, processing, and machine learning tasks.

## Directory Structure

The project directory is organized as follows:

- **data/**: Contains datasets used in the project.
  - `nba_players.csv`: Dataset containing information about NBA players.
  - `nba_player_stats.csv`: Dataset containing NBA player statistics.
- **logs/**: Contains log files generated during the project.

  - `nba_player_stats.log`: Log file for NBA player statistics data processing.

- **src/**: Contains the source code for data collection, data processing, and machine learning tasks.

  - **dataset/**: Contains scripts for processing and cleaning data.
    - `dataset_creation.py`: Module for creating datasets from NBA API using basketball_reference_web_scraper.
    - `dataset_processing.py`: Module for processing datasets to create a useful dataset.
    - `dataset_torch.py`: Module for processing datasets for PyTorch/machine learning evaluation.
    - `filtering.py`: Module for processing datasets further (possibly to be used by `dataset_processing.py`.
  - **machine_learning/**: Contains scripts for machine learning tasks.
    - **models/**: Contains models to be used for the machine learning tasks.
      - `arima.py`: (To Do for better step evaluation)
      - `lstm.py`: LSTM neural networks (custom and PyTorch built-in) for Many-to-Many prediction.
      - `neuralnet.py`: Basic neural net for 1-to-1 prediction
    - `train_models.py`: Module for directly training models in `models/`.
    - `use_models.py`: Module for directly using models in `models/`.

- **utils/**: Contains utility scripts used across the project.

  - `logger.py`: Utility script for logging messages.
  - `config.py`: Utility for settings among files.

- **generate_requirements.bat**: Batch file to generate the requirements.txt file.
- **requirements.txt**: File containing project dependencies.
- **reference**: Any other files related to the project used for referencing.

## Work/To-Do Schedule

| Day       | Task       |
|-----------|------------|
| Monday    | **<input type="checkbox">** (Next) Complete [`arima`](src/machine_learning/models/arima.py) and complete [`lstm`](src/machine_learning/models/lstm.py) |
| Tuesday   | **<input type="checkbox">** None |
| Wednesday | **<input type="checkbox" checked>** Complete [`dataset`](src/dataset/filtering.py) expansion for any 5 year length players |
| Thursday  | **<input type="checkbox">** Complete [`torch_overlap`](src/dataset/torch_overlap.py) to merge custom dataset. **<br>** **<input type="checkbox">** Perform [`analytics`](src/dataset/analytics.py) for tasks and update [`REPORT.md`](REPORT.md) |
| Friday    | **<input type="checkbox">** Create many(4)-to-one and one-to-one [`nn`](src/machine_learning/models/nn.py) |
| Saturday  | **No Progress on Saturdays**       |
| Sunday    | **<input type="checkbox">** Complete [`nn`](src/machine_learning/models/nn.py) and start [`arima`](src/machine_learning/models/arima.py) |


## Contributors

- [Alexander Hernandez](https://github.com/ahernandezjr)

Feel free to contribute to this project by submitting pull requests or opening issues.
