# nba_pistons_report
**Compiling a report is the primary goal of this project. It is located in the [`REPORT.md`](REPORT.md) file.**
This is a project for organizing data collection, data processing, and machine learning tasks related to NBA player statistics, specifically to determine valuable players among the DETROIT PISTONS.

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

## Usage

To use this project, clone the repository and set up the necessary dependencies.
Create an environment (Ctrl+Shift+P on VSCODE) using the requirements.txt.
You can then run the scripts in the `main_ipynb.ipynb` for easy use or directly in the `src` directory for data collection, processing, and machine learning tasks.

## Contributors

- [Alexander Hernandez](https://github.com/ahernandezjr)

Feel free to contribute to this project by submitting pull requests or opening issues.
