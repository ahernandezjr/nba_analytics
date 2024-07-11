# nba_analytics

**Compiling a report is the primary goal of this project. It is located in the [`REPORT.md`](REPORT.md) file.**
This is a project for organizing data collection, data processing, and machine learning tasks related to NBA player statistics, specifically to determine valuable players among the DETROIT PISTONS.


## Usage

To use this project, clone the repository and set up the necessary dependencies.
Create an environment (Ctrl+Shift+P on VSCODE) using the requirements.txt.
You can then run the scripts in the `main_ipynb.ipynb` for easy use or directly in the `src` directory for data collection, processing, and machine learning tasks.


## Directory Structure

The project directory is organized as follows:

- **data/**: Contains datasets used in the project.
  - **datasets/**
    - `nba_players.csv`: Dataset containing information about NBA players.
    - `nba_player_stats_5years_overlap.csv`: Dataset containing every 5 consecutive years of NBA player statistics (from `nba_player_stats_5years.csv`).
    - `nba_player_stats_5years_tensor_ready.csv`: PyTorch import version of `nba_player_stats_5years.csv`.
    - `nba_player_stats_5years.csv`: Dataset (csv) containing first 5 years of NBA player statistics.
    - `nba_player_stats_5years.json`: Json version of `nba_player_stats_5years.csv`.
    - `nba_players_advanced.csv`: Dataset containing advanced NBA player statistics.
    - `nba_players_basic.csv`: Dataset containing basic NBA player statistics.
    - `nba_player_stats.csv`: Dataset containing combined NBA player statistics.
  - **graphs**: Contains data analytic graphs from `analytics/`.
  - **models**: Contains machine learning models from `machine_learning/`.
  - **reports**: Location for PowerBI and local pdf created reports from `src/utils/reporting.py`.

- **logs/**: Contains log files generated during the project.
  - `nba_player_stats.log`: Log file for NBA player statistics data processing.

  - **src/**: Contains the source code for data collection, data processing, and machine learning tasks.

    - **dataset/**: Contains scripts for processing and cleaning data.
      - `creation.py`: Module for creating datasets from NBA API using basketball_reference_web_scraper.
      - `processing.py`: Module for processing datasets to create a useful dataset.
      - `torch.py`: Module for processing datasets for PyTorch/machine learning evaluation.
      - `filtering.py`: Module for processing datasets further (possibly to be used by `dataset_processing.py`).
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


# Work Schedule

<details open>
  <summary>Week of 7/8</summary>

  | Day       | Task | Status |
  | --------- | --------- | --------- |
  | Monday    | Set up Azure. | &#x2714; |
  | Tuesday   | Research ways to implement ETL. | &#x2714; |
  | Wednesday | Reconfigure the project into a module for Azure Functions functionality. | &#x2714; |
  | Thursday  | Complete the module setup and begin the function input pipeline. | &#x2716; |
  | Friday    | Set up the data factory for bronze, silver, gold data layers. | &#x2716; |
  | Saturday  | **N/A: No Progress on Saturdays.** | --- |
  | Sunday    | Integrate with Power BI for visualization and reporting. | &#x2716; |


</details>

</details>
  <summary>To Dos </summary>

  | Task | Description |
  | ---- | ----------- |
  | Set up linked services and define ETL pipelines. | Critical for data transformation. |
  | Create Azure Machine Learning Workspace. | Foundation for machine learning projects. |
  | Set up machine learning environment and upload datasets. | Necessary for model training. |
  | Train models using Azure Machine Learning. | Key for predictive analytics. |
  | Deploy models as web services. | For model accessibility. |
  | Integrate with Azure Blob Storage for data storage. | For data persistence. |
  | Update scripts to use Azure Blob Storage SDK. | To leverage Azure storage capabilities. |
  | Automate workflow using Azure Logic Apps or Azure DevOps. | For streamlined operations. |
  | Finish setting up the Data Factory and integrate with Databricks. | For enhanced data processing and analytics, added for Sunday. |
  | Before Azure Machine Learning Tasks. | Refactor/modify dataset [`processing`]() to use numpy savez for saving with dictionary or label row. |

</details>

<details>
  <summary>Week of 7/1</summary>

  | Task | Result | Status |
  | --------- | --------- | --------- |
  | Explore Power BI, Azure, and Fabric | Decided on adapting project into Azure workflow with analytics into Fabric | &#x2714; |

</details>

<details>
  <summary>Week of 6/24</summary>

  | Day       | Task | Status |
  | --------- | --------- | --------- |
  | Monday    | Complete [`lstm`](src/machine_learning/models/lstm.py). <br> Look into [`REPORT.md`](REPORT.md) automation. | &#x2714; |
  | Tuesday   | Complete automation of [`reports`](reports/). | &#x2714; |
  | Wednesday | Look into Databricks implementation. Begin PowerBI testing. | &#x2714; |
  | Thursday  | Modify [`use_models.py`](src/machine_learning/use_models.py) use_model() for model prediction output. | &#x2714; |
  | Friday    | Complete prediction graphs and create average prediction bar graph in [`analytics`](src/dataset/analytics.py). <br> Look into PowerBI use cases over weekend and plan report. | &#x2714; |
  | Saturday  | **N/A: No Progress on Saturdays.** | --- |
  | Sunday    | Begin including Azure/Fabric/PowerBI for data organization, engineering, and reports. | &#x2714; |

</details>

<details>
  <summary>Week of 6/17</summary>

  | Day       | Task | Status |
  | --------- | --------- | --------- |
  | Monday    | Look into ARIMA and complete LSTM. | &#x2714; |
  | Tuesday   | Perform analytics for tasks and update `REPORT.md`. | &#x2714; |
  | Wednesday | Complete dataset expansion for any 5-year length players. | &#x2714; |
  | Thursday  | Complete `torch_overlap` to merge custom dataset. | &#x2714; |
  | Friday    | Create many(4)-to-one and one-to-one neural networks. | &#x2714; |
  | Saturday  | No Progress on Saturdays. <br> Meanwhile: Re-think dataset names for dataset. | --- |
  | Sunday    | Re-check and complete neural networks and start ARIMA preparation in `use_models`. <br> Perform analytics for tasks and update `REPORT.md`. | &#x2714; |

</details>


## Contributors

- [Alexander Hernandez](https://github.com/ahernandezjr)

Feel free to contribute to this project by submitting pull requests or opening issues.
