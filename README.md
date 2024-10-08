# nba_analytics

**Compiling a report is the primary goal of this project. It is located in the [`REPORT.md`](REPORT.md) file.**
This is a project for organizing data collection, data processing, and machine learning tasks related to NBA player statistics, specifically to determine valuable players among the DETROIT PISTONS.
Another portion of this project is performing the same, local functions through cloud architecture (Azure SQL, Azure Blob Storage, Databricks, Azure Synapse, and PowerBI).

---
### Architecture Comparison of Cloud vs. Local Data Engineering/Analytics:
![architecture of cloud and local variants](data/graphs/NBA_Analytics_Architecture.png)
---

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
        - `arima.py`: (TODO for better step evaluation)
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


# Work Schedule (will be moved to project Github when complete)
<details open>
  <summary>Current Work</summary>

  | Task | Description |
  | ---- | ----------- |
  | Building ML Models in Azure  | Learning to deploy and monitor models in Azure. |

</details>


<details>
  <summary>TODOs </summary>

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
  <summary>Week of 7/29</summary>

  | Day       | Task | Status |
  | --------- | --------- | --------- |
  | Monday    | Completed data format refactoring (seperation of base, filtered, continuous, and continuous_first data formats). | &#x2714; |
  | Tuesday   | Re-worked Databricks code to fix new package setup, refitted SQL, and completed Azure Synapse integration. <br> Restart reports. | &#x2714; |
  | Wednesday | Complete data changes to include team and other excluded columns. Continue with creating report about Piston's players. | &#x2714; |
  | Thursday  | Complete automation in Azure. | &#x2714; |
  | Friday    | Work in Azure. | &#x2714; |
  | Saturday  | **N/A: No Progress on Saturdays.** | --- |
  | Sunday    | Work in Azure. | &#x2714; |

</details>

<details>
  <summary>Week of 7/22</summary>

  | Day       | Task | Status |
  | --------- | --------- | --------- |
  | Monday    | Completed pipeline of bronze_to_silver_to_gold. | &#x2714; |
  | Tuesday   | Rethinking pipeline. | &#x2714; |
  | Wednesday | ~~Start Synapse integration to create gold_db.~~ TODO: Needs refactoring of tables input to SQL database. | &#x2714; |
  | Thursday  | Start refactoring of input data <br> Start refactoring output gold data to integrate dictionaries for 3d data. <br> Start PowerBI integration (using gold output instead of gold_db for now). | &#x2714; |
  | Friday    | Finished refactoring of data. | &#x2714; |
  | Saturday  | **N/A: No Progress on Saturdays.** | --- |
  | Sunday    | Integrate new SQL into pipeline. | &#x2714; |

</details>

<details>
  <summary>Week of 7/15</summary>

  | Day       | Task | Status |
  | --------- | --------- | --------- |
  | Monday    | Continued package integration and SQL setup. | &#x2714; |
  | Tuesday   | Complete package integration. | &#x2714; |
  | Wednesday | Complete SQL -> Data Factory pipeline. | &#x2714; |
  | Thursday  | Begin Databricks/Spark integration with [`bronze_to_silver`](nba_analytics\src\nba_analytics\dataset\bronze_to_silver.py) [`silver_to_gold`](nba_analytics\src\nba_analytics\dataset\silver_to_gold.py). | &#x2714; |
  | Friday    | Continue working on Databricks/Spark data engineering. | &#x2714; |
  | Saturday  | **N/A: No Progress on Saturdays.** | --- |
  | Sunday    | Create a working/or prototype of the bronze_to_silver_to_gold pipeline. | &#x2714; |

</details>

<details>
  <summary>Week of 7/8</summary>

  | Day       | Task | Status |
  | --------- | --------- | --------- |
  | Monday    | Set up settings.cloud. | &#x2714; |
  | Tuesday   | Research ways to implement ETL. | &#x2714; |
  | Wednesday | Reconfigure the project into a module for Azure Functions functionality. | &#x2714; |
  | Thursday  | Modify nba_pistons into 'nba_analytics' package structure. | &#x2714; |
  | Friday    | Continue package modifications. | &#x2714; |
  | Saturday  | **N/A: No Progress on Saturdays.** | --- |
  | Sunday    | Set up Azure, SQL, and data factory. | &#x2714; |

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
