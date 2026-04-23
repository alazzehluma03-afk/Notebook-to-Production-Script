# Model Comparison CLI
A production-ready command-line tool for evaluating and comparing churn prediction models using automated cross-validation pipelines.
# Installation 
Ensure you have Python 3.8 or higher installed. Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```
# Usage
Run the pipeline directly from your terminal using the following command structure:
```bash
python model_comparison --data-path data/telecom_churn.csv
```
# Arguments
Argument,Type,Default,Description
--data-path,Required,N/A,Path to the input dataset (CSV format)
--output-dir,Optional,./output,Directory where metrics and plots are saved
--n-folds,Optional,5,Number of cross-validation folds
--random-seed,Optional,42,Random seed for reproducibility
--dry-run,Flag,False,Validate data and print configuration without training models
# Examples
1. Dry Run (Configuration Validation)
Use this command to ensure your data path is valid and check class distribution before committing to a full training run.
```bash 
python compare_models.py --data-path data/telecom_churn.csv --dry-run
```
2. Full Pipeline Execution
Run the complete model comparison pipeline and save the results (metrics and plots) to a custom directory with 10-fold cross-validation.
```bash 
python compare_models.py --data-path data/telecom_churn.csv --output-dir ./results --n-folds 10
```