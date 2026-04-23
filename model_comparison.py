import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_data(data_path="telecom_churn.csv"):
    """Loads and validates data existence."""
    path = Path(data_path)
    if not path.exists():
        logger.error(f"File not found: {data_path}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(path)
        logger.info(f"Successfully loaded data from {data_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        sys.exit(1)

def validate_data(df):
    """Validates required columns and basic data integrity."""
    required = ['churned', 'tenure', 'monthly_charges']
    for col in required:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            sys.exit(1)
    logger.info("Data validation passed.")
    logger.info(f"Class distribution:\n{df['churned'].value_counts(normalize=True)}")

def build_pipeline():
    """Defines the preprocessing and model architecture."""
    numeric_features = ['tenure', 'monthly_charges', 'total_charges', 'num_support_calls']
    categorical_features = ['gender', 'contract_type', 'internet_service', 'payment_method']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

def main():
    parser = argparse.ArgumentParser(description="Model Comparison Pipeline CLI")
    parser.add_argument("--data-path", required=True, help="Path to the input CSV")
    parser.add_argument("--output-dir", default="./output", help="Directory for results")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print config without training")
    
    args = parser.parse_args()

    # 1. Load and validate data
    df = load_data(args.data_path)
    validate_data(df)
    
    logger.info(f"Pipeline Config: Output={args.output_dir}, Folds={args.n_folds}")

    if args.dry_run:
        logger.info("Dry run complete. No models trained.")
        sys.exit(0)

    # 2. Setup output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 3. Model Training
    logger.info("Starting training...")
    X = df.drop(columns=['churned', 'customer_id'])
    y = df['churned']
    
    model = build_pipeline()
    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_seed)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    # 4. Save results
    results_df = pd.DataFrame({'fold': range(1, args.n_folds + 1), 'accuracy': scores})
    results_df.to_csv(output_path / "results.csv", index=False)
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, args.n_folds + 1), scores)
    plt.title("Cross-Validation Accuracy")
    plt.savefig(output_path / "cv_plot.png")
    
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()