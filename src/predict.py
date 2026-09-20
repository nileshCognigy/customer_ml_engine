import yaml
import joblib
import pandas as pd

def predict_batch(input_csv_path, output_csv_path="data/predictions.csv"):
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"--> Loading trained model from {config['model']['output_model_path']}...")
    pipeline = joblib.load(config["model"]["output_model_path"])

    print(f"--> Ingesting records to score from {input_csv_path}...")
    raw_df = pd.read_csv(input_csv_path)

    # Exclude ID columns from feature matrix while keeping them for final report
    drop_cols = [col for col in config["data"]["id_columns"] if col in raw_df.columns]
    feature_df = raw_df.drop(columns=drop_cols, errors="ignore")
    if config["data"]["target_column"] in feature_df.columns:
        feature_df = feature_df.drop(columns=[config["data"]["target_column"]])

    # Generate probabilities
    probs = pipeline.predict_proba(feature_df)[:, 1]
    threshold = config["model"]["decision_threshold"]
    flags = (probs >= threshold).astype(int)

    # Create output dataframe
    results = raw_df[drop_cols].copy() if drop_cols else pd.DataFrame(index=raw_df.index)
    results["churn_probability"] = probs.round(4)
    results["at_risk_flag"] = flags

    results.to_csv(output_csv_path, index=False)
    print(f"--> Done! Predictions saved to: {output_csv_path}")
    print("\nFirst 5 predictions preview:")
    print(results.head())

if __name__ == "__main__":
    predict_batch("data/raw/new_customers_to_score.csv")