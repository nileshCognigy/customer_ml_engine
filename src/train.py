import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from src.preprocessor import build_preprocessor

def train():
    print("--> Loading configuration...")
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"--> Reading dataset from {config['data']['raw_file_path']}...")
    df = pd.read_csv(config["data"]["raw_file_path"])
    
    # Drop identifier columns (PII)
    drop_cols = [col for col in config["data"]["id_columns"] if col in df.columns]
    df_features = df.drop(columns=drop_cols)

    target = config["data"]["target_column"]
    X = df_features.drop(columns=[target])
    y = df_features[target]

    print(f"--> Splitting data (Test size: {config['model']['test_size']})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y
    )

    print("--> Building automated pipeline & training model...")
    preprocessor = build_preprocessor(X_train)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=config["model"]["random_state"]
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    threshold = config["model"]["decision_threshold"]
    preds = (probabilities >= threshold).astype(int)

    print("\n" + "=" * 50)
    print(f"MODEL PERFORMANCE (Decision Threshold = {threshold})")
    print("=" * 50)
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probabilities):.3f}")

    # Export
    joblib.dump(pipeline, config["model"]["output_model_path"])
    print("=" * 50)
    print(f"Model saved successfully to: {config['model']['output_model_path']}\n")

if __name__ == "__main__":
    train()