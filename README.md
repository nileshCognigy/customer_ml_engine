# Customer ML Engine — Reusable Classification & Churn Pipeline

A configuration-driven, end-to-end Machine Learning pipeline for tabular customer datasets (predicting churn, conversions, escalations, or fraud). 

The codebase automatically handles missing values, numeric scaling, and one-hot encoding for categorical variables. When onboarding a new customer or dataset, **you only need to update the configuration file—no Python code changes are required.**

---

## Project Structure

```text
customer_ml_engine/
├── .venv/                      # Isolated virtual environment
├── config/
│   └── config.yaml             # Primary configuration (only file you modify)
├── data/
│   ├── raw/                    # Raw input customer CSV files
│   └── predictions.csv         # Generated inference outputs
├── models/
│   └── pipeline.joblib         # Serialized, trained ML model artifact
├── src/
│   ├── __init__.py
│   ├── preprocessor.py         # Dynamic numeric & categorical handling
│   ├── train.py                # Training, evaluation & artifact export
│   └── predict.py              # Batch scoring script
├── requirements.txt            # Project dependencies
└── README.md                   # Setup and usage guide



```markdown
# Customer ML Engine: Step-by-Step Execution Guide

Follow these sequential steps to run the pipeline for any customer dataset.

---

### Step 1: Activate Virtual Environment
Open terminal in the project root directory (`customer_ml_engine`):

* **Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1

```

* **Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat

```

* **macOS / Linux:**

```bash
source .venv/bin/activate

```

---

### Step 2: Add Customer Data

Place the new customer CSV files inside `data/raw/`:

* `data/raw/your_customer_training_data.csv`
* `data/raw/your_new_customers_to_score.csv`

---

### Step 3: Configure Settings

Open `config/config.yaml` and update the parameters:

```yaml
data:
  raw_file_path: "data/raw/your_customer_training_data.csv"
  target_column: "churned"       # The column name to predict (0 or 1)
  id_columns:                    # Columns to exclude from training (PII / IDs)
    - "customer_id"
    - "full_name"
    - "email"

model:
  test_size: 0.20
  random_state: 42
  decision_threshold: 0.40       # Probability threshold to flag high-risk accounts
  output_model_path: "models/pipeline.joblib"

```

---

### Step 4: Train and Evaluate Model

Run the training module:

```bash
python -m src.train

```

* Checks output for Precision, Recall, and ROC-AUC score.
* Saves the trained pipeline to `models/pipeline.joblib`.

---

### Step 5: Score New Customers

Run inference on un-scored customer records:

```bash
python -m src.predict

```

* Predictions with `churn_probability` and `at_risk_flag` are generated at:
`data/predictions.csv`

```

```