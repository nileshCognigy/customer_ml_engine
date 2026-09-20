import numpy as np
import pandas as pd

def generate_datasets():
    np.random.seed(42)
    n = 1000

    # 1. Generate historical training dataset
    first_names = ["Aarav", "Priya", "Rahul", "Sneha", "Vikram", "Ananya", "Rohan", "Neha", "Amit", "Pooja"]
    last_names = ["Sharma", "Patel", "Verma", "Kulkarni", "Deshmukh", "Mehta", "Iyer", "Nair", "Reddy", "Joshi"]

    customer_ids = [f"CUST-{1000 + i}" for i in range(n)]
    full_names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n)]
    emails = [f"user_{1000 + i}@example.com" for i in range(n)]
    plan_tier = np.random.choice(["Basic", "Pro", "Enterprise"], size=n, p=[0.5, 0.35, 0.15])
    payment_method = np.random.choice(["Credit Card", "UPI", "Bank Transfer"], size=n, p=[0.4, 0.45, 0.15])
    tenure_months = np.random.randint(1, 60, size=n)
    monthly_bill = np.random.uniform(25.0, 200.0, size=n)
    support_calls_30d = np.random.poisson(lam=1.5, size=n)
    dropped_call_rate = np.random.uniform(0.01, 0.25, size=n)

    # Churn risk formula
    churn_score = (
        (support_calls_30d * 0.9)
        + (dropped_call_rate * 8.0)
        + (monthly_bill * 0.015)
        - (tenure_months * 0.05)
    )
    churned = (churn_score > np.percentile(churn_score, 65)).astype(int)

    df_train = pd.DataFrame({
        "customer_id": customer_ids,
        "full_name": full_names,
        "email": emails,
        "plan_tier": plan_tier,
        "payment_method": payment_method,
        "tenure_months": tenure_months,
        "monthly_bill": monthly_bill.round(2),
        "support_calls_30d": support_calls_30d,
        "dropped_call_rate": dropped_call_rate.round(4),
        "churned": churned
    })

    df_train.to_csv("data/raw/customer_data.csv", index=False)
    print("Created: data/raw/customer_data.csv (1,000 rows)")

    # 2. Generate small un-scored batch (future callers)
    df_unseen = df_train.drop(columns=["churned"]).head(10).copy()
    df_unseen["customer_id"] = [f"NEW-{2000 + i}" for i in range(10)]
    df_unseen.to_csv("data/raw/new_customers_to_score.csv", index=False)
    print("Created: data/raw/new_customers_to_score.csv (10 rows)")

if __name__ == "__main__":
    generate_datasets()