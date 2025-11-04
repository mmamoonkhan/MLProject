import pandas as pd, numpy as np, os, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import yaml
from utils import set_seed

def preprocess(config):
    set_seed(config["seed"])
    df = pd.read_csv("data/train.csv", nrows=config["nrows"])
    df["hour"] = pd.to_datetime(df["hour"], format="%y%m%d%H")
    df["day"] = df["hour"].dt.day
    df["hour"] = df["hour"].dt.hour
    df.rename(columns={"click":"reward"}, inplace=True)

    cat_cols = [c for c in df.columns if c not in ["reward", "id"]]
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    top_ads = df["device_id"].value_counts().nlargest(config["n_actions"]).index
    df = df[df["device_id"].isin(top_ads)].copy()
    df["action"] = LabelEncoder().fit_transform(df["device_id"])

    X_cols = ["hour", "banner_pos", "site_id", "app_id", "device_model", "day"]
    X_train, X_val, y_train, y_val = train_test_split(df[X_cols], df["action"], test_size=0.1, random_state=42)

    print("Training behavior model...")
    model = LogisticRegression(max_iter=100, multi_class="multinomial")
    model.fit(X_train, y_train)

    df["p_log"] = model.predict_proba(df[X_cols])[np.arange(len(df)), df["action"]]
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_parquet("data/processed/train.parquet")
    val_df.to_parquet("data/processed/val.parquet")
    joblib.dump(model, "data/processed/behavior_model.pkl")
    print("✅ Preprocessing complete. Files saved in data/processed/")

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))
    preprocess(config)
