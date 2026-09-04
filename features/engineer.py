import pandas as pd
import numpy as np
import os

# --- Configuration ---
RAW_PROCESSED_PATH = "data/processed/merged.csv"
OUTPUT_PATH = "data/processed/features.csv"

IDENTIFIERS = ["TransactionID", "isFraud", "TransactionDT"]
CARD_COLS = ["card1", "card2", "card3", "card4", "card5", "card6"]
ADDR_COLS = ["addr1", "addr2"]
DIST_COLS = ["dist1", "dist2"]
EMAIL_COLS = ["P_emaildomain", "R_emaildomain"]
V_COLS_PREFIX = "V"

# Leakage decision: is_high_value retained as transaction-amount threshold.
# Not the target label itself, but a deterministic threshold on TransactionAmt.
# Documented for transparency; may be retained with awareness.
RETENTION_IS_HIGH_VALUE = True


def load_data():
    print(f"Loading data from {RAW_PROCESSED_PATH}...")
    df = pd.read_csv(RAW_PROCESSED_PATH)
    print(f"Loaded shape: {df.shape}")
    return df


def engineer_customer_proxy(df):
    print("Engineering customer proxy...")
    df["customer_id"] = (
        df["card1"].astype(str) + "_" + df["card2"].astype(str) + "_" + df["addr1"].astype(str)
    )
    return df


def engineer_transaction_value(df):
    print("Engineering transaction value features...")
    df["TransactionAmt_raw"] = df["TransactionAmt"]
    df["TransactionAmt_log1p"] = np.log1p(df["TransactionAmt"].fillna(0))
    return df


def engineer_card_features(df):
    print("Engineering card features...")
    for col in CARD_COLS:
        df[col] = df[col].fillna("missing").astype("category").cat.codes
    return df


def engineer_address_features(df):
    print("Engineering address features...")
    for col in ADDR_COLS:
        df[col] = df[col].fillna(-1)
    return df


def engineer_distance_features(df):
    print("Engineering distance features...")
    for col in DIST_COLS:
        df[col] = df[col].fillna(-1)
    return df


def engineer_email_features(df):
    print("Engineering email domain features...")
    for col in EMAIL_COLS:
        df[col] = df[col].fillna("missing").astype("category").cat.codes
    return df


def engineer_time_features(df):
    print("Engineering time features...")
    df["hour_of_day"] = (df["TransactionDT"] % 86400) / 3600
    df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
    return df


def engineer_customer_behavioural(df):
    print("Engineering customer behavioural features (leakage-fixed)...")

    # Create customer_id first if not present
    if "customer_id" not in df.columns:
        df["customer_id"] = (
            df["card1"].astype(str) + "_" + df["card2"].astype(str) + "_" + df["addr1"].astype(str)
        )

    # Sort by TransactionDT per customer to ensure temporal order for expanding averages
    df = df.sort_values(["customer_id", "TransactionDT"]).reset_index(drop=True)

    # Expanding mean of TransactionAmt per customer using ONLY historical transactions
    # (transactions before the current one, excluding current row via shift(1))
    expanding_mean = (
        df.groupby("customer_id")["TransactionAmt"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )
    # Cold-start: customers with no history get 0
    expanding_mean = expanding_mean.fillna(0)

    # Expanding count of previous transactions per customer
    expanding_count = (
        df.groupby("customer_id")["TransactionAmt"]
        .transform(lambda x: x.expanding().count().shift(1))
    )
    expanding_count = expanding_count.fillna(0)

    # Deviation from historical customer average (only historical info, no leakage)
    df["customer_avg_amount"] = expanding_mean
    df["customer_txn_count"] = expanding_count
    df["txn_deviation_from_customer_avg"] = df["TransactionAmt"] - df["customer_avg_amount"]

    # CRITICAL: customer_fraud_rate has been REMOVED entirely.
    # It was direct target leakage (isFraud mean per customer).
    # It must NOT be computed or included in any feature set.

    # is_high_value flag: transaction-amount threshold > 500.
    # Not the target label itself, but a threshold on TransactionAmt.
    # Retained with documentation (see RETENTION_IS_HIGH_VALUE constant).
    if RETENTION_IS_HIGH_VALUE:
        df["is_high_value"] = (df["TransactionAmt"] > 500).astype(int)
    else:
        df["is_high_value"] = 0  # placeholder if not retained

    # Drop helper columns that would leak or are not available at prediction time
    df = df.drop(columns=["customer_avg_amount", "customer_txn_count"], errors="ignore")

    return df


def engineer_v_features(df):
    print("Engineering V-features (low missing <30%)...")
    # Identify V columns
    v_cols = [c for c in df.columns if c.startswith(V_COLS_PREFIX)]
    print(f"Found {len(v_cols)} V-columns")

    # Calculate missing rate per V column
    missing_rate = df[v_cols].isnull().mean()
    # Keep columns with less than 30% missing
    valid_v_cols = missing_rate[missing_rate < 0.3].index.tolist()
    print(f"Keeping {len(valid_v_cols)} V-columns (missing <30%)")

    # Cap at top 30
    if len(valid_v_cols) > 30:
        valid_v_cols = valid_v_cols[:30]
        print(f"Capped at top 30 V-columns")

    # Select and fill with -999 sentinel for missing values
    result = df[valid_v_cols].fillna(-999)
    # Rename columns to indicate processed features
    result_cols = [f"V_{c}" for c in valid_v_cols]
    result.columns = result_cols
    return result


def engineer_identity_features(df):
    print("Engineering identity features (id_ columns, <50% missing)...")
    id_cols = [c for c in df.columns if c.startswith("id_")]
    print(f"Found {len(id_cols)} id-columns")

    # Calculate missing rate per id column
    missing_rate = df[id_cols].isnull().mean()
    # Keep columns with less than 50% missing
    valid_id_cols = missing_rate[missing_rate < 0.5].index.tolist()
    print(f"Keeping {len(valid_id_cols)} id-columns (missing <50%)")

    result = df[valid_id_cols].copy()

    # Encode objects as cat.codes, fill numerics with -1
    for col in valid_id_cols:
        if result[col].dtype == object:
            result[col] = result[col].fillna("missing").astype("category").cat.codes
        else:
            result[col] = result[col].fillna(-1)

    return result


def main():
    df = load_data()

    # === Feature engineering pipeline ===
    df = engineer_customer_proxy(df)
    df = engineer_transaction_value(df)
    df = engineer_card_features(df)
    df = engineer_address_features(df)
    df = engineer_distance_features(df)
    df = engineer_email_features(df)
    df = engineer_time_features(df)
    df = engineer_customer_behavioural(df)  # leakage-fixed version

    # V features (independent feature set from competition dataset)
    v_features = engineer_v_features(df)

    # Identity features (id_ columns with missing-rate filtering)
    id_features = engineer_identity_features(df)

    # === Combine all features ===
    # Core keep list: identifiers + engineered features
    # customer_fraud_rate is deliberately excluded (direct target leakage)
    # is_high_value is retained with documentation per RETENTION_IS_HIGH_VALUE
    keep_cols = IDENTIFIERS + [
        "customer_id",
        "TransactionAmt_raw",
        "TransactionAmt_log1p",
        "hour_of_day",
        "day_of_week",
        "is_high_value",
        "txn_deviation_from_customer_avg",
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "dist1", "dist2",
        "P_emaildomain", "R_emaildomain",
    ]

    # Filter keep_cols to those that exist in the dataframe
    existing_keep = [c for c in keep_cols if c in df.columns]
    result = df[existing_keep].copy()

    # Join V features (capped at top 30 with <30% missing, filled -999)
    v_add = engineer_v_features(df)
    if len(v_add) > 0:
        result = pd.concat([result, v_add], axis=1)

    # Join identity features (<50% missing, coded)
    id_add = engineer_identity_features(df)
    if len(id_add) > 0:
        result = pd.concat([result, id_add], axis=1)

    # Select final feature names (exclude identifiers from the feature set)
    feature_cols = [c for c in result.columns if c not in IDENTIFIERS]

    print(f"\nFinal feature columns ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  {c}")

    print(f"\nFinal feature dataframe shape: {result.shape}")
    print(f"Features saved to {OUTPUT_PATH}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved shape: {result.shape}")


if __name__ == "__main__":
    main()