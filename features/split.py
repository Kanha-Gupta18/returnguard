import pandas as pd
import numpy as np
import os

# --- Configuration ---
FEATURES_PATH = "data/processed/features.csv"
OUTPUT_DIR = "data/splits"

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2


def load_data():
    print(f"Loading features from {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded shape: {df.shape}")
    return df


def temporal_split(df):
    print(f"\nSorting by TransactionDT ascending...")
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    total = len(df)
    train_cut = int(total * TRAIN_RATIO)
    val_cut = train_cut + int(total * VAL_RATIO)

    train = df.iloc[:train_cut].copy()
    val = df.iloc[train_cut:val_cut].copy()
    test = df.iloc[val_cut:].copy()

    print(f"Temporal split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)} (rows out of {total})")
    return train, val, test


def check_leakage(train, val, test):
    print("\nChecking for customer leakage between splits...")

    train_custs = set(train["customer_id"].unique())
    val_custs = set(val["customer_id"].unique())
    test_custs = set(test["customer_id"].unique())

    # Check train vs test overlap
    train_test_overlap = train_custs & test_custs
    if train_test_overlap:
        print(f"LEAKAGE: {len(train_test_overlap)} customer(s) in both train and test")
        for c in sorted(train_test_overlap)[:5]:
            print(f"  Example customer_id: {c}")
        return False
    else:
        print("No customers in both train and test OK")

    # Check train vs val overlap
    train_val_overlap = train_custs & val_custs
    if train_val_overlap:
        print(f"LEAKAGE: {len(train_val_overlap)} customer(s) in both train and val")
        for c in sorted(train_val_overlap)[:5]:
            print(f"  Example customer_id: {c}")
        return False
    else:
        print("No customers in both train and val OK")

    return True


def remove_leaking_customers(train, val, test):
    print("\nRemoving leaking customers from val and test...")

    train_custs = set(train["customer_id"].unique())

    # Remove customers from val that appear in train
    val_before = len(val)
    val = val[~val["customer_id"].isin(train_custs)].reset_index(drop=True)
    val_removed = val_before - len(val)
    if val_removed > 0:
        print(f"Removed {val_removed} customer(s) from validation")

    # Remove customers from test that appear in train
    test_before = len(test)
    test = test[~test["customer_id"].isin(train_custs)].reset_index(drop=True)
    test_removed = test_before - len(test)
    if test_removed > 0:
        print(f"Removed {test_removed} customer(s) from test")

    return train, val, test


def generate_report(train, val, test):
    print("\n===== SPLIT REPORT =====")

    def fraud_rate(df):
        return df["isFraud"].mean() * 100

    def customer_count(df):
        return df["customer_id"].nunique()

    def time_range(df):
        return f"Min: {df['TransactionDT'].min()}, Max: {df['TransactionDT'].max()}"

    print(f"Train:  {len(train)} rows, Fraud rate: {fraud_rate(train):.2f}%, "
          f"Customers: {customer_count(train)}, DT range: {time_range(train)}")
    print(f"Val:    {len(val)} rows, Fraud rate: {fraud_rate(val):.2f}%, "
          f"Customers: {customer_count(val)}, DT range: {time_range(val)}")
    print(f"Test:   {len(test)} rows, Fraud rate: {fraud_rate(test):.2f}%, "
          f"Customers: {customer_count(test)}, DT range: {time_range(test)}")

    # Confirmation
    train_custs = set(train["customer_id"].unique())
    test_custs = set(test["customer_id"].unique())
    overlap = train_custs & test_custs
    print(f"\nZero customers overlap between train and test: {'YES' if len(overlap) == 0 else 'NO'} "
          f"(overlap count: {len(overlap)})")

    print("==========================\n")


def save_splits(train, val, test):
    print(f"\nSaving splits to {OUTPUT_DIR}/...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    val_path = os.path.join(OUTPUT_DIR, "validation.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")


def append_to_decisionsmd():
    # Read the current file
    decisions_path = "DECISIONS.md"
    append_text = (
        "\n## Leakage Prevention\n"
        "- Temporal split: 60/20/20 by TransactionDT order\n"
        "- Customer-level separation: customers in train are removed from validation and test\n"
        "- Result: see split report above\n"
        "- No future information used to construct features for any prediction\n"
    )

    # Check if already appended
    if os.path.exists(decisions_path):
        with open(decisions_path, "r") as f:
            content = f.read()
        if "Leakage Prevention" not in content:
            with open(decisions_path, "a") as f:
                f.write(append_text)
            print(f"Appended leakage prevention notes to {decisions_path}")
        else:
            print(f"Leakage Prevention section already present in {decisions_path}")


def main():
    # 1. Load
    df = load_data()

    # 2. Temporal split
    train, val, test = temporal_split(df)

    # 3. Leakage check
    clean = check_leakage(train, val, test)
    if not clean:
        train, val, test = remove_leaking_customers(train, val, test)

    # 4. Re-check after cleanup
    clean = check_leakage(train, val, test)
    assert clean, "LEAKAGE STILL DETECTED after cleanup - cannot proceed"

    print("LEAKAGE CHECK PASSED OK")

    # 5. Remove leaking customers (already done above, but re-assert)
    train, val, test = remove_leaking_customers(train, val, test)

    # 6. Generate report
    generate_report(train, val, test)

    # 7. Save splits
    save_splits(train, val, test)

    # 8. Append to DECISIONS.md
    append_to_decisionsmd()

    print("\nAll tasks complete.")


if __name__ == "__main__":
    main()