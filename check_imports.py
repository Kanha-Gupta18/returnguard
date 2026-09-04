import sys
sys.path.insert(0, '.')
from explainability.explainer import explain_transaction, explain_with_descriptions, describe_feature
from explainability.feature_descriptions import FEATURE_DESCRIPTIONS

print('Imports successful')

# Test describe_feature
desc = describe_feature('txn_deviation_from_customer_avg')
print('Description:', desc)

# Test FEATURE_DESCRIPTIONS
print('Has TransactionAmt_raw:', 'TransactionAmt_raw' in FEATURE_DESCRIPTIONS)
print('Has is_high_value:', 'is_high_value' in FEATURE_DESCRIPTIONS)

# Minimal explain_transaction test
tx = {'TransactionAmt_raw': 150.0, 'hour_of_day': 14, 'is_high_value': 0,
      'txn_deviation_from_customer_avg': 200.0, 'V_V12': 0.0}

result = explain_transaction(fraud_prob=0.35, transaction_data=tx)
print('Risk prob:', result['risk_probability'])
print('Direction:', result['direction'])
print('Positive contributors:', len(result['positive_contributors']))
print('Negative contributors:', len(result['negative_contributors']))
print('Feature name:', result['feature_name'])
PYEOF