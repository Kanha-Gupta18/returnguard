import pandas as pd

train = pd.read_csv('data/splits/train.csv')
val = pd.read_csv('data/splits/validation.csv')
test = pd.read_csv('data/splits/test.csv')

# Check what customer-related columns exist
print('Columns with customer:')
for c in train.columns:
    if 'customer' in c.lower():
        print(f'  {c}: {train[c].dtype}, unique={train[c].nunique()}')

# Check customer_id specifically
if 'customer_id' in train.columns:
    print('\ncustomer_id:')
    print(f'  train unique: {train["customer_id"].nunique()}')
    print(f'  val unique: {val["customer_id"].nunique()}')
    print(f'  test unique: {test["customer_id"].nunique()}')
    # Check overlap
    t_train = set(train['customer_id'].unique())
    t_val = set(val['customer_id'].unique())
    t_test = set(test['customer_id'].unique())
    print(f'  Train intersect Val: {len(t_train & t_val)}')
    print(f'  Train intersect Test: {len(t_train & t_test)}')
    print(f'  Val intersect Test: {len(t_val & t_test)}')
else:
    print('customer_id NOT in columns')