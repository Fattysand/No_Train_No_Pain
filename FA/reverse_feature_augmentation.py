import pandas as pd

def custom_reverse_bucket_encode(x, max_val, base=10**9, overflow_code=1000000):
    try:
        x = float(x)
    except:
        return overflow_code
    if x == 0:
        return overflow_code
    code = (max_val - x) // base
    return int(code) if code <= overflow_code else overflow_code

input_path = "dataset.csv"
output_path = "FA_reverse_dataset.csv"

df = pd.read_csv(input_path)
# "Idle Mean" can be a column name of any feature
feature_values = df["Idle Mean"].replace(0, pd.NA).dropna().astype(float)
# max_val = feature_values.max()
max_val = 1635931000000000

df["Idle Mean"] = df["Idle Mean"].apply(lambda x: custom_reverse_bucket_encode(x, max_val))
df.to_csv(output_path, index=False)