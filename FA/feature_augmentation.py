import pandas as pd

def custom_forward_bucket_encode(x, base=10**9, overflow_code=1000000):
    try:
        x = float(x)
    except:
        return overflow_code
    code = x // base
    return int(code) if code <= overflow_code else overflow_code

input_path = "dataset.csv"
output_path = "FA_dataset.csv"
column_to_encode = "Idle Mean"  # "Idle Mean" can be a column name of any feature

df = pd.read_csv(input_path)
# feature_values = df[column_to_encode].replace(0, pd.NA).dropna().astype(float)
df[column_to_encode] = df[column_to_encode].apply(lambda x: custom_forward_bucket_encode(x))

df.to_csv(output_path, index=False)

print(f"FA complete. Column '{column_to_encode}' has been replaced and saved to: {output_path}")