import pandas as pd

df = pd.read_csv('all_data.csv')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 80-20 train-val split 
split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx]
val_df = df.iloc[split_idx:]

train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)

print(f"Train set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
