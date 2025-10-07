import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import random

file_path = 'results.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
df_sorted = df.sort_values(by='Timestamp').reset_index(drop=True)

def majority_vote(group):
    labels = group['Predicted_Label']
    most_common = labels.value_counts()
    max_count = most_common.max()
    candidates = most_common[most_common == max_count].index.tolist()
    return random.choice(candidates)

denoised_labels = []
k = 7
for i in range(0, len(df_sorted), k):
    group = df_sorted.iloc[i:i+k]
    label = majority_vote(group)
    denoised_labels.extend([label] * len(group))
df_sorted['Denoised_Label'] = denoised_labels

labels = sorted(df_sorted['Label'].unique())
f1 = f1_score(df_sorted['Label'], df_sorted['Denoised_Label'], average='macro')
cm = confusion_matrix(df_sorted['Label'], df_sorted['Denoised_Label'])

print("F1 Score (macro):", f1)
print("\nLabel order (row = true label, column = predicted label):")
print(labels)
print("\nConfusion matrix:")
print(cm)

df_sorted.to_csv('TA_results.csv', index=False)