import pandas as pd
import re
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

# Model path
model_path = "/path/to/Llama-3.1-8B-Instruct"
# Define sample size
sample_size = 10

output_excel_path = f"result/tfc_llama_{sample_size}_results.xlsx"
os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded successfully.")

training_data_path = f'../../data/USTC-TFC/TFC_sampled_{sample_size}_train.csv'
test_data_path = '../../data/USTC-TFC/TFC_test.csv'
prompt_path = f"../../prompt/USTC-TFC/TFC_{sample_size}_prompt.txt"

training_data = pd.read_csv(training_data_path)
test_data = pd.read_csv(test_data_path)
with open(prompt_path, "r", encoding="utf-8") as f:
    base_prompt = f.read()
training_data_str = training_data.to_csv(index=False)

print("Training set label distribution:")
print(training_data['Label'].value_counts())

print("Test set label distribution:")
print(test_data['Label'].value_counts())

known_conditions = ["BitTorrent", "FTP", "Gmail", "MySQL", "Outlook", "SMB", "Skype", "Weibo", "WorldOfWarcraft"]
test_data['Model_Output'] = ''
test_data['Predicted_Label'] = ''

def predict_traffic_label(row):
    message_content = f"""
    Flow IAT Mean: {row['Flow IAT Mean']}
    FWD Init Win Bytes: {row['FWD Init Win Bytes']}
    Bwd Init Win Bytes: {row['Bwd Init Win Bytes']}
    Fwd Packet Length Max: {row['Fwd Packet Length Max']}
    Average Packet Size: {row['Average Packet Size']}
    Packet Length Std: {row['Packet Length Std']}
    Total Length of Fwd Packet: {row['Total Length of Fwd Packet']}
    Packet Length Mean: {row['Packet Length Mean']}
    Idle Mean: {row['Idle Mean']}
    Label: ?
    """

    prompt = f"""
{base_prompt}

Training data<training_data>{training_data_str}</training_data>
Prediction sample<prediction_case>{{\n{message_content}\n}}</prediction_case>,
Please return the result in the format ===LABEL===, where LABEL is your predicted traffic type.
"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"[Error] Model inference failed: {e}")
        output_text = "===Unknown==="

    match = re.search(r'===\s*(BitTorrent|FTP|Gmail|MySQL|Outlook|SMB|Skype|Weibo|WorldOfWarcraft)\s*===', output_text)
    predicted_label = match.group(1) if match else 'Unknown'
    enhanced_output = f"Result: {output_text}"
    return predicted_label, enhanced_output

predictions = []
actual_labels = []

print(f"\nStarting prediction on {len(test_data)} samples...\n")
for index, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing cases"):
    predicted_label, enhanced_output = predict_traffic_label(row)
    actual_label = row['Label']

    test_data.at[index, 'Predicted_Label'] = predicted_label
    test_data.at[index, 'Model_Output'] = enhanced_output

    predictions.append(predicted_label)
    actual_labels.append(actual_label)

conf_matrix = confusion_matrix(actual_labels, predictions, labels=known_conditions)
accuracy = conf_matrix.diagonal().sum() / conf_matrix.sum()
precision, recall, f1_score, _ = precision_recall_fscore_support(
    actual_labels, predictions, labels=known_conditions, average='macro'
)

print("\nEvaluation Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1_score:.4f}")

with pd.ExcelWriter(output_excel_path) as writer:
    test_data.to_excel(writer, index=False, sheet_name='Predictions')
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=known_conditions, index=known_conditions)
    conf_matrix_df.to_excel(writer, sheet_name='Confusion_Matrix')
    summary_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1_Score'],
        'Value': [accuracy, precision, recall, f1_score]
    })
    summary_df.to_excel(writer, index=False, sheet_name='Metrics')

print(f"\nAll results have been saved to {output_excel_path}")