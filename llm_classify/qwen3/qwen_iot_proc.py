import pandas as pd
import re
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

# Model path
model_path = "/home/fattysand/workspace/llm_models/Qwen3-32B"
# Define sample size
sample_size = 10

output_excel_path = f"result/iot_qwen_32_{sample_size}_results.xlsx"
os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
print("Model loaded successfully.")

training_data_path = f'../../data/CIC-IOT2022/IOT_sampled_{sample_size}_train.csv'
test_data_path = '../../data/CIC-IOT2022/IOT_test.csv'
prompt_path = f"../../prompt/CIC-IOT2022/IOT_{sample_size}_prompt.txt"

training_data = pd.read_csv(training_data_path)
test_data = pd.read_csv(test_data_path)
with open(prompt_path, "r", encoding="utf-8") as f:
    base_prompt = f.read()
training_data_str = training_data.to_csv(index=False)

print("Training set label distribution:")
print(training_data['Label'].value_counts())

print("Test set label distribution:")
print(test_data['Label'].value_counts())

known_conditions = [
    "ComingHome", "LeavingHome", "Interactions_Audio", "Power_Other", "Interactions_Other",
    "Interactions_Cameras", "Power_Audio", "Power_Cameras", "Idle"
]
test_data['Model_Output'] = ''
test_data['Predicted_Label'] = ''

def predict_traffic_label(row):
    message_content = f"""
Flow IAT Mean: {row['Flow IAT Mean']}
Flow Duration: {row['Flow Duration']}
Fwd Packets/s: {row['Fwd Packets/s']}
Flow Packets/s: {row['Flow Packets/s']}
FWD Init Win Bytes: {row['FWD Init Win Bytes']}
Flow Bytes/s: {row['Flow Bytes/s']}
Idle Mean: {row['Idle Mean']}
Label: ?
"""

    prompt = f"""
{base_prompt}

Training data<training_data>{training_data_str}</training_data>
Prediction sample<prediction_case>{{\n{message_content}\n}}</prediction_case>,
Please return the result in the format ===LABEL===, where LABEL is your predicted traffic type.
"""

    messages = [
        {
            "role": "system",
            "content": "You are a professional network traffic analyst, proficient in application-layer protocol analysis, and skilled in classification based on statistical features. Please make judgments based on the statistical characteristics of the dataset, without using hierarchical strategies."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)


    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.do_sample = False

    try:
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        output_ids = outputs[0][model_inputs.input_ids.shape[1]:].tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"[Error] Model inference failed: {e}")
        output_text = "===Unknown==="

    match = re.search(r'===\s*(ComingHome|LeavingHome|Interactions_Audio|Power_Other|Interactions_Other|Interactions_Cameras|Power_Audio|Power_Cameras|Idle)\s*===', output_text)
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