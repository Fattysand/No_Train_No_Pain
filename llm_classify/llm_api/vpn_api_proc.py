import pandas as pd
import re
import os
from openai import OpenAI
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

client = OpenAI(base_url='your_url', api_key='your_api_key')
# Define sample size
sample_size = 10

output_excel_path = f"result/vpn_llama_{sample_size}_results.xlsx"
os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)

training_data_path = f'../../data/ISCXVPN2016/VPN_sampled_{sample_size}_train.csv'
test_data_path = '../../data/ISCXVPN2016/VPN_test.csv'
prompt_path = f"../../prompt/ISCXVPN2016/VPN_{sample_size}_prompt.txt"

training_data = pd.read_csv(training_data_path)
test_data = pd.read_csv(test_data_path)
with open(prompt_path, "r", encoding="utf-8") as f:
    base_prompt = f.read()
training_data_str = training_data.to_csv(index=False)

print("Training set label distribution:")
print(training_data['Label'].value_counts())

print("Test set label distribution:")
print(test_data['Label'].value_counts())

known_conditions = ["BROWSING", "CHAT", "MAIL", "FTP", "P2P", "Streaming", "VoIP"]
test_data['Model_Output'] = ''
test_data['Predicted_Label'] = ''

def predict_traffic_label(row):
    # 构建消息内容
    message_content = f"""
    Flow IAT Max: {row['Flow IAT Max']}
    Flow Duration: {row['Flow Duration']}
    Fwd Packets/s: {row['Fwd Packets/s']}
    Flow Packets/s: {row['Flow Packets/s']}
    Bwd Packets/s: {row['Bwd Packets/s']}
    Packet Length Mean: {row['Packet Length Mean']}
    Fwd Packet Length Mean: {row['Fwd Packet Length Mean']}
    Average Packet Size: {row['Average Packet Size']}
    Flow Bytes/s: {row['Flow Bytes/s']}
    Packet Length Variance: {row['Packet Length Variance']}
    Bwd Segment Size Avg: {row['Bwd Segment Size Avg']}
    Idle Mean: {row['Idle Mean']}
    Label: ?
    """

    try:
        response = client.chat.completions.create(
            model="model_name",
            messages=[
                {
                    "role": "system",
                    "content": base_prompt
                },
                {
                    "role": "user",
                    "content": f"<training_data>{training_data_str}</training_data>"
                },
                {
                    "role": "user",
                    "content": f"<prediction_case>{{\n{message_content}\n}}</prediction_case>"
                }
            ],
            temperature=0,
            max_tokens=64,
        )

        if (response and 
            hasattr(response, "choices") and 
            len(response.choices) > 0 and 
            hasattr(response.choices[0], "message") and 
            response.choices[0].message and 
            hasattr(response.choices[0].message, "content") and 
            response.choices[0].message.content is not None):
            output_text = response.choices[0].message.content.strip()
        else:
            raise ValueError(f"Invalid LLM response structure: {response}")
    
    except Exception as e:
        print(f"[Error] during LLM call or parsing: {e}")
        output_text = "===Unknown==="

    match = re.search(r'===\s*(CHAT|MAIL|FTP|P2P|Streaming|VoIP|BROWSING)\s*===', output_text)
    predicted_label = match.group(1) if match else 'Unknown'
    return predicted_label, f"Result: {output_text}"

predictions = []
actual_labels = []
print(len(test_data))

for index, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing cases"):
    predicted_label, enhanced_output = predict_traffic_label(row)

    actual_label = row['Label']

    test_data.at[index, 'Predicted_Label'] = predicted_label
    test_data.at[index, 'Model_Output'] = enhanced_output

    predictions.append(predicted_label)
    actual_labels.append(actual_label)

conf_matrix = confusion_matrix(actual_labels, predictions, labels=known_conditions)
accuracy = (conf_matrix.diagonal().sum()) / conf_matrix.sum()
precision, recall, f1_score, _ = precision_recall_fscore_support(actual_labels, predictions, labels=known_conditions,
                                                                 average='macro')

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