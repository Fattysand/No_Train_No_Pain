# No_Train_No_Pain


Dataset details：

VPN(service)：
| Class   | Count |
|------------|------|
| BROWSING   | 500  |
| CHAT       | 500  |
| FTP        | 500  |
| MAIL       | 500  |
| P2P        | 500  |
| Streaming  | 500  |
| VoIP       | 500  |

TFC(application)：
| Class   | Count |
|--------------------|------|
| BitTorrent         | 500  |
| FTP                | 500  |
| Gmail              | 500  |
| MySQL              | 500  |
| Outlook            | 500  |
| SMB                | 500  |
| Skype              | 500  |
| Weibo              | 500  |
| WorldOfWarcraft    | 500  |

IOT(application)：
| Class   | Count |
|------------------------|------|
| Interactions_Cameras   | 500  |
| Power_Audio            | 500  |
| Power_Cameras          | 500  |
| Idle                   | 500  |
| Interactions_Audio     | 500  |
| Power_Other            | 315  |
| ComingHome             | 267  |
| LeavingHome            | 167  |
| Interactions_Other     | 133  |


The provided dataset is a pre-processed dataset. The original dataset is too large to be included here, but you can download it and use CICFlowMeter for feature extraction.

The workflow is to directly locate the scripts under the `llm_classify` folder, fill in the local LLM model path or API token, and execute them using Python. Then, use the scripts under the `TD` folder on the obtained results to get the final output.

### 🔧 Running Instructions

To reproduce the results, please follow the steps below:

1. **Modify the local path** in the script `llm_classify/qwen3/qwen_tfc_proc.py` to match your environment.
2. Execute the script by running:

   ```bash
   python llm_classify/qwen3/qwen_tfc_proc.py
   ```

3. After obtaining the intermediate output, update the **input path** accordingly in `TD/temporal_denoising.py`.

4. Run the temporal denoising module:

   ```bash
   python TD/temporal_denoising.py
   ```

By following these steps, you will obtain the final classification results as reported in the study.