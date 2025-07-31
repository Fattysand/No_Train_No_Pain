# No_Train_No_Pain


Dataset details：

VPN(service)：
| class   | count |
|------------|------|
| BROWSING   | 500  |
| CHAT       | 500  |
| FTP        | 500  |
| MAIL       | 500  |
| P2P        | 500  |
| Streaming  | 500  |
| VoIP       | 500  |

TFC(application)：
| class   | count |
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
| class   | count |
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
