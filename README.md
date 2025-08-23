# Eye-tracking method for knowing where people are looking on the monitor
## Wrap up Report
[eye tracking report.pdf](https://github.com/user-attachments/files/17470304/eye.tracking.report.pdf)

## Pipeline
![eye tracking pipeline](https://github.com/user-attachments/assets/2ff796d5-0ff7-4f22-b6f0-9455c1d725ce)

## Demo Video
https://github.com/user-attachments/assets/96e0ff93-f963-4b24-81b4-fbee77e12944

## Usage
### Install packages and prepare 
1.  `git clone https://github.com/gawon1224/EyeTracking.git`
2.  In your Conda virtual environment or locally, run `pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main`.
3.  Download the **`L2CSNet_gaze360.pkl`** file from the [pretrained model download link](https://www.google.com/search?q=https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd) and upload it to the `models` folder.
4.  Run `demo0922.py`.
      * You need to adjust the path to the **`L2CSNet_gaze360.pkl`** file to match your desktop's directory structure.

```python
# Example command
python demo0922.py --snapshot models/L2CSNet_gaze360.pkl
```
