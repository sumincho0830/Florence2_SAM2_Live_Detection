# Prompt-Detection
Real-time prompt-based detection and tracking system for demonstration.

## Project Structure

The project is organized as follows:
- `data/pictures/`: Contains image data for training and testing
- `video/`: Video files for demonstration and testing
- `script/`: Scripts for running the system
- `src/`: Source code
  - `action_recognition/`: Code for action recognition functionality
  - `florence2_model/`: Implementation of the Florence2 model
  - `sam2_model/`: Implementation of the SAM2 model
- `Pipfile` and `Pipfile.lock`: Python dependency management

## Installation

### Prerequisites
- Python 3.10
- pipenv (for dependency management)

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/sumincho0830/Prompt-Detection.git
   cd Prompt-Detection
   ```

2. Install dependencies using pipenv:
   ```
   pipenv install
   ```

3. Activate the virtual environment:
   ```
   pipenv shell
   ```

### Download Checkpoint
1. Download SAM2 model checkpoint:
   ```
   # git bash
   
   cd checkpoints
   ./download_ckpts.sh
   ```

### Run Live Caption Demo
```
python .\script\florence2_live_caption_app.py
```

https://github.com/user-attachments/assets/f8c599b2-b559-403f-b2db-ba3277402bb3


### Run Live Tracking Demo
```
python .\script\florence2_sam2_live_detection_app.py
```
https://github.com/user-attachments/assets/eb34ff5a-4483-43cf-aada-88f229915cde



