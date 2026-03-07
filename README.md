## Setup

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Copy config: `cp config.example.py config.py`
4. Edit `config.py` — set `DATA_ROOT` to your local dataset path
5. Download data: `python download_data.py`
   - Requires a Kaggle account and API token in `~/.kaggle/kaggle.json`
   - Only needs to be run once