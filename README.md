# Installation

## Requirements
- [Git](https://git-scm.com/)

```bash
# Copy the code to your computer
git clone https://github.com/spjy/covid-ct.git

# Create virtual environment
python3 -m venv env

# Activate
source env/bin/activate

# Install packages needed
pip3 install -r requirements.txt
```

## Running

```bash
python3 imgs.py /path/to/extracted/folders # Extracts images from class folders
python3 randomize-five-fold.py # Distributes data into folds (Evenly)
python3 normalize.py # Gets mean and std of dataset
python3 eg.py # Trains model
```
