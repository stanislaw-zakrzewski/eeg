# EEG
This repository is meant to provide full coverage of processing and testing eeg data.

## Installing dependencies
To install all required dependencies please use poetry:
```
pip install poetry
poetry install
```
This will create a new `.venv` inside the eeg repository and install all dependencies in it.

## Processing and testing
Main workflow in this repository is stored in [`src/main.py`](src/main.py) file. 

[Main workflow documentation](src/Readme.md)

## Tools
Tools help with different operations on eeg data that are not part of the main workflow.

[Tools documentation](tools/Readme.md)

List of tools:
- [create_dataset](tools/create_dataset.py) - converts multiple EDF files into one BIDS dataset
