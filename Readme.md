# EEG
This repository is meant to provide full coverage of processing and testing eeg data.

## Running in Docker
Before running the app in the container make sure that the EDF data is available, the EDF files should be in the folder that is also referenced in a configuration for tools, see [detailed instructions](src/tools/Readme.md#create_dataset).

Build the image
```
docker build -t eeg-app . 
```
Run the image 
```
docker run --rm -p 5000:5000 -v <<<IMAGE_ID>>> eeg-app
```
Process the data into BIDS: `http://127.0.0.1:5000/process-data`
Classify the processed data: `http://127.0.0.1:5000/test-classifiers` - returns json file with results of 5 fold cross valiation

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

[Tools documentation](src/tools/Readme.md)

List of tools:
- [create_dataset](src/tools/create_dataset.py) - converts multiple EDF files into one BIDS dataset
