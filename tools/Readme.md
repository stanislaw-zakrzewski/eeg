# EEG Tools
## create_dataset
### Description
This tool is intended to transform separate EDF files into one BIDS dataset.

### Preparing EDF files
All EDF files intended to be parsed into BIDS dataset need to be in one folder. This folder needs to match `edf_location` path from [`create_dataset.json`](create_dataset.json) configuration file.

Each EDF file name must follow the pattern:
```
<<timestamp>>_<<subject>>_<<session>>_<<type>>.edf
```
where `timestamp` is just for user convenience and is not used in the dataset, `subject` is subject name/identifier, `session` is session number and `type` is type of performed task.

Example:
```
2023-02-23T10-03-35_BS_1_realAudioOpen.edf
```

After parsing to BIDS format `subject` and `type` are fused together to create a unique identifier needed for further processing. If we were to use just the `subject` identifier all tasks will be mixed up into one and by specifying `subject` and `task` relation we avoid this.

### Configuration File
Configuration file contains all of the parsing parameters needed for multi EDF to BIDS parsing.

Those parameters are:
#### `annotation_mapping`
Dict containing mappings of annotations from EDF files to the ones available in BIDS. Currently available values:
- `left_hand`
- `right_hand`
- `hands`
- `feet`
- `rest`
- `left_hand_right_foot`
- `right_hand_left_foot`
- `tongue`
- `navigation`
- `subtraction`
- `word_ass` (for word association)

#### `edf_location`
Path to folder in which EDF files ready for parsing to BIDS are stored.

#### `bids_location`
Path to be used as a place to store new BIDS dataset.

#### `channel_mapping`
Dict containing mappings of channel names.

Configuration file example:
```json
{
  "annotation_mapping": {
    "movement": "hands",
    "rest": "rest"
  },
  "edf_location": "../data",
  "bids_location": "../bids_datasets/sample_dataset",
  "channel_mapping": {
    "CZ": "Cz"
  }
}
```

### Running create_dataset.py tool
After making sure that all edf files are named appropriately and configuration file is up to date run [`create_dataset.py`](create_dataset.py) from the main directory. 
