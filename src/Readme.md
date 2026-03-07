# Main Workflow
Currently running main will process all subjects from selected dataset (`dataset_path` in [config.json](config.json)).

Make sure that [config.json](config.json) reflects your desired configuration and `main.py` is run from main repo folder (`eeg`).

## config.json
Config for main workflow contains:
- `dataset_path` - path to BIDS dataset
- `interval` - list of two values, first indicating window start time and second one for window end time, both in relation to cue
- `overwrite` - when false will use cached results, if true will overwrite them and process again
- `paradigm` - one of: `imagery`, `p300` or `ssvep`
- `save_results_to_csv` - when true will save results to csv file
- `selected_channels` - list of channels to process, passing empty list will process all channels 
- `subject_list` - list of subjects to process (names of the subjects), passing empty list will process all subjects