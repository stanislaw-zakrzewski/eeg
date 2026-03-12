import os
import mne
import numpy as np
import json
from mne_bids import write_raw_bids, BIDSPath


def create_bids_dataset():
    # Read configuration file
    with open('src/tools/create_dataset.json', 'r') as file:
        config = json.load(file)

    annotation_numbering = {}

    # Get a list of all EDF files in the data directory
    edf_files = [f for f in os.listdir(config['edf_location']) if f.endswith('.edf')]

    # Loop through each EDF file and convert to BIDS
    for edf_file in edf_files:
        # Construct the full path to the EDF file
        raw_path = os.path.join(config['edf_location'], edf_file)

        # Read the raw data
        raw = mne.io.read_raw_edf(raw_path, preload=True)

        # Create a new stimulus channel
        stim_data = np.zeros((1, raw.n_times))

        # Rename annotations as specified in the configuration file
        annotations = raw.annotations
        annotations.rename(config['annotation_mapping'])

        # Add events from annotations to the stimulus channel
        for unique_annotation_name in set(raw.annotations.description):
            if unique_annotation_name not in annotation_numbering:
                annotation_index = len(annotation_numbering) + 1
                annotation_numbering[unique_annotation_name] = len(annotation_numbering) + 1

        # Add events from annotations to the stimulus channel
        for ann in raw.annotations:
            onset_sample = int(ann['onset'] * raw.info['sfreq'])
            stim_data[0, onset_sample] = annotation_numbering[ann['description']]
        info = mne.create_info(['Stim'], raw.info['sfreq'], ['stim'])
        stim_raw = mne.io.RawArray(stim_data, info)
        raw.add_channels([stim_raw], force_update_info=True)

        # Rename channels as specified in the configuration file
        raw.rename_channels(config['channel_mapping'])

        # Extract timestamp, subject, type and session from the filename
        # Assuming filename format is 'subject_session.edf'
        timestamp, subject, session, type = os.path.splitext(edf_file)[0].split('_')

        # Create a BIDSPath object
        bids_path = BIDSPath(subject=f"{subject}{type}", task=type, session=session, root=config['bids_location'])

        # Write the raw data to BIDS format
        write_raw_bids(raw, bids_path, overwrite=True, allow_preload=True, format='EDF', event_id=annotation_numbering)

    # Save events for future use within BIDS dataset
    events_json_path = os.path.join(config['bids_location'], 'events.json')
    with open(events_json_path, 'w') as f:
         json.dump(annotation_numbering, f)

if __name__ == "__main__":
    create_bids_dataset()
