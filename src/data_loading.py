import os
import json
from moabb.datasets.base import LocalBIDSDataset
from moabb.paradigms import MotorImagery

#### `paradigm` available values: `imagery`, `p300`, `ssvep`

def get_dataset(dataset_path, subject_list, interval, paradigm):
    # Load event identifiers from custom events.json file stored within BIDS dataset
    events_json_path = os.path.join(dataset_path, 'events.json')
    with open(events_json_path, 'r') as file:
        events = json.load(file)

    dataset = LocalBIDSDataset(dataset_path, events=events, paradigm=paradigm, interval=interval)

    if len(subject_list) != 0:
        dataset.subject_list = subject_list

    return dataset


def get_paradigm(channels):
    return MotorImagery(channels=channels)
