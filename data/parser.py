from subject import Subject
from tmvmd.select_channels import select_channels
from mne import Epochs, pick_types
import numpy as np
from sklearn.model_selection import ShuffleSplit



def raw_to_train_test(edf_paths, n_splits=4):
    """
    Args:
        edf_paths: paths to edf files
        n_splits: number of train/test splits
    """

    splits = [{} for _ in range(n_splits)]

    fs = None
    for edf_path in edf_paths:
        try:
            subject = Subject(edf_path)
            if fs is None:
                fs = subject.sampling_frequency
            elif fs != subject.sampling_frequency:
                raise AssertionError("sampling frequencies across edf files do not match")
            label_dict = dict((v, k) for k, v in subject.id_dict.items())

            raw_signal = subject.raw
            select_channels(raw_signal, ['C3', 'Cz', 'C4'])
            channel_names = raw_signal.info['ch_names']
            filtered_raw_signal = raw_signal.filter(l_freq=5.0, h_freq=28,
                                                    l_trans_bandwidth=0.5,
                                                    fir_design='firwin')

            # Prepare training data
            picks = pick_types(filtered_raw_signal.info, meg=False, eeg=True, stim=False, eog=False,
                               exclude='bads')
            epochs = Epochs(filtered_raw_signal, subject.events, subject.id_dict, .5, 2.5, proj=True, picks=picks,
                            baseline=None, preload=True)
            epochs_train = epochs.copy()
            epochs_data_train = epochs_train.get_data(copy=False)
            cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
            cv_split = cv.split(epochs_data_train)
            labels = np.array(epochs.events[:, -1])
            for split_id, (train_idx, test_idx) in enumerate(cv_split):
                x_train, x_test = epochs_data_train[train_idx], epochs_data_train[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]
                splits[split_id][edf_path] = {'x_train': x_train,
                                              'y_train': y_train,
                                              'x_test': x_test,
                                              'y_test': y_test, }
        except:
            print(edf_path)

    return fs, splits, label_dict


def raw_to_subject_signals(edf_paths, n_splits=4):
    """
    Args:
        edf_paths: paths to edf files
        n_splits: number of train/test splits
    """

    fs = None
    subject_signals = {}
    for edf_path in edf_paths:
        subject = Subject(edf_path)
        if fs is None:
            fs = subject.sampling_frequency
        elif fs != subject.sampling_frequency:
            raise AssertionError("sampling frequencies across edf files do not match")
        label_dict = dict((v, k) for k, v in subject.id_dict.items())

        raw_signal = subject.raw
        select_channels(raw_signal, ['C3', 'Cz', 'C4'])
        filtered_raw_signal = raw_signal.filter(l_freq=5.0, h_freq=28,
                                                l_trans_bandwidth=0.5,
                                                fir_design='firwin')

        # Prepare training data
        picks = pick_types(filtered_raw_signal.info, meg=False, eeg=True, stim=False, eog=False,
                           exclude='bads')
        epochs = Epochs(filtered_raw_signal, subject.events, subject.id_dict, .5, 2.5, proj=True, picks=picks,
                        baseline=None, preload=True)
        epochs_train = epochs.copy()
        epochs_data_train = epochs_train.get_data(copy=False)
        labels = np.array(epochs.events[:, -1])

        subject_signals[edf_path] = {
            'x': epochs_data_train,
            'y': labels,
        }

    return fs, subject_signals, label_dict
