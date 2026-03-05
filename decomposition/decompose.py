import enum
import numpy as np
import torch
from decomposition.BoundedTensorMVMD import BoundedTensorMVMD
from decomposition.FixedMVMD import FixedMVMD


class DecompositionMethod(enum.Enum):
    MVMD = 1


def decompose(method: DecompositionMethod, splits, alpha, K, frequency_bound, fs):
    if method == DecompositionMethod.MVMD:
        return decompose_mvmd(splits, alpha, K, frequency_bound, fs)


def decompose_mvmd(splits, alpha, K, frequency_bound, fs):
    bounded_tensor_MVMD = BoundedTensorMVMD(alpha=alpha, K=K)
    fixed_MVMD = FixedMVMD(alpha=alpha, K=K)

    combined_x_train = []
    combined_y_train = []
    combined_x_test = []
    combined_y_test = []
    combined_train_freqs = []
    combined_test_freqs = []
    for subject in splits[0]:
        subject_data = splits[0][subject]
        modes_3d, omegas_3d = bounded_tensor_MVMD(subject_data['x_train'],
                                                  freq_bounds=[frequency_bound for _ in range(K)], fs=fs)

        modes_3d = np.swapaxes(modes_3d, 0, 1)
        modes_3d = np.expand_dims(modes_3d, axis=1)
        subject_data['x_train'] = modes_3d
        combined_x_train.append(modes_3d)
        combined_y_train.append(subject_data['y_train'])

        mode_frequency_centers = (omegas_3d[-1, :] * fs).flatten()
        subject_data['frequency_centers'] = mode_frequency_centers

        new_x_test = []
        for test_trial_idx, test_trial in enumerate(subject_data['x_test']):
            modes, omegas = fixed_MVMD(test_trial, fixed_freqs=mode_frequency_centers, fs=fs)
            new_x_test.append(modes)
        new_x_test = np.expand_dims(new_x_test, axis=1)
        subject_data['x_test'] = new_x_test
        combined_x_test.append(new_x_test)
        combined_y_test.append(subject_data['y_test'])

        combined_train_freqs.append(torch.tensor(mode_frequency_centers, dtype=torch.float32))
        combined_train_freqs[-1] = combined_train_freqs[-1].unsqueeze(0).expand(len(subject_data['y_train']), -1)
        combined_test_freqs.append(torch.tensor(mode_frequency_centers, dtype=torch.float32))
        combined_test_freqs[-1] = combined_test_freqs[-1].unsqueeze(0).expand(len(subject_data['y_test']), -1)

    raw_train_x = np.concatenate(combined_x_train, axis=0)
    train_y = np.concatenate(combined_y_train, axis=0)
    raw_test_x = np.concatenate(combined_x_test, axis=0)
    test_y = np.concatenate(combined_y_test, axis=0)
    train_freqs = np.concatenate(combined_train_freqs, axis=0)
    test_freqs = np.concatenate(combined_test_freqs, axis=0)
    return raw_train_x, train_y, raw_test_x, test_y, train_freqs, test_freqs