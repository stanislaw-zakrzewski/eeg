import numpy as np


def save_preprocessed_data(raw_train_x, train_y, raw_test_x, test_y, train_freqs, test_freqs, folder_path):
    np.save(f'{folder_path}/raw_train_x.npy', raw_train_x)
    np.save(f'{folder_path}/train_y.npy', train_y)
    np.save(f'{folder_path}/raw_test_x.npy', raw_test_x)
    np.save(f'{folder_path}/test_y.npy', test_y)
    np.save(f'{folder_path}/train_freqs.npy', train_freqs)
    np.save(f'{folder_path}/test_freqs.npy', test_freqs)


def load_preprocessed_data(folder_path):
    raw_train_x = np.load(f'{folder_path}/raw_train_x.npy')
    train_y = np.load(f'{folder_path}/train_y.npy')
    raw_test_x = np.load(f'{folder_path}/raw_test_x.npy')
    test_y = np.load(f'{folder_path}/test_y.npy')
    train_freqs = np.load(f'{folder_path}/train_freqs.npy')
    test_freqs = np.load(f'{folder_path}/test_freqs.npy')
    return raw_train_x, train_y, raw_test_x, test_y, train_freqs, test_freqs