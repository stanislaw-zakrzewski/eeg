from classification.DomainAwareRiemannianTransformer import DomainAwareRiemannianTransformer
from classification.DualStreamRiemannNet import DualStreamRiemannNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from classification.VMDInceptionNet import VMDInceptionNet
from data.export_import_npy import save_preprocessed_data, load_preprocessed_data
from data.parser import raw_to_train_test
from os import listdir
from os.path import isfile, join

from decomposition.decompose import decompose, DecompositionMethod


def preprocess_raw(x_tensor):
    # Input: (Batch, Seq, Modes, Chan, Time)
    # 1. Calculate Mean/Std per trial (Instance Normalization)
    # This helps against amplitude shifts between subjects
    mean = x_tensor.mean(dim=-1, keepdim=True)
    std = x_tensor.std(dim=-1, keepdim=True)

    # Avoid division by zero
    x_norm = (x_tensor - mean) / (std + 1e-6)

    return torch.tensor(x_norm, dtype=torch.float32)


def train_and_test_pipeline():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    TRAIN_BATCH_SIZE = 10
    TEST_BATCH_SIZE = 1
    N_EPOCHS = 50

    # Data Dimensions
    SEQ_LEN = 4  # Number of consecutive windows
    N_MODES = 5  # K modes from VMD
    alpha = 1000 ** 2  # Decomposition filter steepness
    frequency_bound = (6, 28)  # Limited band of frequencies for decomposition
    SELECTED_CHANNELS = ['C1', 'C2', 'C5', 'C3', 'C4', 'C6', 'FC3', 'CP3', 'FC4', 'CP4', 'Cz']
    N_CHANNELS = len(SELECTED_CHANNELS)

    if False:
        # ==========================================
        # 2. DATA PREPARATION (The Missing Step!)
        # ==========================================
        print("--- 1. Generating & Preprocessing Data ---")

        edf_path = 'C:/Users/stz/Documents/GitHub/csp_classifier/preprocessed_subjects/'
        files_in_folder = [f for f in listdir(edf_path) if isfile(join(edf_path, f))]
        # edf_file_paths = ['C:/Users/stz/Documents/GitHub/csp_classifier/preprocessed_subjects/s01.edf']
        edf_file_paths = []
        for file in files_in_folder:
            edf_file_paths.append(f'{edf_path}{file}')

        fs, splits, label_dict, channel_names = raw_to_train_test(edf_file_paths, SELECTED_CHANNELS)

        # Perform decompositions
        raw_train_x, train_y, raw_test_x, test_y, train_freqs, test_freqs = decompose(DecompositionMethod.MVMD, splits, alpha, N_MODES, frequency_bound, fs)

        save_preprocessed_data(raw_train_x, train_y, raw_test_x, test_y, train_freqs, test_freqs, 'processed_data/all_K5_alpha1000_11channels_6-28fb')

    raw_train_x, train_y, raw_test_x, test_y, train_freqs, test_freqs = load_preprocessed_data('processed_data/s14')
    # raw_train_x, train_y, raw_test_x, test_y, train_freqs, test_freqs = load_preprocessed_data('processed_data/all_K5_alpha1000_11channels_6-28fb')

    train_y = torch.from_numpy(train_y - 1)
    test_y = torch.from_numpy(test_y - 1)
    train_freqs = torch.from_numpy(train_freqs)
    test_freqs = torch.from_numpy(test_freqs)

    riemann_pipe = DomainAwareRiemannianTransformer(n_modes=N_MODES)

    print("Preprocessing Training Data...")
    riemann_pipe.fit(raw_train_x)
    features_train = torch.tensor(riemann_pipe.transform(raw_train_x), dtype=torch.float32)  # Returns Tensor

    print("Preprocessing Test Data...")
    # riemann_pipe.adapt(raw_train_x)
    # riemann_pipe.fit(raw_train_x)
    features_test = torch.tensor(riemann_pipe.transform(raw_test_x, calibration_data=raw_test_x), dtype=torch.float32)
    # features_test = []
    # # We want to calibrate each trial individually
    # for test_trial in raw_test_x:
    #     test_trial = np.array([test_trial])
    #     features_test.append(torch.tensor(riemann_pipe.transform(test_trial, calibration_data=np.concatenate([raw_train_x, test_trial])), dtype=torch.float32))
    # features_test = torch.cat(features_test, dim=0)

    print(f"Processed Feature Shape: {features_train.shape}")

    # C. Create DataLoaders
    train_ds = TensorDataset(features_train, train_freqs, train_y)
    test_ds = TensorDataset(features_test, test_freqs, test_y)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False)

    # ==========================================
    # 3. MODEL SETUP
    # ==========================================
    model = DualStreamRiemannNet(n_modes=N_MODES, n_channels=N_CHANNELS, dropout_rate=.5)

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # ==========================================
    # 4. TRAINING LOOP
    # ==========================================
    print("\n--- 2. Starting Training ---")
    model.train()  # Enable Learning

    for epoch in range(N_EPOCHS):
        total_loss = 0
        correct_train = 0
        total_train = 0

        for x_batch, freq_batch, y_batch in train_loader:
            # Zero Gradients
            optimizer.zero_grad()

            noise = torch.randn_like(x_batch) * 0.01
            x_batch = x_batch + noise

            # Forward Pass (Model infers weights & classifies)
            logits, att_weights = model(x_batch, freq_batch)

            # Compute Loss
            loss = criterion(logits, y_batch)

            # Backward Pass (Model learns temporal_cnn & attention)
            loss.backward()
            optimizer.step()

            # Track Performance
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += len(y_batch)

        acc = correct_train / total_train
        print(f"Epoch {epoch + 1:02d} | Loss: {total_loss:.4f} | Acc: {acc * 100:.1f}%")

    # ==========================================
    # 5. TESTING LOOP
    # ==========================================
    print("\n--- 3. Starting Evaluation ---")
    model.eval()  # Freeze weights

    total_correct = 0
    with torch.no_grad():
        for x_batch, freq_batch, y_batch in test_loader:
            # Forward
            logits, _ = model(x_batch, freq_batch)

            # Classify
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == y_batch).sum().item()

    final_acc = total_correct / len(test_y)
    print(f"Final Test Accuracy: {final_acc * 100:.1f}%")


# Run it
if __name__ == "__main__":
    # Ensure dependencies are installed: pip install pyriemann torch
    train_and_test_pipeline()
