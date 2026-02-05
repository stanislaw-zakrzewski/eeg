import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from classification.AdaptiveRiemannianTransformer import AdaptiveRiemannianTransformer
from classification.DomainAwareRiemannianTransformer import DomainAwareRiemannianTransformer
from classification.ModeWiseRiemannianTransformer import ModeWiseRiemannianTransformer
from classification.RiemannianModeNet import RiemannianModeNet, preprocess_riemannian

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy import signal

from data.parser import raw_to_train_test
from decomposition.BoundedTensorMVMD import BoundedTensorMVMD
from riemann.RiemannianTransformer import RiemannianTransformer
from tmvmd.FixedMVMD import FixedMVMD
from os import listdir
from os.path import isfile, join


# Import or Paste the previous definitions:
# 1. RiemannianModeNet (Class)
# 2. preprocess_riemannian (Function)
# (Assuming they are defined above in the script)

def train_and_test_pipeline():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    BATCH_SIZE = 50
    N_EPOCHS = 1000

    # Data Dimensions
    N_SAMPLES_TRAIN = 100
    N_SAMPLES_TEST = 20
    SEQ_LEN = 4  # Number of consecutive windows
    N_MODES = 3  # K modes from VMD
    alpha = 100 ** 2  # Decomposition filter steepness
    N_CHANNELS = 3  # EEG Electrodes
    frequency_bound = (6, 28)  # Limited band of frequencies for decomposition

    if False:
        # ==========================================
        # 2. DATA PREPARATION (The Missing Step!)
        # ==========================================
        print("--- 1. Generating & Preprocessing Data ---")

        edf_path = 'C:/Users/stz/Documents/GitHub/csp_classifier/preprocessed_subjects/'
        files_in_folder = [f for f in listdir(edf_path) if isfile(join(edf_path, f))]
        # edf_file_paths = ['C:/Users/stz/Documents/GitHub/csp_classifier/preprocessed_subjects/s14.edf']
        edf_file_paths = []
        for file in files_in_folder:
            edf_file_paths.append(f'{edf_path}{file}')

        fs, splits, label_dict = raw_to_train_test(edf_file_paths)

        # Initialize decomposition methods
        bounded_tensor_MVMD = BoundedTensorMVMD(alpha=alpha, K=N_MODES)
        fixed_MVMD = FixedMVMD(alpha=alpha, K=N_MODES)

        # Perform decompositions
        combined_x_train = []
        combined_y_train = []
        combined_x_test = []
        combined_y_test = []
        combined_train_freqs = []
        combined_test_freqs = []
        for subject in splits[0]:
            subject_data = splits[0][subject]
            modes_3d, omegas_3d = bounded_tensor_MVMD(subject_data['x_train'], freq_bounds=[frequency_bound for _ in range(N_MODES)], fs=fs)

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

        np.save('processed_data/raw_train_x.npy', raw_train_x)
        np.save('processed_data/train_y.npy', train_y)
        np.save('processed_data/raw_test_x.npy', raw_test_x)
        np.save('processed_data/test_y.npy', test_y)
        np.save('processed_data/train_freqs.npy', train_freqs)
        np.save('processed_data/test_freqs.npy', test_freqs)

    raw_train_x = np.load('processed_data/raw_train_x.npy')
    train_y = np.load('processed_data/train_y.npy')
    raw_test_x = np.load('processed_data/raw_test_x.npy')
    test_y = np.load('processed_data/test_y.npy')
    train_freqs = np.load('processed_data/train_freqs.npy')
    test_freqs = np.load('processed_data/test_freqs.npy')

    train_y = torch.from_numpy(train_y - 1)
    test_y = torch.from_numpy(test_y - 1)
    train_freqs = torch.from_numpy(train_freqs)
    test_freqs = torch.from_numpy(test_freqs)





    # # A. Generate RAW Dummy Data (Trials, Seq, Modes, Chan, Time)
    # # In real life, this is your VMD output tensor
    # raw_train_x = np.abs(signal.hilbert(modes_3d))
    # raw_train_x = modes_3d
    # raw_test_x = splits[0]['x_test']
    # new_test_x = []
    # for test_trial in raw_test_x:
    #     modes, omegas = fixed_MVMD(test_trial, fixed_freqs=mode_frequency_centers, fs=fs)
    #     new_test_x.append([modes])
    # raw_test_x = np.array(new_test_x)
    #raw_test_x = np.abs(signal.hilbert(np.array(new_test_x)))
    # # Labels (0=Left, 1=Right)
    # train_y = torch.from_numpy(splits[0]['y_train'])
    # test_y = torch.from_numpy(splits[0]['y_test'])
    # Frequencies (Context for Attention)

    # train_freqs = torch.tensor(mode_frequency_centers, dtype=torch.float32)
    # train_freqs = train_freqs.unsqueeze(0).expand(len(train_y), -1)

    # test_freqs = torch.tensor(mode_frequency_centers, dtype=torch.float32)
    # test_freqs = test_freqs.unsqueeze(0).expand(len(test_y), -1)

    # B. Apply Riemannian Preprocessing
    # This converts (B, S, K, C, T) -> (B, S, K, Features)
    # Note: We process Train and Test separately to avoid data leakage,
    # though strictly speaking, Riemannian Tangent Space often needs a
    # global 'Mean Covariance' reference. For this example, we fit independently.

    # riemann_pipe = RiemannianTransformer()
    # riemann_pipe = AdaptiveRiemannianTransformer()
    # riemann_pipe = ModeWiseRiemannianTransformer(n_modes=N_MODES)
    riemann_pipe = DomainAwareRiemannianTransformer(n_modes=N_MODES)

    print("Preprocessing Training Data...")
    riemann_pipe.fit(raw_train_x)
    features_train = torch.tensor(riemann_pipe.transform(raw_train_x), dtype=torch.float32)  # Returns Tensor
    # features_train = torch.tensor(preprocess_riemannian(raw_train_x), dtype=torch.float32)  # Returns Tensor

    print("Preprocessing Test Data...")
    # riemann_pipe.adapt(raw_train_x)
    # riemann_pipe.fit(raw_train_x)
    features_test = torch.tensor(riemann_pipe.transform(raw_test_x, calibration_data=raw_test_x), dtype=torch.float32)   # Returns Tensor
    # features_test = torch.tensor(preprocess_riemannian(raw_test_x), dtype=torch.float32)   # Returns Tensor

    print(f"Processed Feature Shape: {features_train.shape}")
    # Expected: (100, 4, 5, 36) for 8 channels

    # C. Create DataLoaders
    train_ds = TensorDataset(features_train, train_freqs, train_y)
    test_ds = TensorDataset(features_test, test_freqs, test_y)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================================
    # 3. MODEL SETUP
    # ==========================================
    model = RiemannianModeNet(n_modes=N_MODES, n_channels=N_CHANNELS)

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
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