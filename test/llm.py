import mne
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    exponential_moving_standardize, preprocess, Preprocessor
)

# 1. Load a standard Motor Imagery dataset (e.g., BCI Competition IV 2a)
# This automatically downloads the data for Subject 1
dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1])

# 2. Define Preprocessing Pipeline
# - Pick EEG channels (ignore EOG/Stim channels)
# - Convert to Microvolts (scaling)
# - Bandpass Filter (4-38 Hz)
# - Exponential Moving Standardization (Standardizes data based on previous time steps)
preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),
    Preprocessor(lambda x: x * 1e6),  # Convert V to uV
    Preprocessor('filter', l_freq=4, h_freq=38),
    Preprocessor(exponential_moving_standardize,
                 factor_new=1e-3, init_block_size=1000)
]

# 3. Apply Preprocessing
preprocess(dataset, preprocessors, n_jobs=-1)


from braindecode.preprocessing import create_windows_from_events

# Define window size (e.g., 4 seconds usually used in BCI IV 2a)
# sampling_rate is usually 250Hz for this dataset
sfreq = dataset.datasets[0].raw.info['sfreq']
window_size_samples = int(4 * sfreq)

# Cut the data into epochs based on markers
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples,
    drop_last_window=False,
    preload=True
)

# Split into Train/Valid/Test (Standard PyTorch split)
splitted = windows_dataset.split('session')
train_set = splitted['0train']
valid_set = splitted['1test']


import torch
from braindecode.models import EEGConformer
from braindecode.util import set_random_seeds

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

set_random_seeds(seed=2023, cuda=cuda)

# Get input dimensions from the dataset
n_channels = train_set[0][0].shape[0]
input_window_samples = train_set[0][0].shape[1]
n_classes = 4  # Left Hand, Right Hand, Feet, Tongue

model = EEGConformer(
    n_outputs=n_classes,
    n_chans=n_channels,
    n_times=input_window_samples,
    n_filters_time=40,    # Standard parameter from Song et al.
    filter_time_length=25,# Standard parameter
    pool_time_length=75,  # Standard parameter
    num_layers=6,          # 6 Transformer layers
    num_heads=10,         # 10 Attention heads
    final_fc_length='auto' # Automatically calculate FC size
)

# Move model to GPU
model.to(device)


from braindecode import EEGClassifier
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
import torch

# Ensure n_classes is defined (for BCI IV 2a, it is 4)
n_classes = 4
classes = list(range(n_classes)) # [0, 1, 2, 3]

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=0.0002,
    optimizer__weight_decay=0.01,
    batch_size=64,
    callbacks=[
        "accuracy",
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=200 - 1))
    ],
    device=device,
    classes=classes,  # <--- ADD THIS LINE
    max_epochs=200
)

# Now this will work
clf.fit(train_set, y=None)


import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.plot(clf.history[:, 'train_accuracy'], label='Train')
plt.plot(clf.history[:, 'valid_accuracy'], label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Get final accuracy
print(f"Final Validation Accuracy: {clf.history[-1, 'valid_accuracy']:.2f}")