from tkinter import filedialog as fd
from subject import Subject
import mne
import numpy as np
from scipy.fft import rfft, rfftfreq
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def generate_edf_paths(prefix, subject_count):
    generated_edf_paths = []
    for subject_index in range(subject_count):
        if subject_index < 9:
            subject_id = f's0{subject_index + 1}'
        else:
            subject_id = f's{subject_index + 1}'
        generated_edf_paths.append(f'{prefix}/{subject_id}.edf')
    return generated_edf_paths


''' Constants   '''
# subject_edf_path = fd.askopenfilename(filetypes=[("European Data Format files", "*.edf")])
subject_edf_path = 'C:/Users/stz/Documents/GitHub/csp_classifier/preprocessed_subjects/s01.edf'
EDF_PATHS = generate_edf_paths('C:/Users/stz/Documents/GitHub/csp_classifier/preprocessed_subjects', 52)

subject =
ELECTRODES = subject.electrode_names
# ELECTRODES = ['C3', 'C4']

sampling_frequency = subject.sampling_frequency

def process_subject(subject):
    raw = subject.get_raw_copy()
    epochs = mne.Epochs(
        raw,
        event_id=["left", "right"],
        tmin=0,
        tmax=7,
        picks=ELECTRODES,
        baseline=None,
        preload=True,
    )
    trials = epochs.get_data()
    processed_data = []
    for trial in trials:
        window_start = 0
        window_stop = int(sampling_frequency)
        while window_stop < len(trial[0]):
            processed_channels = []
            for channel in trial:
                window_frame = channel[window_start:window_stop]
                fft_window_frame = np.abs(rfft(window_frame))
                freqs = rfftfreq(sampling_frequency, 1 / sampling_frequency)
                processed_channels.append(fft_window_frame[6:15])
            processed_data.append(np.concatenate(processed_channels).ravel().tolist())
            window_start += int(sampling_frequency / 2)
            window_stop += int(sampling_frequency / 2)
    return processed_data

subjects_data = []
for subject_path in EDF_PATHS:

    s = Subject(subject_path)
    subjects_data = [*subjects_data, *process_subject(s)]

# subject_data = process_subject(subject)
kmeans = KMeans(init="random", n_clusters=20, n_init=4, random_state=0)
kmeans.fit(subjects_data)

buckets = [[] for _ in range(13)]
times = []
classes = []
for idx, el in enumerate(subjects_data):
    pred = kmeans.predict([el])[0]
    buckets[idx % 13].append(pred)
    times.append(idx % 13 / 2 - 1.5)
    classes.append(pred)
for idx, bucket in enumerate(buckets):
    values, counts = np.unique(bucket, return_counts=True)
unique_times = list(set(times))
unique_times.append(5)
unique_times.sort()
print(unique_times)

df = pd.DataFrame(data={'time': times, 'class': classes})
ax = sns.histplot(data=df, x="time", hue="class", multiple="stack", bins=unique_times)
ax.axvspan(0, .5, color='red', alpha=0.3, label='P300', zorder=0)
ax.axvspan(.5, 3, color='yellow', alpha=0.3, label='Movement', zorder=0)
ax.legend()
plt.show()
