#
# # Parameters
# K = 3 # Number of modes
# alpha = 1000**2 # Decomposition filter steepness
# frequency_bound = (6, 28) # Limited band of frequencies for decomposition
# n_splits = 1
#
# # Load Subject
# subject_edf_path = 'C:/Users/stz/Documents/GitHub/csp_classifier/preprocessed_subjects/s14.edf'
# subject = Subject(subject_edf_path)
# fs = subject.sampling_frequency
# label_dict = dict((v,k) for k,v in subject.id_dict.items())
#
# # Preprocess signal
# raw_signal = subject.raw
# select_channels(raw_signal, ['C3', 'Cz', 'C4'])
# channel_names = raw_signal.info['ch_names']
# filtered_raw_signal = raw_signal.filter(l_freq=5.0, h_freq=28,
#                                         l_trans_bandwidth=0.5,
#                                         fir_design='firwin')
#
#
# # Prepare training data
# picks = pick_types(filtered_raw_signal.info, meg=False, eeg=True, stim=False, eog=False,
#                    exclude='bads')
# epochs = Epochs(filtered_raw_signal, subject.events, subject.id_dict, .5, 2.5, proj=True, picks=picks,
#                 baseline=None, preload=True)
# epochs_train = epochs.copy()
# epochs_data_train = epochs_train.get_data(copy=False)
# cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
# cv_split = cv.split(epochs_data_train)
# labels = np.array(epochs.events[:, -1])
#
# # Initialize decomposition methods
# bounded_tensor_MVMD = BoundedTensorMVMD(alpha=alpha, K=K)
# fixed_MVMD = FixedMVMD(alpha=alpha, K=K)
#
# # Run classification
# for train_idx, test_idx in cv_split:
#     x_train, x_test = epochs_data_train[train_idx], epochs_data_train[test_idx]
#     y_train, y_test = labels[train_idx], labels[test_idx]
#
#     start = datetime.now()
#     # Decompose
#     modes_3d, omegas_3d = bounded_tensor_MVMD(x_train, freq_bounds=[frequency_bound for _ in range(K)], fs=fs)
#     mode_frequency_centers = (omegas_3d[-1, :] * fs).flatten()
#
#     # Fit TODO
#
#     train_execution_time = datetime.now() - start
#
#     # Perform tests, capture time to perform each trial
#     test_execution_times = []
#     for test_trial in x_test:
#         start = datetime.now()
#         modes, omegas = fixed_MVMD(test_trial, fixed_freqs=mode_frequency_centers, fs=fs)
#         test_execution_times.append(datetime.now() - start)
#
#
#     # print(f"\nInput Shape: {data_3d.shape}")  # (10, 4, 1000)
#     print(f"\nInput Shape: {epochs_data_train.shape}")  # (10, 4, 1000)
#     print(f"Output Shape: {modes_3d.shape}")  # (3, 10, 4, 1000)
#     print(f"Single frequency shape: {modes_3d[0, :, 0].shape}")  # (3, 10, 4, 1000)
#
#
#
#     avg_powers = {}
#     for mode_idx, mode in enumerate(range(1, K + 1)):
#         avg_powers[mode] = {}
#         current_mode = modes_3d[mode_idx]
#         for channel_idx, channel_name in enumerate(channel_names):
#             avg_powers[mode][channel_name] = {'left': [], 'right': []}
#             for trial_idx, trial in enumerate(x_train):
#                 avg_powers[mode][channel_name][label_dict[y_train[trial_idx]]].append(np.abs(signal.hilbert(current_mode[trial_idx][channel_idx])))
#                 # plt.plot(avg_powers[mode][channel_name][label_dict[y_train[trial_idx]]][-1])
#                 # plt.show()
#                 # left = np.mean(list(map(lambda x: np.abs(signal.hilbert(x)), modes_3d[mode_idx, :100, channel_idx])), axis=0)
#                 # right = np.mean(list(map(lambda x: np.abs(signal.hilbert(x)), modes_3d[mode_idx, 100:, channel_idx])), axis=0)
#                 # avg_powers[mode][channel_name] = {'left': left, 'right': right}
#
#     # 3. Create a time axis based on your 512Hz frequency
#     time = np.arange(len(modes_3d[0][0][0])) / fs
#
#     # 4. Plot
#     plt.figure(figsize=(24, 12))
#     for mode in range(1, K + 1):
#         for channel_idx, channel_name in enumerate(channel_names):
#             plt.subplot(K, len(channel_names), (mode - 1) * len(channel_names) + channel_idx + 1)
#             plt.title(channel_name)
#             x_data_1 = np.average(avg_powers[mode][channel_name]['left'], axis = 0)
#             plt.plot(time, x_data_1, color='blue', label=f'Left {mode_frequency_centers[mode - 1]:.2f}')
#             plt.plot(time, np.mean(avg_powers[mode][channel_name]['right'], axis=0), color='red', label=f'Right {mode_frequency_centers[mode - 1]:.2f}')
#             plt.xlabel("Time (s)")
#             plt.ylabel("Extracted Envelope (Hilbert)")
#             plt.grid(True, alpha=0.3)
#             plt.legend()
#     plt.show()
#
#     # print(f"Consensus Center Frequencies: {mode_frequency_centers}")
#     # print('end')
#     # average_powers = {}
#     # for freq_idx, freq_val_np in enumerate(list(mode_frequency_centers)):
#     #     freq_val = float(freq_val_np)
#     #     average_powers[freq_val] = {}
#     #     for channel_idx, channel_val in enumerate(channel_names):
#     #         average_powers[freq_val][channel_val] = {1: [], 2: []}
#     #         for trial_index, trial_label in enumerate(labels):
#     #             average_powers[freq_val][channel_val][trial_label].append(
#     #                 np.mean(modes_3d[freq_idx][trial_index][channel_idx] ** 2))
#     #         average_powers[freq_val][channel_val] = {'left': float(np.mean(average_powers[freq_val][channel_val][1])),
#     #                                                  'right': float(np.mean(average_powers[freq_val][channel_val][2]))}
#     # print(average_powers)
