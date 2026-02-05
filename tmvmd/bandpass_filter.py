# from config.config import Configurations


def bandpass_filter(signal, sampling_frequency, l_frequency, h_frequency):
    """Band pass filtering for MNE's raw signal.

    Parameters
    ----------
    signal : mne.io.Raw
        EEG signal in MNE's Raw format.
    l_frequency : int
        Highpass frequency for bandpass filter.
    h_frequency : float
        Lowpass frequency for bandpass filter.
    sampling_frequency : int
        Sampling frequency, leaving empty defaults to configuration
    """

    l_trans_bandwidth = min(2, l_frequency)
    return signal.filter(l_frequency, h_frequency, l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=2,
                         filter_length=sampling_frequency * 4,
                         fir_design='firwin',
                         skip_by_annotation='edge')
