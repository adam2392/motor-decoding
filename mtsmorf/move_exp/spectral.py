def compute_freq_band_power(epochs, l_freq, h_freq, name, method='hilbert'):
    epochs = epochs.copy()

    if method == 'hilbert':
        # https://github.com/mne-tools/mne-python/pull/8781
        band_epochs = epochs.filter(l_freq=l_freq, h_freq=h_freq).apply_hilbert(envelope=False)

        # get data

    return band_epochs.get_data()
