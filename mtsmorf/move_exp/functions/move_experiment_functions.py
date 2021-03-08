import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "io"))

from mtsmorf.io.read import get_trial_info_pd, get_unperturbed_trial_inds, read_label, read_dataset


def _preprocess_epochs(epochs, resample_rate=500):
    """Preprocess mne.Epochs object in the following way:
    1. Low-pass filter up to Nyquist frequency
    2. Downsample data to 500 Hz
    """
    # Low-pass filter up to sfreq/2
    fs = epochs.info["sfreq"]
    new_epochs = epochs.filter(l_freq=1, h_freq=fs / 2 - 1)

    # Downsample epochs to 500 Hz
    new_epochs = new_epochs.resample(resample_rate)

    return new_epochs


def get_preprocessed_epochs(
        bids_path,
        kind="ieeg",
        tmin=-0.2,
        tmax=0.5,
        event_key="Left Target",
):
    """Preprocess mne.Epochs object and drop perturbed trials."""
    # Grab original epochs structure
    epochs = read_dataset(
        bids_path, kind=kind, tmin=tmin, tmax=tmax, event_key=event_key
    )
    epochs.load_data()

    # Preprocess epochs
    epochs = _preprocess_epochs(epochs)

    # Drop perturbed trials
    behav, _ = get_trial_info_pd(bids_path)
    unperturbed_trial_inds = get_unperturbed_trial_inds(behav)
    perturbed_trial_inds = [
        i for i in range(len(epochs)) if not i in unperturbed_trial_inds
    ]
    epochs.drop(perturbed_trial_inds)

    return epochs


def get_preprocessed_labels(
        bids_path, trial_id=None, label_keyword="target_direction", verbose=False
):
    """Read labels for each trial for the specified keyword. Keep labels for
    successful and unperturbed trials.
    """
    behav, events = get_trial_info_pd(bids_path, verbose=verbose)
    labels, _ = read_label(bids_path, trial_id=trial_id, label_keyword=label_keyword)

    # Keep perturbed trial inds
    unperturbed_trial_inds = get_unperturbed_trial_inds(behav)
    labels = labels[unperturbed_trial_inds]

    return labels


def get_event_data(
        bids_path,
        kind="ieeg",
        tmin=-0.2,
        tmax=0.5,
        event_key="Left Target",
        trial_id=None,
        label_keyword="target_direction",
):
    """Read preprocessed mne.Epochs data structure time locked to label_keyword
    with corresponding trial information.
    """
    # Get preprocessed epochs data structure
    epochs = get_preprocessed_epochs(
        bids_path,
        kind=kind,
        tmin=tmin,
        tmax=tmax,
        event_key=event_key,
    )

    # Get labels for corresponding label_keyword
    labels = get_preprocessed_labels(
        bids_path, trial_id=trial_id, label_keyword=label_keyword
    )

    return epochs, labels
