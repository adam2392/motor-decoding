import numpy as np
import pandas as pd

from mne_bids.tsv_handler import _from_tsv
from mne_bids.path import _find_matching_sidecar


def _read_ch_anat(bids_path):
    electrodes_fpath = _find_matching_sidecar(
        bids_path, suffix="channels", extension=".tsv"
    )
    electrodes_tsv = _from_tsv(electrodes_fpath)

    # extract channel names and anatomy
    ch_names = electrodes_tsv["name"]
    ch_anat = electrodes_tsv["anat"]

    # create dictionary of ch name to anatomical region
    ch_anat_dict = {name: anat for name, anat in zip(ch_names, ch_anat)}
    return ch_anat_dict


def _get_anatomical_bad_chs(bids_path):
    # get the channel anat dict
    ch_anat_dict = _read_ch_anat(bids_path)

    # get bad channels from anatomy
    bads = []
    for ch_name, anat in ch_anat_dict.items():
        if (
            anat in ["out", "white matter", "cerebrospinal fluid"]
            or "ventricle" in anat
        ):
            bads.append(ch_name)
    return bads


def get_unperturbed_trial_inds(behav):
    """Get trial indices where force magnitude > 0."""
    if not isinstance(behav, pd.DataFrame):
        behav = pd.DataFrame(behav)

    behav[["successful_trial_flag", "force_magnitude"]] = behav[
        ["successful_trial_flag", "force_magnitude"]
    ].apply(pd.to_numeric)

    # filter out failed trials -- we don't want these anyway
    # successes = behav[behav.successful_trial_flag == 1]
    # successes.index = np.arange(len(successes))

    # filter out labels for perturbed trials
    unperturbed_trial_inds = behav[behav.force_magnitude == 0].index
    unperturbed_trial_inds = unperturbed_trial_inds.to_list()

    return unperturbed_trial_inds


def _preprocess_epochs(epochs, resample_rate=500, l_freq=1, h_freq=None):
    """Preprocess mne.Epochs object in the following way:
    1. Band-pass filter between l_freq and h_freq
    2. Downsample data to new resampling rate
    """
    epochs.load_data()

    # Band-pass filter
    new_epochs = epochs.filter(l_freq=l_freq, h_freq=h_freq)

    # Downsample epochs
    if resample_rate is not None:
        new_epochs = new_epochs.resample(resample_rate)

    return new_epochs


def _preprocess_raw(raw, bids_path, notch_filter: bool = True):
    # append bad channels from anatomical labeling
    bads = _get_anatomical_bad_chs(bids_path)
    raw.info["bads"].extend(bads)

    # only keep SEEG chs
    raw = raw.pick_types(meg=False, seeg=True,
                         eeg=False, ecog=True)

    # filter 60 Hz and harmonics
    if notch_filter:
        raw.load_data()
        line_freq = raw.info['line_freq']
        fs = raw.info["sfreq"]
        freqs = np.arange(line_freq, fs / 2, line_freq)
        raw = raw.notch_filter(freqs, verbose=False)

    return raw


def _pl(x, non_pl=""):
    """Determine if plural should be used."""
    len_x = x if isinstance(x, (int, np.generic)) else len(x)
    return non_pl if len_x == 1 else "s"


def _get_bad_epochs(behav_df, remove_perturbed: bool = True, remove_unsuccessful: bool = True):
    """Remove certain epochs: perturbed, unsuccessful"""
    if remove_perturbed:
        # get unperturbed trial indices
        unperturbed_trial_inds = get_unperturbed_trial_inds(behav_df)
    else:
        unperturbed_trial_inds = []

    if remove_unsuccessful:
        # get the successful trial indices
        success_trial_flag = behav_df["successful_trial_flag"].astype(int).values
        success_trial_flag = np.array(success_trial_flag)

        # successful trial indices
        success_inds = np.where(success_trial_flag == 1)[0]
    else:
        success_inds = []

    # keep indices are the unperturbed + successful
    if success_inds != [] and unperturbed_trial_inds != []:
        to_keep_inds = list(set(unperturbed_trial_inds).intersection(set(success_inds)))
    else:
        to_keep_inds = list(set(unperturbed_trial_inds).union(set(success_inds)))

    # drop indices
    drop_inds = [idx for idx in range(len(behav_df)) if idx not in to_keep_inds]
    return drop_inds
