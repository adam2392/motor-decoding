import os
import sys
from pathlib import Path

src = Path(__file__).parents[2]
if src not in sys.path:
    sys.path.append(str(src))
from mtsmorf.io.move.read import read_move_trial_epochs, read_trial_metadata
import pandas as pd
import numpy as np


def get_event_data(root, subject, run="01", event_key="Left Target", tmin=-0.5,
                   tmax=0.2, resample_rate=500, notch_filter=True, bandpass_lfreq=1.,
                   bandpass_hfreq=None, label_keyword="target_direction",
                   verbose=False, return_X_y=False):
    """Wrapper function to get data for a particular trial

    Parameters
    ----------
    root : BIDSPath, str
        Root of the BIDS dataset
    subject : str
        Subject identifier
    run : str, optional
        Which run of th experiment to get data from, by default "01"
    event_key : str, optional
        Event to time-lock to, by default "Left Target"
    tmin : float, optional
        Minimium time-step of time window with respect to event_key, by default -0.5
    tmax : float, optional
        Maximum time-step of time window with respect to event_key, by default 0.2
    resample_rate : int, optional
        Resampling rate, by default 500
    notch_filter : bool, optional
        Whether to apply notch filtering to raw signal, by default True
    bandpass_lfreq : float, optional
        Lower frequency of band-pass filter, by default 1.
    bandpass_hfreq : float, None, optional
        Higher frequency of band-pass filter, by default Nyquist frequency
    label_keyword : str, optional
        Trial keyword to use for generating labels, by default "target_direction"
    return_X_y : bool, optional
        Returns data arrays X, y and image_height, image_width parameters, 
        by default False
    verbose : bool, optional
        [description], by default False

    Returns
    -------
    epochs : mne.Epochs
        MNE data structure containing data matrix and patient metadata
    labels : nummpy.ndarray
        Array of target labels using label_keyword
    X : numpy.ndarray, optional
        Data matrix contained in epochs reshape as (n_samples, n_features)
    y : numpy.ndarray, optional
        Array containing target labels as (n_samples, 1)
    image_height : float, optional
        Number of rows for a single sample image
    image_width : float, optional
        Number of columns for a single sample image
    """
    epochs = read_move_trial_epochs(root, subject, run=run, event_key=event_key,
                                    tmin=tmin, tmax=tmax, resample_rate=resample_rate,
                                    notch_filter=True, l_freq=bandpass_lfreq,
                                    h_freq=bandpass_hfreq, verbose=verbose)

    trials = pd.DataFrame(read_trial_metadata(root, subject, run=run))
    trials = trials[~(trials.perturbed) & (trials.success)]
    labels = trials[label_keyword].values

    if return_X_y:
        data = epochs.get_data()
        ntrials, image_height, image_width = data.shape[0], np.prod(data.shape[1:-1]), data.shape[-1]

        X = data.reshape(ntrials, -1)
        y = labels.reshape(ntrials, -1)
        return X, y, image_height, image_width

    return epochs, labels


# TODO
def apply_transform(epochs, **kwargs):
    raise NotImplementedError