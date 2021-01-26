import sys

from pathlib import Path

import numpy as np
import pandas as pd

from rerf.rerfClassifier import rerfClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from cv import cv_roc, cv_fit

sys.path.append(str(Path(__file__).parent.parent / "io"))

from read import get_trial_info, read_label, read_dataset


def _preprocess_epochs(epochs, resample_rate=500):
    """Preprocess mne.Epochs object in the following way:
    1. Low-pass filter up to Nyquist frequency
    2. Downsample data to 500 Hz
    """
    # low-pass filter up to sfreq/2
    fs = epochs.info["sfreq"]
    new_epochs = epochs.filter(l_freq=1, h_freq=fs / 2 - 1)

    # downsample epochs to 500 Hz
    new_epochs = new_epochs.resample(resample_rate)

    return new_epochs


def _preprocess_labels(labels, behav, events):
    """Preprocess labels by removing unsuccessful or perturbed trials."""
    if not isinstance(behav, pd.DataFrame):
        behav = pd.DataFrame(behav)

    if not isinstance(events, pd.DataFrame):
        events = pd.DataFrame(events)

    # filter out labels for unsuccessful trials
    successful_trials = behav[behav.successful_trial_flag == 1]
    successful_trials.index = np.arange(len(successful_trials))

    # filter out labels for perturbed trials
    perturbed_trial_inds = successful_trials[
        successful_trials.force_magnitude > 0
    ].index
    labels = np.delete(labels, perturbed_trial_inds)

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
    subject = bids_path.subject
    if subject == "":
        raise KeyError("specified bids_path has no subject.")

    # get epochs
    epochs = read_dataset(
        bids_path, kind=kind, tmin=tmin, tmax=tmax, event_key=event_key
    )
    epochs.load_data()
    epochs = _preprocess_epochs(epochs)

    # get labels
    labels, trial_ids = read_label(
        bids_path, trial_id=trial_id, label_keyword=label_keyword
    )
    behav, events = map(pd.DataFrame, get_trial_info(bids_path))
    labels = _preprocess_labels(labels, behav, events)

    return epochs, labels


def initialize_classifiers(image_height, image_width, n_jobs=1, random_state=None):
    """Initialize a list of classifiers to be compared."""

    mtsmorf = rerfClassifier(
        projection_matrix="MT-MORF",
        max_features="auto",
        n_jobs=n_jobs,
        random_state=random_state,
        image_height=image_height,
        image_width=image_width,
    )

    srerf = rerfClassifier(
        projection_matrix="S-RerF",
        max_features="auto",
        n_jobs=n_jobs,
        random_state=random_state,
        image_height=image_height,
        image_width=image_width,
    )

    lr = LogisticRegression(random_state=random_state)
    rf = RandomForestClassifier(random_state=random_state)
    dummy = DummyClassifier(strategy="most_frequent", random_state=random_state)

    clfs = [mtsmorf, srerf, lr, rf, dummy]

    return clfs


def fit_classifiers_cv(X, y, image_height, image_width, cv, metrics, random_state=None):
    clf_scores = dict()
    clfs = initialize_classifiers(
        image_height, image_width, n_jobs=-1, random_state=random_state
    )

    for clf in clfs:
        if clf.__class__.__name__ == "rerfClassifier":
            clf_name = clf.get_params()["projection_matrix"]
        elif clf.__class__.__name__ == "DummyClassifier":
            clf_name = clf.strategy
        else:
            clf_name = clf.__class__.__name__

        clf_scores[clf_name] = cv_fit(
            clf,
            X,
            y,
            cv=cv,
            metrics=metrics,
            n_jobs=None,
            return_train_score=True,
            return_estimator=True,
        )

    return clf_scores
