import argparse
import json
import os
import sys
import traceback
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mne_bids.path import BIDSPath
from mne import Epochs
from mne.time_frequency import EpochsTFR
from mne.time_frequency.tfr import tfr_morlet
from rerf.rerfClassifier import rerfClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold

if os.path.abspath(Path(__file__).parents[2]) not in sys.path:
    sys.path.append(os.path.abspath(Path(__file__).parents[2]))

from mtsmorf.move_exp.cv import fit_classifiers_cv
from mtsmorf.move_exp.functions.time_window_selection_functions import (
    get_event_durations,
    plot_event_durations,
    plot_event_onsets,
)
from mtsmorf.move_exp.plotting import plot_classifier_performance, plot_roc_multiclass_cv, plot_roc_aucs
from mtsmorf.io.move.read import read_move_trial_epochs, read_trial_metadata
from mtsmorf.io.utils import NumpyEncoder
from sklearn.utils import check_random_state


def decode_movement_planning(
    root,
    subject,
    destination_path,
    cv,
    metrics,
    domain,
    shuffle=True,
    l_freq=1,
    h_freq=None,
    band="",
    n_jobs=1,
    random_state=None,
):
    rng = check_random_state(random_state)

    session = 'efri'
    task = 'move'
    acquisition = 'seeg'
    datatype = 'ieeg'
    extension = '.vhdr'
    run = '01'

    bids_path = BIDSPath(
        subject=subject, session=session, task=task,
        acquisition=acquisition, datatype=datatype,
        run=run, suffix=datatype,
        extension=extension, root=root)

    # go_cue_durations = get_event_durations(
    #     bids_path, event_key="Left Target", periods=-1
    # )

    # tmin = 0.0
    # tmax = max(go_cue_durations)
    tmin, tmax = -0.25, 0.0
    destination = Path(destination_path) / f"tmin={tmin}_tmax={tmax}_shuffle={shuffle}/{domain}_domain/"

    epochs = read_move_trial_epochs(root, subject, event_key="Left Target", tmin=tmin, tmax=tmax)
    trials = read_trial_metadata(root, subject)
    trials = pd.DataFrame(trials)
    labels = trials.query("perturbed == False & success == True").target_direction.values

    resample_rate = 500
    if domain.lower() == "time":
        if h_freq is None:
            h_freq = epochs.info['sfreq'] / 2.0 - 1
        epochs = epochs.filter(l_freq=l_freq, h_freq=h_freq)
        epochs = epochs.resample(resample_rate)
        data = epochs.get_data()

        t = epochs.times
        ntrials, nchs, nsteps = data.shape
        image_height = nchs
        image_width = nsteps

    elif domain.lower() in ["frequency", "freq"]:
        frequency_bands = dict(
            delta=(0.5, 4),
            theta=(4, 8),
            alpha=(8, 13),
            beta=(13, 30),
            gamma=(30, 70),
            hi_gamma=(70, 200),
        )
        data = []
        for band, (l_freq, h_freq) in frequency_bands.items():
            # Get data for before movement onset
            epochs_band = epochs.filter(l_freq=l_freq, h_freq=h_freq).apply_hilbert(
                envelope=True
            )
            epochs_band = epochs_band.resample(resample_rate)
            data.append(epochs_band.get_data())

        # Transpose to have data shape (ntrials, nchs, nfreqs, nsteps)
        data = np.array(data).transpose(1, 2, 0, 3)

        ntrials, nchs, nfreqs, nsteps = data.shape
        image_height = nchs * nfreqs
        image_width = nsteps

    else:
        raise ValueError('domain must be one of "time", "freq", or "frequency".')

    if shuffle:
        # Shuffle along image rows
        p = rng.permutation(image_height)
        data = data.reshape(ntrials, image_height, image_width)
        data = data[:, p]

    X = data.reshape(ntrials, -1)
    y = labels

    cv_scores = fit_classifiers_cv(
        X,
        y,
        image_height,
        image_width,
        cv,
        metrics,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    clf_name = "MT-MORF"
    scores = cv_scores[clf_name]
    best_ind = np.argmax(scores["test_roc_auc_ovr"])
    best_estimator = scores["estimator"][best_ind]
    best_train_inds = scores["train_inds"][best_ind]
    best_test_inds = scores["test_inds"][best_ind]

    X_train = X[best_train_inds]
    y_train = y[best_train_inds]
    X_test = X[best_test_inds]
    y_test = y[best_test_inds]

    if not os.path.exists(destination):
        os.makedirs(destination)

    for clf_name, clf_scores in cv_scores.items():

        estimator = clf_scores.get("estimator")
        if estimator is not None:
            del clf_scores["estimator"]

        with open(destination / f"{subject}_{clf_name}_results_{band}.json", "w") as fout:
            json.dump(clf_scores, fout, cls=NumpyEncoder)
            print(f"{subject.upper()} CV results for {clf_name} saved as json.")
        
        if estimator is not None:
            clf_scores["estimator"] = estimator

    ## Plot results
    # fig, axs = plt.subplots(ncols=2, dpi=100, figsize=(16, 6))
    # axs = axs.flatten()
    # plot_classifier_performance(cv_scores, X, y, axs=axs)
    # axs[0].set(
    #     title=f"{subject.upper()} ROC Curves for 'At Center' vs. 'Left Target' ({domain.capitalize()} Domain)",
    # )
    # axs[1].set(
    #     title=f"{subject.upper()}: Accuracies for 'At Center' vs. 'Left Target' ({domain.capitalize()} Domain)",
    # )
    # fig.tight_layout()

    # plt.savefig(destination / f"movement_planning_performance_{domain}_domain.png")
    # plt.close(fig)
    # print(f"Figure saved at {destination}/movement_planning_performance_{domain}_domain.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=str, help="subject ID (e.g. efri02)")
    args = parser.parse_args()
    subject = args.subject

    with open(Path(os.path.dirname(__file__)) / "config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    root = Path(config["bids_root"])
    results_path = Path(config["results_path"])

    seed = 1
    n_splits = 5
    cv = StratifiedKFold(n_splits)
    metrics = dict(
        accuracy="accuracy",
        cohen_kappa_score=make_scorer(cohen_kappa_score),
        roc_auc_ovr="roc_auc_ovr",
    )

    destination_path = results_path / "planning_movement" / subject
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # decode_movement_planning(
    #     root, subject, destination_path, cv, metrics, "time", shuffle=True, random_state=seed
    # )

    # decode_movement_planning(
    #     root, subject, destination_path, cv, metrics, "freq", shuffle=True, random_state=seed
    # )

    frequency_bands = dict(
        delta=(0.5, 4),
        theta=(4, 8),
        alpha=(8, 13),
        beta=(13, 30),
        gamma=(30, 70),
        hi_gamma=(70, 200),
    )

    for band, (l_freq, h_freq) in frequency_bands.items():
        decode_movement_planning(
            root, subject, destination_path, cv, metrics, "time", shuffle=False, l_freq=l_freq, h_freq=h_freq, band=band, random_state=seed
        )
