import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from mne_bids.path import BIDSPath
from mne import Epochs
from mne.time_frequency import EpochsTFR
from mne.time_frequency.tfr import tfr_morlet
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold

from functions.move_experiment_functions import get_event_data
from plotting import plot_classifier_performance
import json
import traceback

if os.path.abspath(Path(__file__).parents[2]) not in sys.path:
    sys.path.append(os.path.abspath(Path(__file__).parents[2]))

from mtsmorf.move_exp.cv import fit_classifiers_cv
from mtsmorf.io.move.read import read_move_trial_epochs
from mtsmorf.move_exp.spectral import compute_freq_band_power

from utils import NumpyEncoder


def _prepare_movement_onset_data(before_data, after_data):
    """Extract data matrix X and labels y from before and after mne.Epochs
    or mne.EpochsTFR data structures.
    """
    if before_data.ndim == 3:
        ntrials, nchs, nsteps = before_data.shape
        image_height = nchs
        image_width = nsteps

    elif before_data.ndim == 4:
        ntrials, nchs, nfreqs, nsteps = before_data.shape
        image_height = nchs * nfreqs
        image_width = nsteps

    else:
        raise TypeError(
            "Either before and after both must be either Epochs or EpochsTFR."
        )

    X = np.vstack(
        [
            before_data.reshape(ntrials, -1),  # class 0
            after_data.reshape(ntrials, -1),  # class 1
        ]
    )
    y = np.concatenate([np.zeros(ntrials), np.ones(ntrials)])
    assert X.shape[0] == y.shape[0], "X and y do not have the same number of trials"

    return X, y, image_height, image_width


def decode_movement(
    root,
    subject,
    destination_path,
    cv,
    metrics,
    domain,
    n_jobs=1,
    random_state=None,
):
    """Run classifier comparison in classifying before or after movement onset."""
    destination = Path(destination_path) / f"{domain}_domain"
    if os.path.exists(destination):
        print("Results folder already exists, we will not overwrite.")
        return


    before = read_move_trial_epochs(root, subject, event_key="At Center", tmin=0, tmax=1.0)
    before.load_data()
    after = read_move_trial_epochs(root, subject, event_key="Left Target", tmin=-0.25, tmax=0.75)
    after.load_data()

    resample_rate = 500
    if domain == "time":
        # Get data for before movement onset
        before = before.filter(l_freq=1, h_freq=before.info["sfreq"] / 2. - 1)
        before = before.resample(resample_rate)
        before_data = before.get_data()

        # Get data for after movement onset
        after = after.filter(l_freq=1, h_freq=after.info["sfreq"] / 2. - 1)
        after = after.resample(resample_rate)
        after_data = after.get_data()

        X, y, image_height, image_width = _prepare_movement_onset_data(
            before_data, after_data
        )
    elif domain in ["freq", "frequency"]:
        frequency_bands = dict(
            delta=(0.5, 4),
            theta=(4, 8),
            alpha=(8, 13),
            beta=(13, 30),
            gamma=(30, 70),
            hi_gamma=(70, 200),
        )
        before_data = []
        after_data = []
        for band, (l_freq, h_freq) in frequency_bands.items():
            # Get data for before movement onset    
            before_band = before.filter(l_freq=l_freq, h_freq=h_freq).apply_hilbert(envelope=True)
            before_band = before_band.resample(resample_rate)
            before_data.append(before_band.get_data())

            # Get data for after movement onset
            after_band = after.filter(l_freq=l_freq, h_freq=h_freq).apply_hilbert(envelope=True)
            after_band = after_band.resample(resample_rate)
            after_data.append(after_band.get_data())
    
        # Transpose to have data shape (ntrials, nchs, nfreqs, nsteps)
        before_data = np.array(before_data).transpose(1,2,0,3)
        after_data = np.array(after_data).transpose(1,2,0,3)

        X, y, image_height, image_width = _prepare_movement_onset_data(
            before_data, after_data
        )
    else:
        raise ValueError("Domain should be 'time' of 'frequency'.")

    # Perform K-Fold cross validation
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

    # Run feat importance for roc_auc_ovr
    try:
        n_repeats = 5  # number of repeats for permutation importance

        scoring_methods = [
            "roc_auc_ovr",
        ]
        for scoring_method in scoring_methods:
            key_mean = f"validate_{scoring_method}_imp_mean"
            if key_mean not in scores:
                scores[key_mean] = []

            key_std = f"validate_{scoring_method}_imp_std"
            if key_std not in scores:
                scores[key_std] = []

            print(f"{subject.upper()}: Running feature importances...")
            result = permutation_importance(
                best_estimator,
                X_test,
                y_test,
                scoring=scoring_method,
                n_repeats=n_repeats,
                n_jobs=1,
                random_state=random_state,
            )

            imp_std = result.importances_std
            imp_vals = result.importances_mean
            scores[key_mean].append(list(imp_vals))
            scores[key_std].append(list(imp_std))

        cv_scores[clf_name] = scores
    except:
        print("feat importances failed...")
        traceback.print_exc()

    if not os.path.exists(destination):
            os.makedirs(destination)

    for clf_name, clf_scores in cv_scores.items():

        estimator = clf_scores["estimator"]
        if estimator is not None:
            del clf_scores["estimator"]

        with open(destination / f"{subject}_{clf_name}_results.json", "w") as fout:
            json.dump(clf_scores, fout, cls=NumpyEncoder)
            print(f"{subject.upper()} CV results for {clf_name} saved as json.")
        cv_scores[clf_name]["estimator"] = estimator

    ## Plot results
    fig, axs = plt.subplots(ncols=2, dpi=100, figsize=(16, 6))
    axs = axs.flatten()
    plot_classifier_performance(cv_scores, X, y, axs=axs)
    axs[0].set(
        title=f"{subject.upper()} ROC Curves for 'At Center' vs. 'Left Target' ({domain.capitalize()} Domain)",
    )
    axs[1].set(
        title=f"{subject.upper()}: Accuracies for 'At Center' vs. 'Left Target' ({domain.capitalize()} Domain)",
    )
    fig.tight_layout()

    plt.savefig(destination / f"movement_onset_{domain}_domain.png")
    plt.close(fig)
    print(
        f"Figure saved at {destination}/movement_onset_{domain}_domain.png"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=str, help="subject ID (e.g. efri02)")
    args = parser.parse_args()
    subject = args.subject

    with open(Path(os.path.dirname(__file__)) / "config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    bids_root = Path(config["bids_root"])
    results_path = Path(config["results_path"])

    seed = 1
    n_splits = 5
    cv = StratifiedKFold(n_splits)
    metrics = dict(
        accuracy="accuracy",
        cohen_kappa_score=make_scorer(cohen_kappa_score),
        roc_auc_ovr="roc_auc_ovr",
    )

    destination_path = results_path / "decode_movement" / subject
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    decode_movement(
        bids_root, subject, destination_path, cv, metrics, "time", n_jobs=1, random_state=seed
    )
    decode_movement(
        bids_root, subject, destination_path, cv, metrics, "freq", n_jobs=1, random_state=seed
    )
