import argparse
import os
import sys
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mne_bids.path import BIDSPath
from mne import Epochs
from mne.time_frequency import EpochsTFR
from mne.time_frequency.tfr import tfr_morlet
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold

from cv import fit_classifiers_cv
from functions.time_window_selection_functions import (
    get_event_durations,
    plot_event_durations,
    plot_event_onsets,
)
from plotting import plot_classifier_performance, plot_roc_multiclass_cv, plot_roc_aucs

if os.path.abspath(Path(__file__).parents[2]) not in sys.path:
    sys.path.append(os.path.abspath(Path(__file__).parents[2]))

from mtsmorf.io.move.read import read_move_trial_epochs, read_trial_metadata

from utils import NumpyEncoder
from rerf.rerfClassifier import rerfClassifier
import traceback
import json


def decode_directionality(
    root,
    subject,
    destination_path,
    cv,
    metrics,
    domain,
    n_jobs=1,
    random_state=None,
):
    destination = Path(destination_path) / f"tmin=-0.2_tmax=0.5/{domain}_domain/"
    if os.path.exists(destination):
        print(f"Results folder already exists for {domain} domain...terminating")
        return

    # go_cue_durations = get_event_durations(
    #     root, event_key="Left Target", periods=-1
    # )
    # left_target_durations = get_event_durations(
    #     root, event_key="Left Target", periods=1
    # )

    # tmin = -max(go_cue_durations)
    # tmax = max(left_target_durations)

    epochs = read_move_trial_epochs(root, subject, tmin=-0.2, tmax=0.5)
    trials = read_trial_metadata(root, subject)
    trials = pd.DataFrame(trials)
    labels = trials[~(trials.perturbed) & (trials.success)].target_direction.values

    resample_rate = 500
    if domain.lower() == "time":
        epochs = epochs.filter(l_freq=1, h_freq=epochs.info["sfreq"] / 2.0 - 1)
        epochs = epochs.resample(resample_rate)
        data = epochs.get_data()

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

            # mtsmorf = rerfClassifier(
            #     projection_matrix="MT-MORF",
            #     max_features="auto",
            #     n_jobs=-1,
            #     random_state=random_state,
            #     image_height=image_height,
            #     image_width=image_width,
            # )

            # mtsmorf.fit(X_test, y_test)  # For some reason need to call this?

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

    for clf_name, clf_scores in cv_scores.items():

        estimator = clf_scores.get("estimator")
        if estimator is not None:
            del clf_scores["estimator"]

        with open(destination / f"{subject}_{clf_name}_results.json", "w") as fout:
            json.dump(clf_scores, fout, cls=NumpyEncoder)
            print(f"{subject.upper()} CV results for {clf_name} saved as json.")
        
        if estimator is not None:
            clf_scores["estimator"] = estimator

    ## Plot results
    # fig, axs = plt.subplots(nrows=2, ncols=3, dpi=100, figsize=(24, 12))
    # axs = axs.flatten()
    # for i, (clf_name, scores) in enumerate(cv_scores.items()):
    #     ax = axs[i]

    #     plot_roc_multiclass_cv(
    #         scores["test_predict_proba"],
    #         X,
    #         y,
    #         scores["test_inds"],
    #         ax=ax,
    #     )

    #     ax.set(
    #         xlabel="False Positive Rate",
    #         ylabel="True Positive Rate",
    #         xlim=[-0.05, 1.05],
    #         ylim=[-0.05, 1.05],
    #         title=f"{subject.upper()} {clf_name} One vs. Rest ROC Curves",
    #     )
    #     ax.legend(loc="lower right")

    # plot_roc_aucs(cv_scores, ax=axs[-1])
    # axs[-1].set(
    #     ylabel="ROC AUC",
    #     title=f"{subject.upper()}: ROC AUCs for Trial-Specific Time Window",
    # )
    # fig.tight_layout()
    # plt.savefig(destination / f"{subject}_rocs.png")
    # plt.close(fig)
    # print(
    #     f"Figure saved at {destination}/{subject}_rocs.png"
    # )

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
    print(f"Figure saved at {destination}/movement_onset_{domain}_domain.png")


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

    destination_path = results_path / "decode_directionality" / subject
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    decode_directionality(
        bids_root, subject, destination_path, cv, metrics, "time", random_state=seed
    )
    decode_directionality(
        bids_root, subject, destination_path, cv, metrics, "freq", random_state=seed
    )
