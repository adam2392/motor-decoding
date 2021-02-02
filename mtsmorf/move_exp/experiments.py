import argparse
import os
import sys
import traceback
from pathlib import Path

import dabest
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
from mne_bids.path import BIDSPath
from mne_bids.tsv_handler import _from_tsv
from mne.time_frequency.tfr import tfr_morlet
from rerf.rerfClassifier import rerfClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, make_scorer, roc_curve
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.utils import check_random_state

from cv import cv_roc, cv_fit
from move_experiment_functions import (
    get_event_data,
    get_event_durations,
    initialize_classifiers,
    fit_classifiers_cv,
)
from plotting import (
    plot_roc_cv,
    plot_roc_multiclass_cv,
    plot_accuracies,
    plot_roc_aucs,
    plot_event_durations,
    plot_event_onsets,
    plot_classifier_performance,
)

# Hack-y way to import from files in sibling "io" directory
sys.path.append(str(Path(__file__).parent.parent / "io"))
sys.path.append(str(Path(__file__).parent.parent / "war_exp"))

from read import read_dataset, read_label, read_trial, get_trial_info, _get_bad_chs
from utils import NumpyEncoder
import json
from sklearn.inspection import permutation_importance


def run_classifier_comparison(
    epochs,
    labels,
    cv,
    freq_domain=False,
    shuffle_channels=False,
    avg_freq=False,
    nfreqs=10,
    lfreq=70,
    hfreq=200,
    random_state=None,
):

    seed = check_random_state(random_state)

    if freq_domain:
        freqs = np.logspace(*np.log10([lfreq, hfreq]), num=nfreqs)
        n_cycles = freqs / 3.0  # different number of cycle per frequency
        power = tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            average=False,
            return_itc=False,
            decim=3,
            n_jobs=1,
        )
        times = np.where((power.times >= -0.5) & (power.times <= 1.0))[0]

        if avg_freq:
            # Trim time window
            avg_power = np.mean(power, axis=2)
            data = avg_power[:, :, times]
            image_height = data.shape[1]
            image_width = data.shape[2]

        else:
            # Trim time window
            data = power.data[:, :, :, times]

            # Parameters for mtsmorf
            image_height = data.shape[1] * data.shape[2]
            image_width = data.shape[3]

    else:
        # Trim time window
        times = np.where((epochs.times >= -0.5) & (epochs.times <= 1.0))[0]
        data = epochs.get_data()[:, :, times]

        # Parameters for mtsmorf
        image_height = data.shape[1]
        image_width = data.shape[2]

    if shuffle_channels:

        if freq_domain and not avg_freq:
            _, nchs, nfreqs, _ = data.shape

            ch_inds = seed.permutation(nchs)
            freq_inds = seed.permutation(nfreqs)

            # Need to do this to shuffle two axes simultaneously
            data = data[:, ch_inds, :, :][:, :, freq_inds, :]

        else:
            _, nchs, _ = data.shape
            ch_inds = seed.permutation(nchs)
            data = data[:, ch_inds, :]

    included_trials = np.isin(labels, [0, 1, 2, 3])

    # Create X, y data
    X = data[included_trials].reshape(np.sum(included_trials), -1)
    y = labels[included_trials]

    clf_scores = fit_classifiers_cv(
        X, y, image_height, image_width, cv, metrics, random_state=random_state
    )

    return clf_scores


def shuffle_channels_experiment(
    bids_path,
    cv,
    destination_path,
    tmin=-0.2,
    tmax=0.5,
    nfreqs=10,
    lfreq=70,
    hfreq=200,
    random_state=None,
):
    """
    docstring
    """

    destination = Path(destination_path) / "shuffle_channels_experiment"

    if not os.path.exists(destination):
        os.makedirs(destination)

    epochs, labels = get_event_data(bids_path, tmin=tmin, tmax=tmax)

    ## Time domain
    clf_scores_unshuffled = run_classifier_comparison(
        epochs,
        labels,
        cv,
        freq_domain=False,
        shuffle_channels=False,
        random_state=random_state,
    )

    clf_scores_shuffled = run_classifier_comparison(
        epochs,
        labels,
        cv,
        freq_domain=False,
        shuffle_channels=True,
        random_state=random_state,
    )

    ## Plot results
    fig, axs = plt.subplots(nrows=2, ncols=2, dpi=100, figsize=(16, 12), sharey="row")
    axs = axs.flatten()

    plot_accuracies(clf_scores_unshuffled, ax=axs[0])
    axs[0].set(
        ylabel="accuracy",
        title=f"{subject.upper()}: Accuracy of Classifiers (Time Domain Signal, Unshuffled)",
    )

    plot_accuracies(clf_scores_shuffled, ax=axs[1])
    axs[1].set(
        title=f"{subject.upper()}: Accuracy of Classifiers (Time Domain Signal, Shuffled)"
    )

    plot_roc_aucs(clf_scores_unshuffled, ax=axs[2])
    axs[2].set(
        ylabel="ROC AUC",
        title=f"{subject.upper()}: ROC AUCs of Classifiers (Time Domain Signal, Unshuffled)",
    )

    plot_roc_aucs(clf_scores_shuffled, ax=axs[3])
    axs[3].set(
        title=f"{subject.upper()}: ROC AUCs of Classifiers (Time Domain Signal, Shuffled)"
    )
    fig.tight_layout()
    plt.savefig(destination / "time_domain_comparison.png")
    plt.close(fig)

    ## Freq Domain (No Averaging)
    # Fit models
    clf_scores_unshuffled = run_classifier_comparison(
        epochs,
        labels,
        cv,
        freq_domain=True,
        avg_freq=False,
        nfreqs=nfreqs,
        lfreq=lfreq,
        hfreq=hfreq,
        random_state=random_state,
    )

    clf_scores_shuffled = run_classifier_comparison(
        epochs,
        labels,
        cv,
        freq_domain=True,
        shuffle_channels=True,
        avg_freq=False,
        nfreqs=nfreqs,
        lfreq=lfreq,
        hfreq=hfreq,
        random_state=random_state,
    )

    ## Plot results
    fig, axs = plt.subplots(nrows=2, ncols=2, dpi=100, figsize=(16, 12), sharey="row")
    axs = axs.flatten()

    plot_accuracies(clf_scores_unshuffled, ax=axs[0])
    axs[0].set(
        ylabel="accuracy",
        title=f"{subject.upper()}: Accuracy of Classifiers (n={nfreqs} freq bins, No Averaging, {lfreq}-{hfreq} Hz, Unshuffled)",
    )

    plot_accuracies(clf_scores_unshuffled, ax=axs[1])
    axs[1].set(
        title=f"{subject.upper()}: Accuracy of Classifiers (n={nfreqs} freq bins, No Averaging, {lfreq}-{hfreq} Hz, Shuffled)"
    )

    plot_roc_aucs(clf_scores_unshuffled, ax=axs[2])
    axs[2].set(
        ylabel="ROC AUC",
        title=f"{subject.upper()}: ROC AUCs of Classifiers (n={nfreqs} freq bins, No Averaging, {lfreq}-{hfreq} Hz, Unshuffled)",
    )

    plot_roc_aucs(clf_scores_shuffled, ax=axs[3])
    axs[3].set(
        title=f"{subject.upper()}: ROC AUCs of Classifiers (n={nfreqs} freq bins, No Averaging, {lfreq}-{hfreq} Hz, Shuffled)"
    )
    fig.tight_layout()
    plt.savefig(destination / "freq_domain_no_averaging_comparison.png")
    plt.close(fig)


def plot_paired_cvs_baseline(scores1, scores2, axs):
    id_col = pd.Series(range(1, n_splits + 1))
    axs = axs.flatten()

    ## Accuracy Comparison
    df = pd.DataFrame(
        {
            "ID": id_col,
            "No Baseline": scores1["test_accuracy"],
            "Baseline": scores2["test_accuracy"],
        }
    )

    my_data = dabest.load(
        df,
        idx=("No Baseline", "Baseline"),
        id_col="ID",
        paired=True,
        resamples=100,
        random_seed=seed,
    )

    my_data.mean_diff.plot(ax=axs[0])
    axs[0].set(title="Accuracy Comparison in Baseline Correction of Time Domain Signal")

    ## ROC AUC Comparison
    df = pd.DataFrame(
        {
            "ID": id_col,
            "No Baseline": scores1["test_roc_auc_ovr"],
            "Baseline": scores2["test_roc_auc_ovr"],
        }
    )

    my_data = dabest.load(
        df,
        idx=("No Baseline", "Baseline"),
        id_col="ID",
        paired=True,
        resamples=100,
        random_seed=seed,
    )

    my_data.mean_diff.plot(ax=axs[1])
    axs[1].set(title="ROC AUC Comparison in Baseline Correction of Time Domain Signal")


def baseline_experiment(bids_path, destination_path, cv, metrics, random_state=None):
    """
    docstring
    """
    destination = Path(destination_path) / "baseline_experiment"

    if not os.path.exists(destination):
        os.makedirs(destination)

    epochs, labels = get_event_data(bids_path, tmin=-0.75, tmax=0.5)
    epochs.load_data()

    cropped = epochs.copy()
    cropped = cropped.crop(tmin=-0.3, tmax=0.3)

    cropped_data = cropped.get_data()
    ntrials, nchs, nsteps = cropped_data.shape

    # Create X, y data
    included_trials = np.isin(labels, [0, 1, 2, 3])

    X = cropped_data[included_trials].reshape(np.sum(included_trials), -1)
    y = labels[included_trials]

    mtsmorf = rerfClassifier(
        projection_matrix="MT-MORF",
        max_features="auto",
        n_jobs=-1,
        random_state=seed,
        image_height=nchs,
        image_width=nsteps,
    )

    stratified_kf_scores = cv_fit(
        mtsmorf,
        X,
        y,
        metrics=metrics,
        cv=cv,
        n_jobs=None,
        return_train_score=True,
        return_estimator=True,
    )

    baseline, _ = get_event_data(bids_path, tmin=0, tmax=0.5, event_key="At Center")
    baseline.load_data()

    # Subtract from epochs data
    epochs_data = epochs.get_data()
    baseline_data = baseline.get_data()

    baseline_avg = np.mean(baseline_data, axis=0)  # Compute mean signal for baseline
    baselined_epochs = epochs_data - baseline_avg  # Subtract mean signal to baseline

    ## Crop
    times = epochs.times
    # inds = np.where((times >= -0.5) & (times <= 1.0))[0]
    inds = np.where((times >= -0.3) & (times <= 0.3))[0]
    baselined_epochs = baselined_epochs[:, :, inds]

    # Create X, y data
    X = baselined_epochs[included_trials].reshape(np.sum(included_trials), -1)
    y = labels[included_trials]

    ## Fit model
    mtsmorf = rerfClassifier(
        projection_matrix="MT-MORF",
        max_features="auto",
        n_jobs=-1,
        random_state=seed,
        image_height=baselined_epochs.shape[1],
        image_width=baselined_epochs.shape[2],
    )
    stratified_kf_scores_baselined = cv_fit(
        mtsmorf,
        X,
        y,
        metrics=metrics,
        cv=cv,
        n_jobs=None,
        return_train_score=True,
        return_estimator=True,
    )

    fig, axs = plt.subplots(ncols=2, figsize=(22, 6), dpi=200)
    try:
        plot_paired_cvs_baseline(
            stratified_kf_scores, stratified_kf_scores_baselined, axs=axs
        )
        axs[0].set(
            title="Accuracy Comparison in Baseline Correction of Time Domain Signal"
        )
        axs[1].set(
            title="ROC AUC Comparison in Baseline Correction of Time Domain Signal"
        )
        plt.savefig(destination / "baseline_experiment_time_domain.png")
    except ValueError as e:
        traceback.print_exc()
    plt.close(fig)

    ## Freq Domain
    nfreqs = 10
    freqs = np.logspace(*np.log10([70, 200]), num=nfreqs)
    n_cycles = freqs / 3.0  # different number of cycle per frequency
    power = tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        average=False,
        return_itc=False,
        decim=3,
        n_jobs=1,
    )

    hi_gamma = power.data

    avg_freq_data = np.mean(hi_gamma, axis=2)

    # Trim time window
    # inds = np.where((power.times >= -0.5) & (power.times <= 1.0))[0]
    inds = np.where((power.times >= -0.3) & (power.times <= 0.3))[0]
    avg_freq_data = avg_freq_data[:, :, inds]

    # Create X, y data
    X = avg_freq_data[included_trials].reshape(np.sum(included_trials), -1)
    y = labels[included_trials]

    # Results for averaging frequency
    stratified_kf_scores_freq_avg = cv_fit(
        mtsmorf,
        X,
        y,
        metrics=metrics,
        cv=cv,
        n_jobs=None,
        return_train_score=True,
        return_estimator=True,
    )

    baseline_power = tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        average=False,
        return_itc=False,
        decim=3,
        n_jobs=1,
    )

    ## Apply zscore baselining
    power_data = power.data
    baseline_power_data = baseline_power.data

    baseline_power_avg = np.mean(baseline_power_data, axis=0)
    baseline_power_std = np.std(baseline_power_data, axis=0)

    baselined_power = (power_data - baseline_power_avg) / baseline_power_std

    ## Crop
    times = power.times
    inds = np.where((times >= -0.3) & (times <= 0.3))[0]
    baselined_power = baselined_power[:, :, :, inds]

    avg_freq_data_baseline = np.mean(baselined_power, axis=2)

    included_trials = np.isin(labels, [0, 1, 2, 3])

    ## Create X, y data
    X = avg_freq_data_baseline[included_trials].reshape(np.sum(included_trials), -1)
    y = labels[included_trials]

    ## Fit model
    mtsmorf = rerfClassifier(
        projection_matrix="MT-MORF",
        max_features="auto",
        n_jobs=-1,
        random_state=random_state,
        image_height=avg_freq_data_baseline.shape[1],
        image_width=avg_freq_data_baseline.shape[2],
    )
    stratified_kf_scores_baselined = cv_fit(
        mtsmorf,
        X,
        y,
        metrics=metrics,
        cv=cv,
        n_jobs=None,
        return_train_score=True,
        return_estimator=True,
    )

    fig, axs = plt.subplots(ncols=2, figsize=(22, 6), dpi=200)
    try:
        plot_paired_cvs_baseline(
            stratified_kf_scores_freq_avg, stratified_kf_scores_baselined, axs=axs
        )
        axs[0].set(
            title="Accuracy Comparison in Baseline Correction of Freq Domain Signal"
        )
        axs[1].set(
            title="ROC AUC Comparison in Baseline Correction of Freq Domain Signal"
        )
        plt.savefig(destination / "baseline_experiment_freq_domain_averaged.png")
    except ValueError as e:
        traceback.print_exc()
    plt.close(fig)


def frequency_band_comparison(
    epochs, destination_path, cv, metrics, nfreqs=10, random_state=None
):
    """
    docstring
    """
    destination = Path(destination_path)

    rng = check_random_state(random_state)
    seed = rng.randint(sys.maxint)

    if not os.path.exists(destination):
        os.makedirs(destination)

    frequency_bands = dict(
        delta=(0.5, 4),
        theta=(4, 8),
        alpha=(8, 13),
        beta=(13, 30),
        gamma=(30, 70),
        hi_gamma=(70, 200),
    )

    scores = dict()

    for name, (lfreq, hfreq) in frequency_bands.items():
        freqs = np.logspace(*np.log10([lfreq, hfreq]), num=nfreqs)
        n_cycles = freqs / 2.0  # different number of cycle per frequency
        power = tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            average=False,
            return_itc=False,
            decim=3,
            n_jobs=1,
        )

        # Extract data and crop
        inds = np.where((power.times >= -0.3) & (power.times <= 0.3))[0]
        power_data = power.data[:, :, :, inds]
        ntrials, nchs, nfreqs, nsteps = power_data.shape

        included_trials = np.isin(labels, [0, 1, 2, 3])

        # Create X, y data
        X = power_data[included_trials].reshape(np.sum(included_trials), -1)
        y = labels[included_trials]

        mtsmorf = rerfClassifier(
            projection_matrix="MT-MORF",
            max_features="auto",
            n_jobs=-1,
            random_state=random_state,
            image_height=nchs * nfreqs,
            image_width=nsteps,
        )

        scores[name] = cv_fit(
            mtsmorf,
            X,
            y,
            metrics=metrics,
            cv=cv,
            n_jobs=None,
            return_train_score=True,
            return_estimator=True,
        )

    fig, axs = plt.subplots(ncols=2, figsize=(22, 6), dpi=100)
    axs = axs.flatten()

    ## Accuracy comparison
    id_col = pd.Series(range(1, n_splits + 1))
    accuracies = {name: score["test_accuracy"] for name, score in scores.items()}
    accuracies["ID"] = id_col

    df = pd.DataFrame(accuracies)

    # Re-order so that control is hi-gamma band
    idx = [list(scores.keys())[-1]] + list(scores.keys())[:-1]
    my_data = dabest.load(df, idx=idx, resamples=100, random_seed=seed)
    my_data.mean_diff.plot(ax=axs[0])
    axs[0].set(title=f"{subject.upper()} Accuracy Comparison between Frequency Bands")

    ## ROC AUC comparison
    roc_auc_ovrs = {name: score["test_roc_auc_ovr"] for name, score in scores.items()}
    roc_auc_ovrs["ID"] = id_col
    df = pd.DataFrame(roc_auc_ovrs)
    my_data = dabest.load(df, idx=idx, resamples=100, random_seed=seed)
    my_data.mean_diff.plot(ax=axs[1])
    axs[1].set(title=f"{subject.upper()} ROC AUC Comparison between Frequency Bands")

    fig.tight_layout()
    plt.savefig(
        destination / f"{subject}_frequency_band_comparison_tmin=-0.5_tmax=1.0.png"
    )
    plt.close(fig)


def time_window_experiment(
    bids_path,
    destination_path,
    domain,
    cv,
    metrics,
    freqs=None,
    n_cycles=None,
    random_state=None,
):
    if domain.lower() in ["frequency", "freq"] and (freqs is None or n_cycles is None):
        raise TypeError("freqs and n_cycles must not be None to run frequency domain")

    subject = bids_path.subject

    destination = (
        Path(destination_path) / subject / f"trial_specific_window/{domain}_domain/"
    )
    if not os.path.exists(destination_path):
        os.makedirs(destination)

    go_cue_durations = get_event_durations(
        bids_path, event_key="Left Target", periods=-1
    )
    left_target_durations = get_event_durations(
        bids_path, event_key="Left Target", periods=1
    )

    tmin = -max(go_cue_durations)
    tmax = max(left_target_durations)

    epochs, labels = get_event_data(bids_path, tmin=tmin - 0.2, tmax=tmax + 0.2)

    if domain.lower() in ["frequency", "freq"]:
        power = tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            average=False,
            return_itc=False,
            decim=3,
            n_jobs=-1,
        )
        data = power.data
        ntrials, nchs, nfreqs, nsteps = data.shape
        print(f"{subject.upper()}: data.shape = ({data.shape})")

        t = power.times
        mask = (t >= -np.asarray(go_cue_durations)[:, None, None, None]) & (
            t <= np.asarray(left_target_durations)[:, None, None, None]
        )
        masked_data = data * mask

        image_height = nchs * nfreqs
        image_width = nsteps

    elif domain.lower() == "time":
        data = epochs.get_data()
        ntrials, nchs, nsteps = data.shape
        print(f"{subject.upper()}: data.shape = ({data.shape})")

        t = epochs.times
        mask = (t >= -np.asarray(go_cue_durations)[:, None, None]) & (
            t <= np.asarray(left_target_durations)[:, None, None]
        )
        masked_data = data * mask

        image_height = nchs
        image_width = nsteps

    else:
        raise ValueError('domain must be one of "time", "freq", or "frequency".')

    X = masked_data.reshape(ntrials, -1)
    y = labels

    cv_scores = fit_classifiers_cv(
        X, y, image_height, image_width, cv, metrics, random_state=random_state
    )

    n_repeats = 5  # number of repeats for permutation importance

    clf_name = "MT-MORF"
    scores = cv_scores[clf_name]
    best_ind = np.argmax(scores["test_roc_auc_ovr"])
    best_estimator = scores["estimator"][best_ind]
    best_test_inds = scores["test_inds"][best_ind]

    X_test = X[best_test_inds]
    y_test = y[best_test_inds]

    # Run feat importance for roc_auc_ovr
    print(f"{subject.upper()}: Running feature importances...")
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

    for clf_name, clf_scores in cv_scores.items():

        estimator = clf_scores["estimator"]
        if estimator is not None:
            del clf_scores["estimator"]

        with open(destination / f"{subject}_{clf_name}_results.json", "w") as fout:
            json.dump(clf_scores, fout, cls=NumpyEncoder)
            print(f"{subject.upper()} CV results for {clf_name} saved as json.")
        clf_scores["estimator"] = estimator

    fig, axs = plt.subplots(nrows=2, ncols=3, dpi=100, figsize=(24, 12))
    axs = axs.flatten()
    for i, (clf_name, scores) in enumerate(cv_scores.items()):
        ax = axs[i]

        plot_roc_multiclass_cv(
            scores["test_predict_proba"],
            X,
            y,
            scores["test_inds"],
            ax=ax,
        )

        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=f"{subject.upper()} {clf_name} One vs. Rest ROC Curves",
        )
        ax.legend(loc="lower right")

    plot_roc_aucs(clf_scores, ax=axs[-1])
    axs[-1].set(
        ylabel="ROC AUC",
        title=f"{subject.upper()}: ROC AUCs for Trial-Specific Time Window",
    )
    fig.tight_layout()
    plt.savefig(destination / f"{subject}_trial_specific_time_window_rocs.png")
    plt.close(fig)
    print(
        f"Figure saved at {destination}/{subject}_trial_specific_time_window_rocs.png"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=str, help="subject ID (e.g. efri02)")
    parser.add_argument(
        "-experiment",
        type=str,
        choices=[
            "shuffle",
            "baseline",
            "frequency_bands",
            "trial_specific_time_window_time",
            "trial_specific_time_window_freq",
            "plot_event_durations",
            "plot_event_onsets",
        ],
        help="which experiment to run",
    )

    args = parser.parse_args()
    subject = args.subject
    experiment = args.experiment

    bids_root = Path("/workspaces/research/mnt/data/efri/")
    results_path = Path(
        "/workspaces/research/efri OneDrive/Adam Li - efri/derivatives/workstation_output"
    )

    # path identifiers
    path_identifiers = dict(
        subject=subject,
        session="efri",
        task="move",
        acquisition="seeg",
        run="01",
        suffix="ieeg",
        extension=".vhdr",
        root=bids_root,
    )
    bids_path = BIDSPath(**path_identifiers)

    # Prep data for model fitting
    epochs, labels = get_event_data(bids_path, tmin=tmin, tmax=tmax)

    if not os.path.exists(results_path / subject):
        try:
            os.makedirs(results_path / subject)
        except FileExistsError as e:
            print(
                f"Tried making results directory for {subject}, but file already exists."
            )
        except Exception as e:
            print(
                f"Tried making results directory for {subject}, but an error occurred:"
            )
            traceback.print_exc()

    seed = 1
    n_splits = 5
    tmin, tmax = (-0.75, 1.25)
    cv = StratifiedKFold(n_splits)
    metrics = dict(
        accuracy="accuracy",
        cohen_kappa_score=make_scorer(cohen_kappa_score),
        roc_auc_ovr="roc_auc_ovr",
    )

    if experiment == "shuffle":
        shuffle_channels_experiment(
            epochs,
            labels,
            cv,
            metrics,
            results_path / subject,
            tmin=tmin,
            tmax=tmax,
            nfreqs=10,
            lfreq=70,
            hfreq=200,
            random_state=seed,
        )

    elif experiment == "baseline":
        baseline_experiment(
            bids_path,
            results_path / subject,
            cv,
            metrics,
            random_state=seed,
        )

    elif experiment == "frequency_bands":
        epochs.crop(tmin=-0.5, tmax=1.0)
        frequency_band_comparison(
            epochs, results_path / subject, cv, metrics, random_state=seed
        )

    elif experiment == "trial_specific_time_window_time":
        nfreqs = 10
        lfreq, hfreq = (70, 200)
        freqs = np.logspace(*np.log10([lfreq, hfreq]), num=nfreqs)
        n_cycles = freqs / 3.0

        time_window_experiment(
            bids_path,
            results_path / subject,
            "time",
            cv,
            metrics,
            random_state=seed,
        )

    elif experiment == "trial_specific_time_window_freq":
        nfreqs = 10
        lfreq, hfreq = (70, 200)
        freqs = np.logspace(*np.log10([lfreq, hfreq]), num=nfreqs)
        n_cycles = freqs / 3.0

        time_window_experiment(
            bids_path,
            results_path / subject,
            "freq",
            cv,
            metrics,
            freqs=freqs,
            n_cycles=n_cycles,
            random_state=seed,
        )

    elif experiment == "plot_event_durations":
        fig, ax = plt.subplots(dpi=150, figsize=(8, 6))

        behav, events = map(pd.DataFrame, get_trial_info(bids_path))
        plot_event_durations(behav, events, ax=ax, random_state=seed)

        ax.set(ylabel="duration (s)", title=f"{subject.upper()}: Duration of Events")
        fig.tight_layout()
        plt.savefig(results_path / subject / f"{subject}_event_durations.png")

    elif experiment == "plot_event_onsets":
        fig, ax = plt.subplots(dpi=150, figsize=(8, 6))

        behav, events = map(pd.DataFrame, get_trial_info(bids_path))
        plot_event_onsets(behav, events, ax=ax, random_state=seed)

        ax.set(
            ylabel='Onset Relative to "Go Cue" (s)',
            title=f"{subject.upper()}: Onset of Events",
        )
        fig.tight_layout()
        plt.savefig(results_path / subject / f"{subject}_event_onsets.png")
