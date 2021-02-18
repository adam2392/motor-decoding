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
from functions.move_experiment_functions import get_event_data
from functions.time_window_selection_functions import (
    fit_classifiers_cv,
    get_event_durations,
    plot_event_durations,
    plot_event_onsets,
)
from plotting import (
    plot_roc_multiclass_cv,
    plot_accuracies,
    plot_roc_aucs,
    plot_classifier_performance,
)

# Hack-y way to import from files in sibling "io" directory
sys.path.append(str(Path(__file__).parent.parent / "io"))
sys.path.append(str(Path(__file__).parent.parent / "war_exp"))

from read import read_dataset, read_label, read_trial, get_trial_info, _get_bad_chs
from utils import NumpyEncoder
import json
from sklearn.inspection import permutation_importance
import yaml


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

    destination = Path(destination_path) / f"trial_specific_window/{domain}_domain/"
    if not os.path.exists(destination):
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
        power.crop(tmin=tmin, tmax=tmax)
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
        epochs.crop(tmin=tmin, tmax=tmax)
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
        X,
        y,
        image_height,
        image_width,
        cv,
        metrics,
        n_jobs=-1,
        random_state=random_state,
    )

    n_repeats = 5  # number of repeats for permutation importance

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

            mtsmorf = rerfClassifier(
                projection_matrix="MT-MORF",
                max_features="auto",
                n_jobs=-1,
                random_state=random_state,
                image_height=image_height,
                image_width=image_width,
            )

            mtsmorf.fit(X_test, y_test)  # For some reason need to call this?
            
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

    plot_roc_aucs(cv_scores, ax=axs[-1])
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

    with open(Path(os.path.abspath(__file__)).parent / "config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    bids_root = Path(config["bids_root"])
    results_path = Path(config["results_path"])

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
        epochs, labels = get_event_data(bids_path, tmin=tmin, tmax=tmax)
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
        epochs, labels = get_event_data(bids_path, tmin=tmin, tmax=tmax)
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
