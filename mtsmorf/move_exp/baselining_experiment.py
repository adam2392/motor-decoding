
import argparse
import os
import sys
import traceback
import yaml
from pathlib import Path

import dabest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne_bids.path import BIDSPath
from mne import Epochs
from mne.time_frequency.tfr import EpochsTFR, tfr_morlet
from rerf.rerfClassifier import rerfClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold

src = Path(__file__).parents[2]
if src not in sys.path:
    sys.path.append(str(src))
from mtsmorf.move_exp.cv import cv_fit
from mtsmorf.move_exp.functions.move_experiment_functions import get_event_data


def plot_paired_cvs_baseline(scores1, scores2, axs, random_seed=1):
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
        random_seed=random_seed,
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
        random_seed=random_seed,
    )

    my_data.mean_diff.plot(ax=axs[1])
    axs[1].set(title="ROC AUC Comparison in Baseline Correction of Time Domain Signal")


# def _prepare_baseline_data(inst, labels, baseline):

#     data = inst.get_data()

#     if isinstance(inst, Epochs):
#         ntrials, nchs, nsteps = data.shape
    
#     elif isinstance(inst, EpochsTFR):
#         ntrials, nchs, nfreqs, nsteps = data.shape

#     else:
#         raise TypeError("inst is not an Epochs or EpochsTFR instance.")

#     return X, y, image_height, image_width


def baseline_experiment(bids_path, destination_path, cv, metrics, random_state=None):
    """
    docstring
    """
    destination = Path(destination_path)

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
        random_state=random_state,
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
        random_state=random_state,
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
        plt.savefig(destination / "time_domain/baseline_experiment_time_domain.png")
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
        plt.savefig(destination / "freq_domain/baseline_experiment_freq_domain_averaged.png")
    except ValueError as e:
        traceback.print_exc()
    plt.close(fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=str, help="subject ID (e.g. efri02)")
    args = parser.parse_args()
    subject = args.subject

    with open(Path(os.path.dirname(__file__)) / "config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    bids_root = Path(config["bids_root"])
    results_path = Path(config["results_path"])
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

    seed = 1
    n_splits = 5
    cv = StratifiedKFold(n_splits)
    metrics = dict(
        accuracy="accuracy",
        cohen_kappa_score=make_scorer(cohen_kappa_score),
        roc_auc_ovr="roc_auc_ovr",
    )

    destination_path = results_path / subject / "baseline_experiment"
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    baseline_experiment(
        bids_path,
        destination_path,
        cv,
        metrics,
        random_state=seed,
    )