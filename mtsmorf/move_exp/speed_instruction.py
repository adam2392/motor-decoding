import argparse
import json
import os
import sys
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mne_bids import BIDSPath, read_raw_bids
from rerf.rerfClassifier import rerfClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    make_scorer,
    roc_auc_score,
)

if Path(__file__).parents[2] not in sys.path:
    sys.path.append(str(Path(__file__).parents[2]))

from mtsmorf.io.move.read import read_move_trial_epochs, read_trial_metadata
from mtsmorf.io.utils import NumpyEncoder
from mtsmorf.move_exp.cv import fit_classifiers_cv
from mtsmorf.move_exp.plotting import plot_accuracies, plot_classifier_performance, plot_roc_aucs


def speed_instruction_experiment(
    root, subject, destination_path, cv, metrics, domain, n_jobs=1, random_state=None
):
    destination = Path(destination_path) / f"{domain}_domain/"
    if os.path.exists(destination):
        print(f"Results folder already exists for {domain} domain...terminating")
        return

    trials = read_trial_metadata(root, subject)
    trials = pd.DataFrame(trials)
    epochs = read_move_trial_epochs(root=root, subject=subject)

    # Drop bad trials
    trials = trials[~(trials.perturbed) & (trials.success)].reset_index(drop=True)
    
    # Stratify data by speed instruction
    slow_trials = trials[trials.speed_instruction.str.startswith("slow")]
    fast_trials = trials[trials.speed_instruction.str.startswith("fast")]

    resample_rate = 500
    if domain.lower() == "time":
        epochs = epochs.filter(l_freq=1, h_freq=epochs.info["sfreq"] / 2.0 - 1)
        epochs = epochs.resample(resample_rate)
        data = epochs.get_data()
        
        ntrials, nchs, nsteps = data.shape
        image_height = nchs
        image_width = nsteps

    elif domain.lower() in ["freq", "frequency"]:
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

    slow_inds = slow_trials.index
    fast_inds = fast_trials.index
    Xslow = data[slow_inds]
    Xfast = data[fast_inds]
    yslow = slow_trials.target_direction.values
    yfast = fast_trials.target_direction.values

    if not os.path.exists(destination):
        os.makedirs(destination)

    fig, axs = plt.subplots(1, 2)
    axs = axs.flatten()

    label_names = {0: "Down", 1: "Right", 2: "Up", 3: "Left"}
    axs[0].hist(yslow, bins=4, alpha=0.4, color="b")
    axs[0].set_xticks(np.arange(4))
    axs[0].set_xticklabels([label_names[i] for i in np.arange(4)])
    axs[0].set_title("slow trials")
    axs[1].hist(yfast, bins=4, alpha=0.4, color="r")
    axs[1].set_xticks(np.arange(4))
    axs[1].set_xticklabels([label_names[i] for i in np.arange(4)])
    axs[1].set_title("fast trials")
    fig.tight_layout()
    plt.savefig(destination / "data_distribution.png")

    stratified_data = dict(slow=(Xslow, yslow), fast=(Xfast, yfast))
    for trial_type, (X, y) in stratified_data.items():
        X = X.reshape(X.shape[0], -1)  # Vectorize

        # Perform cross validation
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
        fig, axs = plt.subplots(ncols=2, dpi=100, figsize=(16, 6))
        axs = axs.flatten()
        plot_classifier_performance(cv_scores, X, y, axs=axs)
        axs[0].set(
            title=f"{subject.upper()} ROC Curves for {trial_type} trials ({domain.capitalize()} Domain)",
        )
        axs[1].set(
            title=f"{subject.upper()}: Accuracies for {trial_type} trials ({domain.capitalize()} Domain)",
        )
        fig.tight_layout()

        plt.savefig(destination / f"speed_instruction{trial_type}_{domain}_domain.png")
        plt.close(fig)
        print(f"Figure saved at {destination}/speed_instruction{trial_type}_{domain}_domain.png")


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

    destination_path = results_path / "speed_instruction" / subject
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    speed_instruction_experiment(
        bids_root, subject, destination_path, cv, metrics, "time", random_state=seed
    )

    speed_instruction_experiment(
        bids_root, subject, destination_path, cv, metrics, "freq", random_state=seed
    )
