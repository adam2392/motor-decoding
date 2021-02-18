import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from mne_bids.path import BIDSPath
from mne import Epochs
from mne.time_frequency import AverageTFR, EpochsTFR
from mne.time_frequency.tfr import tfr_morlet
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold

from cv import cv_fit, fit_classifiers_cv
from move_experiment_functions import get_event_data
from plotting import plot_classifier_performance


def _prepare_onset_data(before, after):
    """Extract data matrix X and labels y from before and after mne.Epochs
    data structures.
    """
    before.load_data()
    before_data = before.get_data()

    after.load_data()
    after_data = after.get_data()

    ntrials = len(before)
    X = np.vstack(
        [
            before_data.reshape(ntrials, -1),  # class 0
            after_data.reshape(ntrials, -1),  # class 1
        ]
    )
    y = np.concatenate([np.zeros(ntrials), np.ones(ntrials)])
    assert X.shape[0] == y.shape[0], "X and y do not have the same number of epochs"

    if isinstance(before, Epochs) and isinstance(after, Epochs):
        ## Time Domain
        ntrials, nchs, nsteps = before_data.shape
        image_height = nchs
        image_width = nsteps

    elif isinstance(before, EpochsTFR) and isinstance(after, EpochsTFR):
        ## Freq Domain
        ntrials, nchs, nfreqs, nsteps = before_data.shape
        image_height = nchs * nfreqs
        image_width = nsteps

    else:
        raise TypeError("Either before or after are not Epochs or EpochsTFR.")

    return X, y, image_height, image_width


def movement_onset_experiment(
    bids_path,
    destination_path,
    cv,
    metrics,
    domain,
    random_state=None,
):
    """Run classifier comparison in classifying before or after movement onset."""
    subject = bids_path.subject
    destination_path = Path(destination_path)

    before, _ = get_event_data(bids_path, tmin=0, tmax=1.0, event_key="At Center")
    before.load_data()

    after, _ = get_event_data(bids_path, tmin=-0.25, tmax=0.75, event_key="Left Target")
    after.load_data()

    if domain == "time":
        X, y, image_height, image_width = _prepare_onset_data(before, after, domain)
    elif domain in ["freq", "frequency"]:
        nfreqs = 10
        lfreq, hfreq = (70, 200)
        freqs = np.logspace(*np.log10([lfreq, hfreq]), num=nfreqs)
        n_cycles = freqs / 3.0  # different number of cycle per frequency

        before_power = tfr_morlet(
            before,
            freqs=freqs,
            n_cycles=n_cycles,
            average=False,
            return_itc=False,
            decim=3,
            n_jobs=1,
        )
        after_power = tfr_morlet(
            after,
            freqs=freqs,
            n_cycles=n_cycles,
            average=False,
            return_itc=False,
            decim=3,
            n_jobs=1,
        )
        X, y, image_height, image_width = _prepare_onset_data(
            before_power, after_power, domain
        )
    else:
        raise ValueError("Domain should be time of frequency.")

    # Perform K-Fold cross validation
    clf_scores = fit_classifiers_cv(
        X, y, image_height, image_width, cv, metrics, n_jobs=-1, random_state=seed
    )

    ## Plot results
    # 1. Plot roc curves
    fig, axs = plt.subplots(ncols=2, dpi=100, figsize=(16, 6))
    axs = axs.flatten()
    axs = plot_classifier_performance(clf_scores, X, y, axs=axs)

    axs[0].set(
        title=f"{subject.upper()} ROC Curves for 'At Center' vs. 'Left Target' ({domain.capitalize()} Domain)",
    )

    # 2. Plot accuracies
    axs[1].set(
        title=f"{subject.upper()}: Accuracies for 'At Center' vs. 'Left Target' ({domain.capitalize()} Domain)",
    )
    fig.tight_layout()

    plt.savefig(destination_path / f"movement_onset_{domain}_domain.png")
    plt.close(fig)
    print(f"Figure saved at {destination_path}/movement_onset_{domain}_domain.png")


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

    destination_path = results_path / subject / "movement_onset_experiment"
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    movement_onset_experiment(
        bids_path, destination_path, cv, metrics, random_state=seed
    )
    movement_onset_experiment(
        bids_path, destination_path, cv, metrics, random_state=seed
    )
