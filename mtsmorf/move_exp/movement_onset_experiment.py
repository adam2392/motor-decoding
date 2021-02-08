import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from mne_bids.path import BIDSPath
from mne.time_frequency.tfr import tfr_morlet
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold

from cv import cv_fit, initialize_classifiers
from move_experiment_functions import get_event_data
from plotting import plot_classifier_performance


def _get_classifier_name(clf):
    """Get the classifier name based on class type."""
    if clf.__class__.__name__ == "rerfClassifier":
        clf_name = clf.get_params()["projection_matrix"]
    elif clf.__class__.__name__ == "DummyClassifier":
        clf_name = clf.strategy
    else:
        clf_name = clf.__class__.__name__

    return clf_name


def _prepare_onset_data(before, after, domain):
    """Extract data matrix X and labels y from before and after mne.Epochs
    data structures.
    """
    before.load_data()
    before_data = before.get_data()

    after.load_data()
    after_data = after.get_data()

    if domain.lower() == "time":
        ## Time Domain
        ntrials, nchs, nsteps = before_data.shape

        X = np.vstack(
            [
                before_data.reshape(before_data.shape[0], -1),  # class 0
                after_data.reshape(after_data.shape[0], -1),  # class 1
            ]
        )
        y = np.concatenate([np.zeros(len(before_data)), np.ones(len(after_data))])
        image_height = nchs
        image_width = nsteps

    elif domain.lower() in ["freq", "frequency"]:
        ## Freq Domain
        nfreqs = 10
        lfreq, hfreq = (70, 200)
        freqs = np.logspace(*np.log10([lfreq, hfreq]), num=nfreqs)
        n_cycles = freqs / 3.0  # different number of cycle per frequency

        after_power = tfr_morlet(
            after,
            freqs=freqs,
            n_cycles=n_cycles,
            average=False,
            return_itc=False,
            decim=3,
            n_jobs=1,
        ).data
        before_power = tfr_morlet(
            before,
            freqs=freqs,
            n_cycles=n_cycles,
            average=False,
            return_itc=False,
            decim=3,
            n_jobs=1,
        ).data

        ntrials, nchs, nfreqs, nsteps = before_power.shape

        X = np.vstack(
            [
                before_power.reshape(before_power.shape[0], -1),  # class 0
                after_power.reshape(after_power.shape[0], -1),  # class 1
            ]
        )
        y = np.concatenate([np.zeros(len(before_power)), np.ones(len(after_power))])

        image_height = nchs * nfreqs
        image_width = nsteps

    else:
        raise ValueError("'domain' is not one of 'time' or 'frequency'")

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
    destination = Path(destination_path) / "movement_onset_experiment"

    if not os.path.exists(destination):
        os.makedirs(destination)

    before, _ = get_event_data(bids_path, tmin=0, tmax=1.0, event_key="At Center")
    before.load_data()

    after, _ = get_event_data(bids_path, tmin=-0.25, tmax=0.75, event_key="Left Target")
    after.load_data()

    X, y, image_height, image_width = _prepare_onset_data(before, after, domain)

    assert X.shape[0] == y.shape[0], "X and y do not have the same number of epochs"

    # Perform K-Fold cross validation
    clf_scores = dict()
    clfs = initialize_classifiers(
        image_height, image_width, n_jobs=-1, random_state=random_state
    )

    for clf in clfs:
        clf_name = _get_classifier_name(clf)
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

    ## Plot results
    # 1. Plot roc curves
    fig, axs = plt.subplots(ncols=2, dpi=100, figsize=(16, 6))
    axs = axs.flatten()
    axs = plot_classifier_performance(clf_scores, X, y, axs=axs)

    axs[0].set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"{subject.upper()} ROC Curves for 'At Center' vs. 'Left Target' ({domain.capitalize()} Domain)",
    )
    axs[0].legend(loc="lower right")

    # 2. Plot accuracies
    axs[1].set(
        ylabel="accuracy",
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
    metrics = dict(
        accuracy="accuracy",
        cohen_kappa_score=make_scorer(cohen_kappa_score),
        roc_auc_ovr="roc_auc_ovr",
    )
    n_splits = 5
    cv = StratifiedKFold(n_splits)

    destination_path = results_path / subject
    movement_onset_experiment(
        bids_path, destination_path, cv, metrics, domain="time", random_state=seed
    )
    movement_onset_experiment(
        bids_path, destination_path, cv, metrics, domain="freq", random_state=seed
    )
