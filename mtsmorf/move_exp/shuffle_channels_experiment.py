import argparse
import os
from pathlib import Path
import yaml


import matplotlib.pyplot as plt
import numpy as np
from mne import Epochs
from mne.time_frequency.tfr import EpochsTFR, tfr_morlet
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from mne_bids.path import BIDSPath
from mne.utils.check import check_random_state

from cv import fit_classifiers_cv
from functions.move_experiment_functions import get_event_data
from plotting import plot_classifier_performance


def _prepare_shuffle_channels_data(inst, labels, shuffle, random_state=None):
    """
    docstring
    """
    rng = check_random_state(random_state)

    if isinstance(inst, Epochs):
        ## Time Domain
        data = inst.get_data()
        ntrials, nchs, nsteps = data.shape
        image_height = nchs
        image_width = nsteps

        if shuffle:
            ch_inds = rng.permutation(nchs)
            data = data[:, ch_inds, :]

    elif isinstance(inst, EpochsTFR):
        ## Freq Domain
        data = inst.data
        ntrials, nchs, nfreqs, nsteps = data.shape
        image_height = nchs * nfreqs
        image_width = nsteps

        if shuffle:
            ch_inds = rng.permutation(nchs)
            freq_inds = rng.permutation(nfreqs)

            # Need to do this to shuffle two axes simultaneously
            data = data[:, ch_inds, :, :][:, :, freq_inds, :]

    else:
        raise TypeError("inst is not an Epochs or EpochsTFR instance.")

    X = data.reshape(ntrials, -1)
    y = labels

    return X, y, image_height, image_width


def shuffle_channels_experiment(
    bids_path,
    destination_path,
    cv,
    metrics,
    domain,
    tmin=-0.2,
    tmax=0.5,
    n_jobs=1,
    random_state=None,
):
    """
    docstring
    """

    destination = Path(destination_path)

    subject = bids_path.subject
    epochs, labels = get_event_data(bids_path, tmin=tmin, tmax=tmax)

    if domain == "time":
        inst = epochs
    elif domain in ["freq", "frequency"]:
        nfreqs = 10
        lfreq, hfreq = (70, 200)
        freqs = np.logspace(*np.log10([lfreq, hfreq]), num=nfreqs)
        n_cycles = freqs / 3.0  # different number of cycle per frequency

        inst = tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            average=False,
            return_itc=False,
            decim=3,
            n_jobs=n_jobs,
        )

    # Keep channels unshuffled
    X_unshuff, y_unshuff, image_height, image_width = _prepare_shuffle_channels_data(
        inst, labels, shuffle=False, random_state=random_state
    )
    clf_scores_unshuffled = fit_classifiers_cv(
        X_unshuff,
        y_unshuff,
        image_height,
        image_width,
        cv,
        metrics,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    # Shuffle channels
    X_shuff, y_shuff, image_height, image_width = _prepare_shuffle_channels_data(
        inst, labels, shuffle=True, random_state=random_state
    )
    clf_scores_shuffled = fit_classifiers_cv(
        X_shuff,
        y_shuff,
        image_height,
        image_width,
        cv,
        metrics,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    ## Plot results
    fig, axs = plt.subplots(nrows=2, ncols=2, dpi=100, figsize=(16, 12), sharey="row")
    axs = axs.flatten()

    plot_classifier_performance(
        clf_scores_unshuffled, X_unshuff, y_unshuff, axs=axs[:2]
    )
    axs[0].set(
        title=f"{subject.upper()}: Accuracy of Classifiers ({domain.capitalize()} Domain Signal, Unshuffled)",
    )
    axs[1].set(
        title=f"{subject.upper()}: ROC AUCs of Classifiers ({domain.capitalize()} Domain Signal, Unshuffled)",
    )

    plot_classifier_performance(clf_scores_shuffled, X_shuff, y_shuff, axs=axs[2:])
    axs[2].set(
        title=f"{subject.upper()}: Accuracy of Classifiers ({domain.capitalize()} Domain Signal, Shuffled)"
    )
    axs[3].set(
        title=f"{subject.upper()}: ROC AUCs of Classifiers ({domain.capitalize()} Domain Signal, Shuffled)"
    )
    fig.tight_layout()
    plt.savefig(destination / f"{domain}_domain/{domain}_domain_comparison.png")
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

    tmin, tmax = (-0.75, 1.25)
    destination_path = results_path / subject / "shuffle_channels_experiment"
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    shuffle_channels_experiment(
        bids_path,
        cv,
        metrics,
        destination_path,
        "time",
        tmin=tmin,
        tmax=tmax,
        n_jobs=-1,
        random_state=seed,
    )

    shuffle_channels_experiment(
        bids_path,
        cv,
        metrics,
        destination_path,
        "freq",
        tmin=tmin,
        tmax=tmax,
        n_jobs=-1,
        random_state=seed
    )
