import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mne_bids.path import BIDSPath
from mne import Epochs
from rerf.rerfClassifier import rerfClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import GroupKFold, cross_validate

if os.path.abspath(Path(__file__).parents[2]) not in sys.path:
    sys.path.append(os.path.abspath(Path(__file__).parents[2]))

from mtsmorf.io.move.read import read_move_trial_epochs, read_trial_metadata
from mtsmorf.move_exp.preprocess import get_event_data
from mtsmorf.io.utils import NumpyEncoder
from sklearn.utils import check_random_state
from sklearn.dummy import DummyClassifier
import yaml


def read_cohort_movement_data(root, cohort, picks, resample_rate=500):
    data = []
    all_labels = []
    groups = []

    for subject in cohort:
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
        
        before = read_move_trial_epochs(root, subject, event_key="At Center", tmin=0, tmax=1.)
        before = before.pick(picks[subject])
        before.load_data()
        after = read_move_trial_epochs(root, subject, event_key="Left Target", tmin=-0.25, tmax=0.75)
        after = after.pick(picks[subject])
        after.load_data()
        
        # Get data for before movement onset
        before = before.filter(l_freq=1, h_freq=before.info["sfreq"] / 2. - 1)
        before = before.resample(resample_rate)
        data.append(before.get_data())
        all_labels.append([0] * len(before))
        groups.extend([subject] * len(before))

        # Get data for after movement onset
        after = after.filter(l_freq=1, h_freq=after.info["sfreq"] / 2. - 1)
        after = after.resample(resample_rate)
        data.append(after.get_data())
        all_labels.append([1] * len(after))
        groups.extend([subject] * len(after))

    data = np.vstack(data)
    all_labels = np.hstack(all_labels)

    assert data.shape[0] == all_labels.shape[0], f"Unequal array lengths: {data.shape[0]} and {all_labels.shape[0]}"
    
    return data, all_labels, groups


def read_cohort_directionality_data(root, cohort, picks, tmin=0, tmax=0.25, resample_rate=500, return_epochs=False):
    data = []
    all_labels = []
    groups = []
    if return_epochs:
        epochs_dict = {}

    for subject in cohort:
        epochs, labels = get_event_data(root, subject, tmin=tmin, tmax=tmax, resample_rate=None)
        epochs = epochs.pick(picks[subject])
        epochs = epochs.filter(l_freq=1, h_freq=epochs.info["sfreq"] / 2.0 - 1)
        epochs = epochs.resample(resample_rate)

        all_labels.append(labels)
        data.append(epochs.get_data())
        groups.extend([subject] * len(epochs))

        if return_epochs:
            epochs_dict[subject] = epochs

    data = np.vstack(data)
    all_labels = np.hstack(all_labels)

    assert data.shape[0] == all_labels.shape[0], f"Unequal array lengths: {data.shape[0]} and {all_labels.shape[0]}"
    
    if return_epochs:
        return data, all_labels, groups, epochs_dict

    return data, all_labels, groups


def read_cohort_planning_data(root, cohort, picks, tmin=-0.5, tmax=0, resample_rate=500, return_epochs=False):
    data = []
    all_labels = []
    groups = []
    if return_epochs:
        epochs_dict = {}

    for subject in cohort:
        epochs, labels = get_event_data(root, subject, tmin=tmin, tmax=tmax, resample_rate=None)        
        epochs = epochs.pick(picks[subject])        
        epochs = epochs.filter(l_freq=1, h_freq=epochs.info["sfreq"] / 2.0 - 1)
        epochs = epochs.resample(resample_rate)

        all_labels.append(labels)
        data.append(epochs.get_data())
        groups.extend([subject] * len(epochs))

        if return_epochs:
            epochs_dict[subject] = epochs

    data = np.vstack(data)
    all_labels = np.hstack(all_labels)

    assert data.shape[0] == all_labels.shape[0], f"Unequal array lengths: {data.shape[0]} and {all_labels.shape[0]}"
    
    if return_epochs:
        return data, all_labels, groups, epochs_dict

    return data, all_labels, groups

if __name__ == "__main__":
    cohort = [
        "efri07",
        "efri13",
        "efri14",
        "efri18",
        "efri20",
    ]

    with open(Path(os.path.dirname(__file__)) / "config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    root = Path(config["bids_root"])
    results_path = Path(config["results_path"])

    seed = 1
    n_splits = 5
    cv = GroupKFold(n_splits)
    metrics = dict(accuracy="accuracy", 
                cohen_kappa_score=make_scorer(cohen_kappa_score),
                roc_auc_ovr="roc_auc_ovr")

    resample_rate = 500

    channels = {
        "efri07" : [
            "F1", "E3",       # Fusiform gyrus
            "E5", "E6",       # ITG
            "B7", "B8",       # MTG
            "U4", "U5"        # STG
        ],
        "efri13" : [
            "F'1", "E'4",
            "E'8", "E'7",
            "F'7", "F'9",
            "U'3", "U'4",
        ],
        "efri14" : [
            "F'1", "F'2",
            "F'3", "F'6",
            "B'7", "B'8",
            "U'8", "U'7"
        ],
        "efri18" : [
            "F'1", "F'2",
            "B'6", "B'7",
            "TP'9", "F'14",
            "U'8", "U'6"
        ],
        "efri20" : [
            "E4", "O1",
            "O8", "O9",
            "C12", "C14",
            "C10", "C9"
        ],
    }

    destination_path = results_path / "cohort_decode_movement"
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    data, labels, groups = read_cohort_movement_data(root, cohort, channels, resample_rate=resample_rate)

    random_state = 1
    image_height, image_width = data.shape[1:]
    clf = rerfClassifier(
        n_estimators=500,
        projection_matrix="MT-MORF",
        max_features="auto",
        n_jobs=-1,
        random_state=random_state,
        image_height=image_height,
        image_width=image_width,
    )

    X = data.reshape(data.shape[0], -1)
    y = labels

    # Plot for MT-MORF
    scores = cross_validate(clf, X, y, groups=groups, scoring=metrics, cv=cv, 
                            return_train_score=True, return_estimator=True)
    fig, ax = plt.subplots(dpi=100, figsize=(8, 6))
    ax.bar(["efri13", "efri14", "efri18", "efri20", "efri07"], scores["test_accuracy"])
    ax.set(xlabel="Test Subject", ylabel="Test Accuracy", ylim=[0, 1], title="Leave-One-Out Testing")
    fig.tight_layout()
    plt.show()
    plt.savefig(destination_path / "leave_one_out_accuracy_baseline_MT-MORF.png")

    fig, ax = plt.subplots(dpi=100, figsize=(8, 6))
    ax.bar(["efri13", "efri14", "efri18", "efri20", "efri07"], scores["test_roc_auc_ovr"])
    ax.set(xlabel="Test Subject", ylabel="Test AUROC", ylim=[0, 1], title="Leave-One-Out Testing")
    fig.tight_layout()
    plt.show()
    plt.savefig(destination_path / "leave_one_out_auroc_baseline_MT-MORF.png")

    # Plot for Dummy
    dummy = DummyClassifier(strategy="most_frequent", random_state=random_state)
    dummy_scores = cross_validate(dummy, X, y, groups=groups, scoring=metrics, cv=cv, 
                                return_train_score=True, return_estimator=True)
    fig, ax = plt.subplots(dpi=100, figsize=(8, 6))
    ax.bar(["efri13", "efri14", "efri18", "efri20", "efri07"], dummy_scores["test_accuracy"])
    ax.set(xlabel="Test Subject", ylabel="Test Accuracy", ylim=[0, 1], title="Leave-One-Out Testing")
    fig.tight_layout()
    plt.show()
    plt.savefig(destination_path / "leave_one_out_accuracy_baseline_dummy.png")

    fig, ax = plt.subplots(dpi=100, figsize=(8, 6))
    ax.bar(["efri13", "efri14", "efri18", "efri20", "efri07"], dummy_scores["test_roc_auc_ovr"])
    ax.set(xlabel="Test Subject", ylabel="Test AUROC", ylim=[0, 1], title="Leave-One-Out Testing")
    fig.tight_layout()
    plt.show()
    plt.savefig(destination_path / "leave_one_out_auroc_baseline_dummy.png")