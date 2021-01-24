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

from .experiment_functions import preprocess_epochs
from plotting import (
    plot_roc_cv,
    plot_accuracies,
    plot_roc_aucs,
    plot_event_durations,
    plot_event_onsets,
)

# Hack-y way to import from files in sibling "io" directory
sys.path.append(str(Path(__file__).parent.parent / "io"))
sys.path.append(str(Path(__file__).parent.parent / "war_exp"))

from read import read_dataset, read_label, read_trial, get_trial_info, _get_bad_chs
from utils import initialize_classifiers


def prepare_epochs(bids_path):
    # fetch labels
    labels, trial_ids = read_label(
        bids_path, trial_id=None, label_keyword="target_direction"
    )

    # we don't want perturbed trials
    behav_tsv, events_tsv = get_trial_info(bids_path)
    success_trial_flag = np.array(list(map(int, behav_tsv["successful_trial_flag"])))
    success_inds = np.where(success_trial_flag == 1)[0]
    force_mag = np.array(behav_tsv["force_magnitude"], np.float64)[success_inds]

    # filter out labels for unsuccessful trials
    unsuccessful_trial_inds = np.where((np.isnan(labels) | (force_mag > 0)))[0]
    labels = np.delete(labels, unsuccessful_trial_inds)

    # get preprocessed epochs data
    fname = os.path.splitext(bids_path.basename)[0] + "-epo.fif"
    fpath = derivatives_path / subject / fname

    epochs = mne.read_epochs(fpath, preload=True)
    epochs = epochs.drop(unsuccessful_trial_inds)

    return epochs, labels


def cv_roc(clf, X, y, cv):

    scores = {}

    scores["train_predict_proba"] = []
    scores["train_preds"] = []
    scores["train_inds"] = []
    scores["train_fpr"] = []
    scores["train_tpr"] = []
    scores["train_fnr"] = []
    scores["train_tnr"] = []
    scores["train_thresholds"] = []
    scores["train_confusion_matrix"] = []

    scores["test_predict_proba"] = []
    scores["test_preds"] = []
    scores["test_inds"] = []
    scores["test_fpr"] = []
    scores["test_tpr"] = []
    scores["test_fnr"] = []
    scores["test_tnr"] = []
    scores["test_thresholds"] = []
    scores["test_confusion_matrix"] = []

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train, test) in enumerate(cv.split(X=X, y=y)):
        clf.fit(X[train], y[train])

        y_train_prob = clf.predict_proba(X[train])
        y_train_pred = clf.predict(X[train])
        y_train = y[train]
        cm_train = confusion_matrix(y_train, y_train_pred)

        fpr, tpr, thresholds = roc_curve(y_train, y_train_prob[:, 1], pos_label=1)
        fnr, tnr, _ = roc_curve(y_train, y_train_prob[:, 1], pos_label=0)

        scores["train_predict_proba"].append(y_train_prob.tolist())
        scores["train_preds"].append(list(y_train_pred))
        scores["train_inds"].append(train.tolist())
        scores["train_fpr"].append(fpr.tolist())
        scores["train_tpr"].append(tpr.tolist())
        scores["train_thresholds"].append(thresholds.tolist())
        scores["train_fnr"].append(fnr.tolist())
        scores["train_tnr"].append(tnr.tolist())
        scores["train_confusion_matrix"].append(cm_train.tolist())

        # For binary classification get probability for class 1
        y_pred_prob = clf.predict_proba(X[test])
        y_pred = clf.predict(X[test])
        y_test_pred = clf.predict(X[test])
        y_test = y[test]

        # Compute the curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1], pos_label=1)
        fnr, tnr, _ = roc_curve(y_test, y_pred_prob[:, 1], pos_label=0)
        cm_test = confusion_matrix(y_test, y_test_pred)

        scores["test_predict_proba"].append(y_pred_prob.tolist())
        scores["test_preds"].append(list(y_pred))
        scores["test_inds"].append(test.tolist())
        scores["test_fpr"].append(fpr.tolist())
        scores["test_tpr"].append(tpr.tolist())
        scores["test_thresholds"].append(thresholds.tolist())
        scores["test_fnr"].append(fnr.tolist())
        scores["test_tnr"].append(tnr.tolist())
        scores["test_confusion_matrix"].append(cm_test.tolist())

    return scores


def cv_fit(
    clf,
    X,
    y,
    cv=None,
    metrics=None,
    n_jobs=None,
    return_train_score=False,
    return_estimator=False,
):
    # Create a reset copy of estimator with same parameters

    # See table of sklearn metrics with a corresponding string at
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    if metrics is None:
        metrics = [
            "accuracy",  # accuracy_score
            "f1",  # f1_score
            "neg_brier_score",  # brier_score_loss
            "precision",  # precision_score
            "recall",  # recall_score
            "roc_auc",  # roc_auc_score
        ]

    # Applying cross validation with specified metrics and keeping training scores and estimators.
    scores = cross_validate(
        clf,
        X,
        y,
        scoring=metrics,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=return_train_score,
        return_estimator=return_estimator,
    )

    # Appending metrics for computing ROC curve
    scores.update(cv_roc(clf, X, y, cv))

    # Appending model parameters
    scores["model_params"] = clf.get_params()

    return scores


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

    metrics = [
        "accuracy",
        "roc_auc_ovr",
    ]

    clf_scores = dict()
    clfs = initialize_classifiers(
        image_height, image_width, n_jobs=-1, random_state=random_state
    )

    for clf in clfs:
        if clf.__class__.__name__ == "rerfClassifier":
            clf_name = clf.get_params()["projection_matrix"]
        elif clf.__class__.__name__ == "DummyClassifier":
            clf_name = clf.strategy
        else:
            clf_name = clf.__class__.__name__

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

    return clf_scores


def shuffle_channels_experiment(
    epochs,
    labels,
    cv,
    destination_path,
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

    ## Time domain
    clf_scores_unshuffled = run_classifier_comparison(
        epochs_anat,
        labels,
        cv,
        freq_domain=False,
        shuffle_channels=False,
        random_state=random_state,
    )

    clf_scores_shuffled = run_classifier_comparison(
        epochs_anat,
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

    ## Freq Domain (Averaging)
    clf_scores_unshuffled = run_classifier_comparison(
        epochs_anat,
        labels,
        cv,
        freq_domain=True,
        shuffle_channels=False,
        avg_freq=True,
        nfreqs=nfreqs,
        lfreq=lfreq,
        hfreq=hfreq,
        random_state=random_state,
    )

    clf_scores_shuffled = run_classifier_comparison(
        epochs_anat,
        labels,
        cv,
        freq_domain=True,
        shuffle_channels=True,
        avg_freq=True,
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
        title=f"{subject.upper()}: Accuracy of Classifiers (n={nfreqs} freq bins, Averaged over {lfreq}-{hfreq} Hz, Unshuffled)",
    )

    plot_accuracies(clf_scores_shuffled, ax=axs[1])
    axs[1].set(
        title=f"{subject.upper()}: Accuracy of Classifiers (n={nfreqs} freq bins, Averaged over {lfreq}-{hfreq} Hz, Shuffled)"
    )

    plot_roc_aucs(clf_scores_unshuffled, ax=axs[2])
    axs[2].set(
        ylabel="ROC AUC",
        title=f"{subject.upper()}: ROC AUCs of Classifiers (n={nfreqs} freq bins, Averaged over {lfreq}-{hfreq} Hz, Unshuffled)",
    )

    plot_roc_aucs(clf_scores_shuffled, ax=axs[3])
    axs[3].set(
        title=f"{subject.upper()}: ROC AUCs of Classifiers (n={nfreqs} freq bins, Averaged over {lfreq}-{hfreq} Hz, Shuffled)"
    )
    fig.tight_layout()
    plt.savefig(destination / "freq_domain_averaging_comparison.png")
    plt.close(fig)

    ## Freq Domain (No Averaging)
    ## Fit models
    clf_scores_unshuffled = run_classifier_comparison(
        epochs_anat,
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
        epochs_anat,
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


def movement_onset_experiment(bids_path, destination_path, domain, random_state=None):
    """
    docstring
    """
    subject = bids_path.subject
    destination = Path(destination_path) / "movement_onset_experiment"

    if not os.path.exists(destination):
        os.makedirs(destination)

    before = read_dataset(bids_path, kind="ieeg", tmin=0, tmax=1.0, 
                          event_key="At Center")
    before.load_data()
    before_data = preprocess_epochs(before).get_data()

    after = read_dataset(bids_path, kind="ieeg", tmin=-0.25, tmax=0.75, 
                         event_key="Left Target")
    after.load_data()
    after_data = preprocess_epochs(after).get_data()

    if domain.lower() == "time":
        ## Time Domain
        ntrials, nchs, nsteps = before_data.shape

        X = np.vstack([
                before_data.reshape(before_data.shape[0], -1),  # class 0
                after_data.reshape(after_data.shape[0], -1),  # class 1
            ])
        y = np.concatenate([np.zeros(len(before_data)), np.ones(len(after_data))])
        image_height = nchs
        image_width = nsteps

    elif domain.lower() == "frequency":
        ## Freq Domain
        nfreqs = 10
        lfreq, hfreq = (70, 200)
        freqs = np.logspace(*np.log10([lfreq, hfreq]), num=nfreqs)
        n_cycles = freqs / 3.0  # different number of cycle per frequency

        after_power = tfr_morlet(after, freqs=freqs, n_cycles=n_cycles,
                                 average=False, return_itc=False, decim=3,
                                 n_jobs=1).data
        before_power = tfr_morlet(before, freqs=freqs, n_cycles=n_cycles,
                                  average=False, return_itc=False, decim=3,
                                  n_jobs=1).data

        ntrials, nchs, nfreqs, nsteps = before_power.shape

        X = np.vstack([
                before_power.reshape(before_power.shape[0], -1),  # class 0
                after_power.reshape(after_power.shape[0], -1),  # class 1
            ])
        y = np.concatenate([np.zeros(len(before_power)), np.ones(len(after_power))])

        image_height = nchs * nfreqs
        image_width = nsteps

    else:
        raise ValueError("'domain' is not one of 'time' or 'frequency'")

    assert X.shape[0] == y.shape[0], "X and y do not have the same number of epochs"

    # Perform K-Fold cross validation
    n_splits = 5
    cv = StratifiedKFold(n_splits)

    metrics = dict(
        accuracy="accuracy",
        cohen_kappa_score=make_scorer(cohen_kappa_score),
        roc_auc_ovr="roc_auc_ovr",
    )

    clf_scores = dict()
    clfs = initialize_classifiers(
        image_height, image_width, n_jobs=-1, random_state=random_state
    )

    for clf in clfs:
        if clf.__class__.__name__ == "rerfClassifier":
            clf_name = clf.get_params()["projection_matrix"]
        elif clf.__class__.__name__ == "DummyClassifier":
            clf_name = clf.strategy
        else:
            clf_name = clf.__class__.__name__

        clf_scores[clf_name] = cv_fit(clf, X, y, cv=cv, metrics=metrics, 
                                      n_jobs=None, return_train_score=True, 
                                      return_estimator=True)

    ## Plot results
    # 1. Plot roc curves
    fig, axs = plt.subplots(ncols=2, dpi=100, figsize=(16, 6))
    axs = axs.flatten()

    for clf_name, scores in clf_scores.items():
        plot_roc_cv(scores["test_predict_proba"], X, y, scores["test_inds"], 
                    label=clf_name, show_chance=False, ax=axs[0])

    axs[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", 
            xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
            title=f"{subject.upper()} ROC Curves for 'At Center' vs. 'Left Target' ({domain.capitalize()} Domain)")
    axs[0].legend(loc="lower right")

    # 2. Plot accuracies
    plot_accuracies(clf_scores, ax=axs[1])
    axs[1].set(ylabel="accuracy", title=f"{subject.upper()}: Accuracies for 'At Center' vs. 'Left Target' ({domain.capitalize()} Domain)")
    fig.tight_layout()

    plt.savefig(destination / f"movement_onset_{domain}_domain.png")
    plt.close(fig)
    print(f"Figure saved at {destination}/movement_onset_{domain}_domain.png")


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
        random_seed=rng,
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
        random_seed=rng,
    )

    my_data.mean_diff.plot(ax=axs[1])
    axs[1].set(title="ROC AUC Comparison in Baseline Correction of Time Domain Signal")


def baseline_experiment(bids_path, destination_path, random_state=None):
    """
    docstring
    """
    destination = Path(destination_path) / "baseline_experiment"

    if not os.path.exists(destination):
        os.makedirs(destination)

    # fetch labels
    labels, trial_ids = read_label(
        bids_path, trial_id=None, label_keyword="target_direction"
    )

    # we don't want perturbed trials
    behav_tsv, events_tsv = get_trial_info(bids_path)
    success_trial_flag = np.array(list(map(int, behav_tsv["successful_trial_flag"])))
    success_inds = np.where(success_trial_flag == 1)[0]
    force_mag = np.array(behav_tsv["force_magnitude"], np.float64)[success_inds]

    # filter out labels for unsuccessful trials
    unsuccessful_trial_inds = np.where((np.isnan(labels) | (force_mag > 0)))[0]
    labels = np.delete(labels, unsuccessful_trial_inds)

    epochs = read_dataset(
        bids_path,
        kind="ieeg",
        tmin=-0.75,
        # tmax=1.25,
        tmax=0.5,
        picks=None,
        event_key="Left Target",
    )
    epochs.load_data()

    ## Low-pass filter up to sfreq/2
    fs = epochs.info["sfreq"]
    epochs = epochs.filter(l_freq=1, h_freq=fs / 2 - 1)

    ## Downsample epochs to 500 Hz
    resample_rate = 500
    epochs = epochs.resample(resample_rate)
    epochs.drop(unsuccessful_trial_inds)

    cropped = epochs.copy()
    # cropped = cropped.crop(tmin=-0.5, tmax=1.0)
    cropped = cropped.crop(tmin=-0.3, tmax=0.3)

    cropped_data = cropped.get_data()
    ntrials, nchs, nsteps = cropped_data.shape

    # Create X, y data
    included_trials = np.isin(labels, [0, 1, 2, 3])

    X = cropped_data[included_trials].reshape(np.sum(included_trials), -1)
    y = labels[included_trials]

    # Perform K-Fold cross validation
    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)

    metrics = [
        "accuracy",
        "roc_auc_ovr",
    ]

    mtsmorf = rerfClassifier(
        projection_matrix="MT-MORF",
        max_features="auto",
        n_jobs=-1,
        random_state=rng,
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

    ## Apply baseline
    baseline = read_dataset(
        bids_path,
        kind="ieeg",
        tmin=0.0,
        # tmax=2.0,
        tmax=1.25,
        picks=None,
        event_key="At Center",
    )
    baseline.load_data()

    # Low-pass filter up to sfreq/2
    fs = baseline.info["sfreq"]
    baseline = baseline.filter(l_freq=0.5, h_freq=fs / 2 - 1)

    # Downsample epochs to 500 Hz
    resample_rate = 500
    baseline = baseline.resample(resample_rate)
    baseline.drop(unsuccessful_trial_inds)

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
        random_state=rng,
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

    mtsmorf = rerfClassifier(
        projection_matrix="MT-MORF",
        max_features="auto",
        n_jobs=-1,
        random_state=random_state,
        image_height=avg_freq_data.shape[1],
        image_width=avg_freq_data.shape[2],
    )

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
    # inds = np.where((times >= -0.5) & (times <= 1.0))[0]
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


def frequency_band_comparison(epochs, destination_path, random_state=None):
    """
    docstring
    """
    destination = Path(destination_path)

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

    rng = 1
    nfreqs = 10
    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits)

    metrics = [
        "accuracy",
        "roc_auc_ovr",
    ]

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
    idx = [list(scores.keys())[-1]] + list(scores.keys())[
        :-1
    ]  # Re-order so that control is hi-gamma band
    my_data = dabest.load(df, idx=idx, resamples=100, random_seed=rng)
    my_data.mean_diff.plot(ax=axs[0])
    axs[0].set(title=f"{subject.upper()} Accuracy Comparison between Frequency Bands")

    ## ROC AUC comparison
    roc_auc_ovrs = {name: score["test_roc_auc_ovr"] for name, score in scores.items()}
    roc_auc_ovrs["ID"] = id_col
    df = pd.DataFrame(roc_auc_ovrs)
    my_data = dabest.load(df, idx=idx, resamples=100, random_seed=rng)
    my_data.mean_diff.plot(ax=axs[1])
    axs[1].set(title=f"{subject.upper()} ROC AUC Comparison between Frequency Bands")

    fig.tight_layout()
    plt.savefig(
        destination / f"{subject}_frequency_band_comparison_tmin=-0.5_tmax=1.0.png"
    )
    plt.close(fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=str, help="subject ID (e.g. efri02)")
    parser.add_argument(
        "-experiment",
        type=str,
        choices=[
            "shuffle",
            "movement_onset_time",
            "movement_onset_frequency",
            "baseline",
            "frequency_bands",
            "plot_event_durations",
            "plot_event_onsets",
        ],
        help="which experiment to run",
    )

    args = parser.parse_args()
    subject = args.subject
    experiment = args.experiment

    tmin, tmax = (-0.75, 1.25)
    bids_root = Path("/workspaces/research/mnt/data/efri/")
    derivatives_path = (
        bids_root
        / "derivatives"
        / "preprocessed"
        / f"tmin={tmin}-tmax={tmax}"
        / "band-pass=1-1000Hz-downsample=500"
    )

    results_path = Path(
        "/workspaces/research/efri OneDrive/Adam Li - efri/derivatives/workstation_output"
    )

    # subject identifiers
    session = "efri"
    task = "move"
    acquisition = "seeg"
    run = "01"
    kind = "ieeg"
    trial_id = 2

    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        acquisition=acquisition,
        run=run,
        suffix=kind,
        extension=".vhdr",
        root=bids_root,
    )

    # Prep data for model fitting
    epochs, labels = prepare_epochs(bids_path)

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

    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        acquisition=acquisition,
        run=run,
        suffix=kind,
        extension=".tsv",
        root=bids_root,
    )
    bids_path.update(suffix="channels")
    bads = _get_bad_chs(bids_path)

    tmp = _from_tsv(bids_path)
    tmp = [
        (name, anat)
        for (name, anat) in zip(tmp["name"], tmp["anat"])
        if name not in bads
    ]

    channels = pd.DataFrame(tmp, columns=["name", "anat"])
    sorted_channels = channels.sort_values(["anat"])
    sorted_channels.head()
    anat = sorted_channels["anat"].str.contains(
        "insular cortex|central sulcus|middle temporal gyrus"
    )
    picks = list(sorted_channels[anat]["name"])
    epochs_anat = epochs.pick_channels(picks)

    rng = 1
    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits)

    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        acquisition=acquisition,
        run=run,
        suffix=kind,
        extension=".vhdr",
        root=bids_root,
    )

    if experiment == "shuffle":
        shuffle_channels_experiment(
            epochs_anat,
            labels,
            cv,
            results_path / subject,
            nfreqs=10,
            lfreq=70,
            hfreq=200,
            random_state=rng,
        )
    elif experiment == "movement_onset_time":
        movement_onset_experiment(
            bids_path, results_path / subject, domain="time", random_state=rng
        )

    elif experiment == "movement_onset_frequency":
        movement_onset_experiment(
            bids_path, results_path / subject, domain="frequency", random_state=rng
        )

    elif experiment == "baseline":
        baseline_experiment(
            bids_path,
            results_path / subject,
            random_state=rng,
        )

    elif experiment == "frequency_bands":
        epochs.crop(tmin=-0.5, tmax=1.0)
        frequency_band_comparison(epochs, results_path / subject, random_state=rng)

    elif experiment == "plot_event_durations":
        subjects = [
            "efri02",
            "efri06",
            "efri07",
            # "efri09",  # Too few samples
            # "efri10",  # Unequal data size vs label size
            "efri13",
            "efri14",
            "efri15",
            "efri18",
            "efri20",
            "efri26",
        ]

        fig, axs = plt.subplots(nrows=3, ncols=3, dpi=300, figsize=(18, 18))
        axs = axs.flatten()

        for i, subject in enumerate(subjects):
            bids_path = BIDSPath(
                subject=subject,
                session=session,
                task=task,
                acquisition=acquisition,
                run=run,
                suffix=kind,
                extension=".vhdr",
                root=bids_root,
            )

            ax = axs[i]

            behav, events = map(pd.DataFrame, get_trial_info(bids_path))
            plot_event_durations(behav, events, ax=ax, random_state=rng)

            ax.set(
                ylabel='Onset Relative to "Go Cue" (s)',
                title=f"{subject.upper()}: Onset of Events",
            )
        fig.tight_layout()
        plt.savefig(results_path / "all_subjects_event_durations.png")

        # fig, ax = plt.subplots(dpi=150, figsize=(8, 6))

        # behav, events = map(pd.DataFrame, get_trial_info(bids_path))
        # plot_event_durations(behav, events, ax=ax, random_state=rng)

        # ax.set(ylabel="duration (s)", title=f"{subject.upper()}: Duration of Events")
        # fig.tight_layout()
        # plt.savefig(results_path / subject / f"{subject}_event_durations.png")

    elif experiment == "plot_event_onsets":
        subjects = [
            "efri02",
            "efri06",
            "efri07",
            # "efri09",  # Too few samples
            # "efri10",  # Unequal data size vs label size
            "efri13",
            "efri14",
            "efri15",
            "efri18",
            "efri20",
            "efri26",
        ]

        fig, axs = plt.subplots(nrows=3, ncols=3, dpi=300, figsize=(18, 18))
        axs = axs.flatten()

        for i, subject in enumerate(subjects):
            bids_path = BIDSPath(
                subject=subject,
                session=session,
                task=task,
                acquisition=acquisition,
                run=run,
                suffix=kind,
                extension=".vhdr",
                root=bids_root,
            )

            ax = axs[i]

            behav, events = map(pd.DataFrame, get_trial_info(bids_path))
            plot_event_onsets(behav, events, ax=ax, random_state=rng)

            ax.set(
                ylabel='Onset Relative to "Go Cue" (s)',
                title=f"{subject.upper()}: Onset of Events",
            )
        fig.tight_layout()
        plt.savefig(results_path / "all_subjects_event_onsets.png")
        # fig, ax = plt.subplots(dpi=150, figsize=(8, 6))

        # behav, events = map(pd.DataFrame, get_trial_info(bids_path))
        # plot_event_onsets(behav, events, ax=ax, random_state=rng)

        # ax.set(
        #     ylabel='Onset Relative to "Go Cue" (s)',
        #     title=f"{subject.upper()}: Onset of Events",
        # )
        # fig.tight_layout()
        # plt.savefig(results_path / subject / f"{subject}_event_onsets.png")