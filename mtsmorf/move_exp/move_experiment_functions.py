import sys

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from rerf.rerfClassifier import rerfClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state

from cv import cv_roc, cv_fit

sys.path.append(str(Path(__file__).parent.parent / "io"))

from read import get_trial_info, read_label, read_dataset


def _preprocess_epochs(epochs, resample_rate=500):
    """Preprocess mne.Epochs object in the following way:
    1. Low-pass filter up to Nyquist frequency
    2. Downsample data to 500 Hz
    """
    # low-pass filter up to sfreq/2
    fs = epochs.info["sfreq"]
    new_epochs = epochs.filter(l_freq=1, h_freq=fs / 2 - 1)

    # downsample epochs to 500 Hz
    new_epochs = new_epochs.resample(resample_rate)

    return new_epochs


def _get_unperturbed_trial_inds(behav):
    """Get trial indices where force magnitude > 0."""
    if not isinstance(behav, pd.DataFrame):
        behav = pd.DataFrame(behav)

    behav[["successful_trial_flag", "force_magnitude"]] = behav[
        ["successful_trial_flag", "force_magnitude"]
    ].apply(pd.to_numeric)

    # filter out failed trials -- we don't want these anyway
    successes = behav[behav.successful_trial_flag == 1]
    successes.index = np.arange(len(successes))

    # filter out labels for perturbed trials
    unperturbed_trial_inds = successes[successes.force_magnitude == 0].index
    unperturbed_trial_inds = unperturbed_trial_inds.to_list()

    return unperturbed_trial_inds


def get_trial_info_pd(bids_path, verbose=False):
    """Convert behav and events OrderedDict objects to pd.DataFrame objects
    with numerical columns appropriately typed.
    """
    behav, events = map(pd.DataFrame, get_trial_info(bids_path, verbose=verbose))

    behav_numerical_cols = [
        "trial_id",
        "successful_trial_flag",
        "missed_target_flag",
        "correct_speed_flag",
        "force_angular",
        "force_magnitude",
        "target_direction",
    ]
    behav[behav_numerical_cols] = behav[behav_numerical_cols].apply(pd.to_numeric)

    events_numerical_cols = ["onset", "duration", "value", "sample"]
    events[events_numerical_cols] = events[events_numerical_cols].apply(pd.to_numeric)
    return behav, events


def get_preprocessed_labels(
    bids_path, trial_id=None, label_keyword="target_direction", verbose=False
):
    """Read labels for each trial for the specified keyword. Keep labels for
    successful and unperturbed trials.
    """
    behav, events = get_trial_info_pd(bids_path, verbose=verbose)
    labels, _ = read_label(bids_path, trial_id=trial_id, label_keyword=label_keyword)

    # keep perturbed trial inds
    unperturbed_trial_inds = _get_unperturbed_trial_inds(behav)
    labels = labels[unperturbed_trial_inds]

    return labels


def get_event_durations(bids_path, event_key="Left Target", periods=1, verbose=False):
    """Get the event durations for the specified event_key for the specified
    period.
    """
    behav, events = get_trial_info_pd(bids_path, verbose=verbose)

    # get difference between Left Target onset and its preceding and succeeding events
    inds = events.trial_type == event_key
    durations = events.onset.diff(periods=periods).abs()[inds]
    durations.index = np.arange(len(durations))

    # remove perturbed trial indices
    unperturbed_trial_inds = _get_unperturbed_trial_inds(behav)
    durations = durations.iloc[unperturbed_trial_inds]

    return durations


def get_event_data(
    bids_path,
    kind="ieeg",
    tmin=-0.2,
    tmax=0.5,
    event_key="Left Target",
    trial_id=None,
    label_keyword="target_direction",
):
    """Read preprocessed mne.Epochs data structure time locked to label_keyword
    with corresponding trial information.
    """
    # get epochs
    behav, _ = get_trial_info_pd(bids_path)

    epochs = read_dataset(
        bids_path, kind=kind, tmin=tmin, tmax=tmax, event_key=event_key
    )
    epochs.load_data()
    epochs = _preprocess_epochs(epochs)

    unperturbed_trial_inds = _get_unperturbed_trial_inds(behav)
    perturbed_trial_inds = [
        i for i in range(len(epochs)) if not i in unperturbed_trial_inds
    ]
    epochs.drop(perturbed_trial_inds)

    # get labels
    labels = get_preprocessed_labels(
        bids_path, trial_id=trial_id, label_keyword=label_keyword
    )

    return epochs, labels


def independence_test(X, y):
    """Compute point estimates for coefficient between X and y."""
    covariates = sm.add_constant(X)
    model = sm.MNLogit(y, covariates)

    res = model.fit(disp=False)
    coeff = res.params.iloc[1]

    return coeff


def bootstrap_independence_test(
    X, y, num_bootstraps=200, alpha=0.05, random_state=None
):
    """Bootstrap esitmates for coefficients between X and y."""
    rng = check_random_state(random_state)

    Ql = alpha / 2
    Qu = 1 - alpha / 2

    estimates = []

    n = len(X)

    for i in range(num_bootstraps):

        # Compute OR estimate for bootstrap sample
        inds = rng.randint(n, size=n)
        Xboot = X.iloc[inds]
        yboot = y.iloc[inds]

        estimates.append(independence_test(Xboot, yboot))

    # Get desired lower and upper percentiles of approximate sampling distribution
    q_low = np.percentile(estimates, Ql * 100)
    q_up = np.percentile(estimates, Qu * 100)

    return q_low, q_up, estimates


def independence_test_OLS(X, y):
    """Compute point estimates for regression coefficient between X and y."""
    covariates = sm.add_constant(X)
    model = sm.OLS(y, covariates)

    res = model.fit(disp=False)
    coeff = res.params[1]

    return coeff


def bootstrap_independence_test_OLS(
    X, y, num_bootstraps=200, alpha=0.05, random_state=None
):
    """Bootstrap esitmates for regression coefficients between X and y."""
    rng = check_random_state(random_state)

    Ql = alpha / 2
    Qu = 1 - alpha / 2

    estimates = np.empty(num_bootstraps,)

    n = len(X)

    for i in range(num_bootstraps):

        # Compute OR estimate for bootstrap sample
        inds = rng.randint(n, size=n)
        Xboot = X.iloc[inds]
        yboot = y.iloc[inds]

        estimates[i] = independence_test_OLS(Xboot, yboot)

    # Get desired lower and upper percentiles of approximate sampling distribution
    q_low = np.percentile(estimates, Ql * 100)
    q_up = np.percentile(estimates, Qu * 100)

    return q_low, q_up, estimates


def initialize_classifiers(image_height, image_width, n_jobs=1, random_state=None):
    """Initialize a list of classifiers to be compared."""

    mtsmorf = rerfClassifier(
        projection_matrix="MT-MORF",
        max_features="auto",
        n_jobs=n_jobs,
        random_state=random_state,
        image_height=image_height,
        image_width=image_width,
    )

    srerf = rerfClassifier(
        projection_matrix="S-RerF",
        max_features="auto",
        n_jobs=n_jobs,
        random_state=random_state,
        image_height=image_height,
        image_width=image_width,
    )

    lr = LogisticRegression(random_state=random_state)
    rf = RandomForestClassifier(random_state=random_state)
    dummy = DummyClassifier(strategy="most_frequent", random_state=random_state)

    clfs = [mtsmorf, srerf, lr, rf, dummy]

    return clfs


def fit_classifiers_cv(X, y, image_height, image_width, cv, metrics, random_state=None):
    """Run cross-validation for classifiers listed in initialize_classifiers()."""
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
