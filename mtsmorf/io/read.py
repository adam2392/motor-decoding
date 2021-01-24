import json
import sys
from collections import OrderedDict
from pprint import pprint
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from mne_bids import read_raw_bids
from mne_bids.tsv_handler import _from_tsv
from mne_bids.path import _find_matching_sidecar
from sklearn.preprocessing import LabelBinarizer

sys.path.append(str(Path(__file__).parent.parent / "io"))
from utils import NumpyEncoder


def get_trial_info(bids_path):
    """Get behavior and events trial info from tsv files."""
    behav_fpath = _find_matching_sidecar(bids_path, suffix="behav", extension=".tsv")
    behav_tsv = _from_tsv(behav_fpath)

    events_fpath = _find_matching_sidecar(bids_path, suffix="events", extension=".tsv")
    events_tsv = _from_tsv(events_fpath)

    success_trial_flag = np.array(list(map(int, behav_tsv["successful_trial_flag"])))

    # successful trial indices
    success_inds = np.where(success_trial_flag == 1)[0]
    num_trials = len(behav_tsv["trial_id"])
    print(
        f"Out of {num_trials} trials, there were {len(success_inds)} successful trials "
        f"in {bids_path}."
    )

    return behav_tsv, events_tsv


def _read_ch_anat(bids_path):
    electrodes_fpath = _find_matching_sidecar(
        bids_path, suffix="channels", extension=".tsv"
    )
    electrodes_tsv = _from_tsv(electrodes_fpath)

    # extract channel names and anatomy
    ch_names = electrodes_tsv["name"]
    ch_anat = electrodes_tsv["anat"]

    # create dictionary of ch name to anatomical region
    ch_anat_dict = {name: anat for name, anat in zip(ch_names, ch_anat)}
    return ch_anat_dict


def _get_maximum_trial_length(events_tsv):
    max_trial_len = _get_trial_length_by_kwarg(
        events_tsv, "Reserved (Start Trial)", "Reserved (End Trial)"
    )
    return max_trial_len


def _get_trial_length_by_kwarg(
    events_tsv, start_trial_type, stop_trial_type, successful_trials
):
    start_trial_inds = [
        i
        for i, x in enumerate(events_tsv["trial_type"])
        if x.lower() == start_trial_type.lower()
        if i in successful_trials
    ]
    end_trial_inds = [
        i
        for i, x in enumerate(events_tsv["trial_type"])
        if x.lower() == stop_trial_type.lower()
        if i in successful_trials
    ]

    print(len(start_trial_inds), len(end_trial_inds))
    assert len(start_trial_inds) == len(end_trial_inds)

    max_trial_len = 0
    trial_lens = []
    for i, (start_ind, stop_ind) in enumerate(zip(start_trial_inds, end_trial_inds)):
        start_sample = int(events_tsv["sample"][start_ind])
        stop_sample = int(events_tsv["sample"][stop_ind])
        trial_sample_len = stop_sample - start_sample
        trial_lens.append(trial_sample_len)
        max_trial_len = max(max_trial_len, trial_sample_len)

    if any(max_trial_len > x for x in trial_lens):
        raise RuntimeError(
            f"Trial length between {start_trial_type} and {stop_trial_type} "
            f"is too strict. If we hard code a max trial length of {max_trial_len} "
            f"samples, then we will cut into other trials. Please choose other "
            f"start and stop keywords."
        )

    return max_trial_len


def _get_bad_chs(bids_path):
    # get the channel anat dict
    ch_anat_dict = _read_ch_anat(bids_path)

    # get bad channels from anatomy
    bads = []
    for ch_name, anat in ch_anat_dict.items():
        if (
            anat in ["out", "white matter", "cerebrospinal fluid"]
            or "ventricle" in anat
        ):
            bads.append(ch_name)
    return bads


def read_trial(bids_path, trial_id, picks=None):
    """Read Raw from specific trial id."""
    raw = read_raw_bids(bids_path)

    bads = _get_bad_chs(bids_path)
    raw.info["bads"].extend(bads)

    if picks is None:
        good_chs = [ch for ch in raw.ch_names if ch not in raw.info["bads"]]
    else:
        good_chs = [
            ch for ch in raw.ch_names if ch in picks and ch not in raw.info["bads"]
        ]

    # get trial information
    behav, events = map(pd.DataFrame, get_trial_info(bids_path))
    behav[["trial_id", "successful_trial_flag"]] = behav[
        ["trial_id", "successful_trial_flag"]
    ].apply(pd.to_numeric)
    events["sample"] = pd.to_numeric(events["sample"])

    # collect all rows with trial_type of "start trial" or "end trial"
    start_trial_events = events[events.trial_type == "Reserved (Start Trial)"]
    end_trial_events = events[events.trial_type == "Reserved (End Trial)"]

    # number of trials should match
    assert len(start_trial_events) == len(end_trial_events)
    assert len(start_trial_events) == len(behav.trial_id)

    # get "start trial" row and "end trial" row for specified trial_id
    trial_metadata = behav[behav.trial_id == trial_id]

    if len(trial_metadata) < 1:
        raise ValueError(f"Did not find info for given trial_id {trial_id}.")

    trial_ind = trial_metadata.index
    start_trial_event = start_trial_events.iloc[trial_ind]
    end_trial_event = end_trial_events.iloc[trial_ind]

    # get all the events for specified trial_id and convert to OrderedDict
    trial_events = events.iloc[
        start_trial_event.index[0] : end_trial_event.index[0] + 1
    ]
    events_tsv = trial_events.to_dict(orient="list", into=OrderedDict)

    # get start and stop sampling indices of specified trial
    start = start_trial_event["sample"].iloc[0]
    stop = end_trial_event["sample"].iloc[0]

    # load the data for this trial
    rawdata, times = raw.get_data(
        picks=good_chs, start=start, stop=stop, return_times=True
    )

    return rawdata, times, events_tsv


def read_label(bids_path, trial_id=None, label_keyword="bet_amount"):
    """Read trial's label"""
    # get trial information
    behav_tsv, _ = get_trial_info(bids_path)
    behav = pd.DataFrame(behav_tsv)
    behav[["trial_id", "successful_trial_flag"]] = behav[
        ["trial_id", "successful_trial_flag"]
    ].apply(pd.to_numeric)

    trial_ids = behav.trial_id
    trial_id = None
    label_keyword = "target_direction"

    if trial_id is not None:
        # info for specified trial_id from behav tsv
        trial_metadata = behav[behav.trial_id == trial_id]
        y = trial_metadata[label_keyword]
    else:
        # info for all trials from behav tsv
        y = behav[label_keyword]

    # successful trial indices
    successful_trial_inds = behav.successful_trial_flag == 1

    y = y[successful_trial_inds]
    y = np.array(y).astype(float)

    trial_ids = trial_ids[successful_trial_inds]
    trial_ids = np.array(trial_ids).astype(int)

    return y, trial_ids


def read_dataset(
    bids_path,
    kind="ieeg",
    tmin=-0.2,
    tmax=0.5,
    event_key="show card",
    notch_filter=True,
    verbose=True,
):
    """Read entire dataset as an Epoch."""

    # read in the dataset from mnebids
    raw = read_raw_bids(bids_path)

    # append bad channels
    bads = _get_bad_chs(bids_path)
    raw.info["bads"].extend(bads)

    # only keep SEEG chs
    raw = raw.pick_types(meg=False, seeg=True)

    # filter 60 Hz and harmonics
    raw.load_data()
    fs = raw.info["sfreq"]
    raw = raw.notch_filter(np.arange(60, fs / 2, 60))

    # get the events and events id structure
    events, event_id = mne.events_from_annotations(raw, verbose=verbose)
    event_id = event_id[event_key]  # Change time locked event

    # get the epochs
    epochs = mne.Epochs(
        raw, events, event_id, tmin=tmin, tmax=tmax, baseline=None, verbose=verbose
    )

    return epochs


def cv_fit(
    clf,
    X,
    y,
    num_trials=1,
    apply_grid=False,
    cv_type="KFold",
    shuffle=False,
    n_splits=10,
    seed=1234,
):
    """Cross-validation experiments.

    Allows:

     i) either a grid-search (randomized), or fixed parameters,
     ii) different types of cross-validation folding
     iii) loop over many times (num_trials defaults to 1 though)
     iv) shuffling of the dataset, if we want to remove ordering of trials
     v) control of splitting

    Applies a nested cross-validation that performs bootstrapping to estimate the actual held-out
    data error.

    Parameters
    ----------
    clf :
    X :
    y :
    num_trials :
    apply_grid :
    cv_type :
    shuffle :
    n_splits :
    seed :

    Returns
    -------

    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    from sklearn.model_selection import (
        KFold,
        StratifiedKFold,
        train_test_split,
        TimeSeriesSplit,
    )
    from sklearn.model_selection import cross_val_score, RandomizedSearchCV

    # seed things
    np.random.seed(seed)

    if cv_type == "KFold":
        cv_func = KFold
    elif cv_type == "StratifiedKFold":
        cv_func = StratifiedKFold
    elif cv_type == "TimeSeriesSplit":
        cv_func = TimeSeriesSplit
    else:
        cv_func = train_test_split

    # create groups along the trial ids
    groups = np.hstack([[ii] * (len(y) // 10) for ii in range(10)])
    if len(groups) < len(y):
        groups = np.concatenate((groups, np.tile(groups[-1], (len(y) - len(groups)))))

    """ Perform CV both nested and non-nested."""
    # Arrays to store scores
    non_nested_scores = np.zeros(num_trials)
    nested_scores = np.zeros(num_trials)

    if apply_grid:
        # parameter grid
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=5)]
        # Number of features to consider at every split
        max_features = ["auto", "sqrt", "log2"]

        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=5)]
        max_depth.append(None)

        # Minimum number of samples required to split a node
        min_samples_split = [1, 2, 5, 10]

        # For RERF
        patch_height_min = [2, 3, 4, 5, 10]
        patch_width_min = [1, 5, 10, 20, 30, 40, 50, 100, 500]
        patch_height_max = [2, 3, 4, 5, 10, 15]
        patch_width_max = [10, 20, 30, 40, 50, 100, 500]

        # number of iterations to RandomSearchCV
        n_iter = 100
    else:
        n_estimators = [200]
        max_features = ["auto"]
        max_depth = [None]
        min_samples_split = [2]

        # For RERF
        patch_height_min = [2]
        patch_width_min = [20]
        patch_height_max = [10]
        patch_width_max = [50]

        # number of iterations to RandomSearchCV
        n_iter = 1

    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        # 'patch_height_min': patch_height_min,
        # 'patch_width_min': patch_width_min,
        # 'patch_height_max': patch_height_max,
        # 'patch_width_max': patch_width_max,
    }
    pprint(random_grid)

    # loop over number of trials
    for itrial in range(num_trials):
        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
        inner_cv = cv_func(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        outer_cv = cv_func(n_splits=n_splits, shuffle=shuffle, random_state=seed)

        # Non_nested parameter search and scoring
        # clf = GridSearchCV(estimator=clf, param_grid=random_grid, cv=inner_cv)
        cv_clf = RandomizedSearchCV(
            estimator=clf,
            param_distributions=random_grid,
            n_iter=n_iter,
            cv=inner_cv,
            verbose=1,
            n_jobs=-1,
        )
        print(X.shape, y.shape, groups.shape)
        cv_clf.fit(X, y, groups=groups)
        non_nested_scores[itrial] = cv_clf.best_score_

        # store metrics
        tpr, fpr, aucs, thresholds = dict(), dict(), dict(), dict()
        test_inds = dict()

        # store ROC metrics
        for i, (train, test) in enumerate(outer_cv.split(X=X, y=y, groups=groups)):
            # for binary classification get probability for class 1
            ypredict_proba = cv_clf.predict_proba(X[test])[:, 1]
            ytest = y[test]

            # print(ytest.shape, ypredict_proba.shape)
            # print(ypredict_proba)
            # print(ytest)

            # compute the curve and AUC
            fpr[i], tpr[i], thresholds[i] = roc_curve(
                y_true=ytest,
                # pos_label=1,
                y_score=ypredict_proba,
            )
            aucs[i] = roc_curve(y_true=ytest, y_score=ypredict_proba)
            test_inds[i] = test

        # Nested CV with parameter optimization
        nested_score = cross_val_score(cv_clf, X=X, y=y, cv=outer_cv)
        nested_scores[itrial] = nested_score.mean()

        # ypredict_proba = cross_val_predict(cv_clf, X=X, y=y, groups=groups, cv=outer_cv,
        #                                    n_jobs=-1, verbose=1, method='predict_proba')

        # save files
        fpath = f"./{itrial}trial_metrics_roc.json"
        with open(fpath, mode="w") as fout:
            json.dump(
                {
                    "test_inds": test_inds,
                    "tpr": tpr,
                    "fpr": fpr,
                    "thresholds": thresholds,
                },
                fout,
                cls=NumpyEncoder,
            )

        roc_df = pd.DataFrame(
            [tpr, fpr, thresholds], columns=["tpr", "fpr", "thresholds"]
        )
        print(roc_df)

    # show the differences between nested/non-nested
    score_difference = non_nested_scores - nested_scores
    print(
        "Average difference of {:6f} with std. dev. of {:6f}.".format(
            score_difference.mean(), score_difference.std()
        )
    )

    # Plot scores on each trial for nested and non-nested CV
    # can comment out if you don't need
    if num_trials > 1:
        plt.figure()
        plt.subplot(211)
        (non_nested_scores_line,) = plt.plot(non_nested_scores, color="r")
        (nested_line,) = plt.plot(nested_scores, color="b")
        plt.ylabel("score", fontsize="14")
        plt.legend(
            [non_nested_scores_line, nested_line],
            ["Non-Nested CV", "Nested CV"],
            bbox_to_anchor=(0, 0.4, 0.5, 0),
        )
        plt.title(
            "Non-Nested and Nested Cross Validation on Iris Dataset",
            x=0.5,
            y=1.1,
            fontsize="15",
        )

        # Plot bar chart of the difference.
        plt.subplot(212)
        difference_plot = plt.bar(range(num_trials), score_difference)
        plt.xlabel("Individual Trial #")
        plt.legend(
            [difference_plot],
            ["Non-Nested CV - Nested CV Score"],
            bbox_to_anchor=(0, 1, 0.8, 0),
        )
        plt.ylabel("score difference", fontsize="14")
        plt.show()

    # save results
    # ReRF probably can't get pickled I believe.
    try:
        joblib.dump(
            cv_clf.best_estimator_,
            f"./train_{num_trials}trials_{cv_type}.pkl",
            compress=1,
        )
    except Exception as e:
        print(e)

    print(cv_clf.best_params_)
    # at the very least save the best parameters as a json
    with open(f"./train_{num_trials}trials_{cv_type}.json", "w") as fout:
        json.dump(cv_clf.best_params_, fout)

    return cv_clf


def run_exp(epochs_data, y):
    # obtain the X data
    n_trials, image_height, image_width = epochs_data.shape
    X = epochs_data.reshape(n_trials, -1)
    print(X.shape)
    print("'Image' parameters of the multivariate TS raw data: ")
    print(f"{image_height} X {image_width}")
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier()

    # make sure y labels are properly binarized
    lb = LabelBinarizer(neg_label=0, pos_label=1)
    y = lb.fit_transform(y)
    print(y)

    cv_clf = cv_fit(
        clf, X, y, num_trials=1, cv_type="KFold", shuffle=False, n_splits=2, seed=1234
    )


if __name__ == "__main__":
    from pathlib import Path
    from mne_bids import make_bids_basename

    # bids identifiers
    bids_root = Path("/Users/adam2392/Dropbox/efri/")
    # bids_root = Path("/workspaces/research/data/efri/")
    # bids_root = Path("/Users/ChesterHuynh/OneDrive - Johns Hopkins/research/data/efri/")
    deriv_path = Path(bids_root / "derivatives")

    # subject identifiers
    subject = "efri06"
    session = "efri"
    task = "war"
    acquisition = "seeg"
    run = "01"

    kind = "ieeg"
    trial_id = 2

    # bids filename
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        acquisition=acquisition,
        run=run,
        suffix=kind,
        extension=".vhdr",
    )

    # read a specific trial
    rawdata, times, events_tsv = read_trial(bids_path, trial_id)

    # get the label of this trial
    y, trial_ids = read_label(bids_path, trial_id=None, label_keyword="bet_amount")
    unsuccessful_trial_inds = np.where(np.isnan(y))[
        0
    ]  # get unsuccessful trials based on keyword label
    y = np.delete(y, unsuccessful_trial_inds)

    # read dataset as an epoch
    epochs = read_dataset(bids_path)  # N x C x T' flatten to # N x (C x T')
    epochs = epochs.drop(unsuccessful_trial_inds)
    epochs.load_data()
    epochs_data = epochs.get_data()

    print("Selected events in events.tsv: ", epochs.selection)
    print("Number of epochs: ", len(epochs))
    print("Data shape: ", epochs_data.shape)

    print(len(y))
    print("Unique labels (e.g. bet amounts): ", np.unique(y))

    print("Specific trial: ", rawdata.shape)
    print("Events data structure for specific trial: ", events_tsv)
    print(bids_path)
    print(times[0:5])

    run_exp(epochs_data, y)
