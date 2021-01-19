import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mne.time_frequency import tfr_morlet
from mne_bids import BIDSPath
from rerf.rerfClassifier import rerfClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Hack-y way to import from files in sibling "io" directory
sys.path.append(str(Path(__file__).parent.parent / "io"))
sys.path.append(str(Path(__file__).parent.parent / "war_exp"))

from read import read_dataset, read_label, read_trial, get_trial_info
from plotting import (
    plot_signals,
    plot_roc_multiclass_cv,
    plot_feature_importances,
    plot_cv_indices,
)
from utils import NumpyEncoder
from experiments import cv_fit

plt.style.use(["science", "ieee", "no-latex"])

RNG = 1


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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def prepare_freq_data(
    epochs, nfreqs=10, l_freq=70, h_freq=200, decim=3, n_jobs=1, return_freqs=False
):
    # Run morlet wavelet transform
    freqs = np.logspace(*np.log10([l_freq, h_freq]), num=nfreqs)
    n_cycles = freqs / 2.0  # different number of cycle per frequency
    power = tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        average=False,
        return_itc=False,
        decim=decim,
        n_jobs=n_jobs,
    )

    # Reshape spectral power data
    power_data = power.data
    ntrials, nchs, nfreqs, nsteps = power_data.shape
    freq_data = power_data.reshape(ntrials, nchs * nfreqs, nsteps)

    # Trim time window to remove edge artifacts in frequency conversion
    times = np.where((power.times >= -0.5) & (power.times <= 1.0))[0]
    freq_data = freq_data[:, :, times]

    if return_freqs:
        return freq_data, power.times, freqs

    return freq_data, power.times


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=str, help="subject ID (e.g. efri02)")
    parser.add_argument(
        "--replot-signals",
        type=str2bool,
        default=True,
        help="whether to re-plot signals",
    )
    parser.add_argument(
        "--rerun-fit",
        type=str2bool,
        default=True,
        help="whether to rerun model fitting",
    )
    parser.add_argument(
        "--feat-importances",
        type=str2bool,
        default=True,
        help="whether to compute feature importances",
    )
    args = parser.parse_args()

    tmin, tmax = (-0.75, 1.25)

    #### TODO: change file paths
    bids_root = Path("/workspaces/research/mnt/data/efri/")
    derivatives_path = (
        bids_root
        / "derivatives"
        / "preprocessed"
        / f"tmin={tmin}-tmax={tmax}"
        / "low-pass=1000Hz-downsample=500"
    )
    # results_path = Path("/workspaces/research/seeg localization/SPORF/mtsmorf/results")
    results_path = Path(
        "/workspaces/research/efri OneDrive/Adam Li - efri/derivatives/workstation_output"
    )

    # new directory paths for outputs and inputs at Hackerman workstation
    # bids_root = Path("/home/adam2392/hdd/Dropbox/efri/")
    # results_path = bids_root / "derivatives" / "raw" / "mtsmorf" / "results"

    subject = args.subject

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
    ntrials, nchs, nsteps = epochs.get_data().shape

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

    # plot raw lfps
    if args.replot_signals is True:
        fig, axs = plt.subplots(
            dpi=200, nrows=int(np.ceil(nchs / 4)), ncols=4, figsize=(15, 45)
        )
        axs = axs.flatten()
        plot_signals(epochs, labels, axs=axs)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f"{subject.upper()}", fontsize=24)
        plt.savefig(results_path / f"{subject}/{subject}_raw_eeg_updown.png")
        print(
            f"Signals plot saved at {results_path}/{subject}/{subject}_raw_eeg_updown.png"
        )

        plt.close(fig)

    nfreqs = 10
    freq_data, power_times, freqs = prepare_freq_data(
        epochs,
        nfreqs=nfreqs,
        l_freq=70,
        h_freq=200,
        decim=3,
        n_jobs=1,
        return_freqs=True,
    )

    # Extract X and y matrices
    included_trials = np.isin(labels, [0, 1, 2, 3])
    X = freq_data[included_trials].reshape(np.sum(included_trials), -1)
    y = labels[included_trials]

    # check there are equal number of trials and labels
    assert ntrials == labels.shape[0], "Unequal number of trials and labels"

    # Initialize classifiers
    mtsmorf = rerfClassifier(
        projection_matrix="MT-MORF",
        max_features="auto",
        n_jobs=-1,
        random_state=RNG,
        image_height=freq_data.shape[1],
        image_width=freq_data.shape[2],
    )
    srerf = rerfClassifier(
        projection_matrix="S-RerF",
        max_features="auto",
        n_jobs=-1,
        random_state=RNG,
        image_height=freq_data.shape[1],
        image_width=freq_data.shape[2],
    )
    lr = LogisticRegression(random_state=RNG)
    rf = RandomForestClassifier(random_state=RNG)
    dummy = DummyClassifier(strategy="most_frequent", random_state=RNG)

    clfs = [
        mtsmorf,
        srerf,
        lr,
        rf,
        dummy,
    ]

    estimators = None
    if args.rerun_fit is True:

        metrics = [
            "accuracy",
            "roc_auc_ovr",
        ]

        clf_scores = dict()

        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=False)

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
                cv=kf,
                metrics=metrics,
                n_jobs=None,
                return_train_score=True,
                return_estimator=True,
            )

        fig, ax = plt.subplots(dpi=100, figsize=(8, 6))
        argsort_inds = np.argsort(y)
        plot_cv_indices(kf, X[argsort_inds], y[argsort_inds], ax, n_splits, lw=10)
        plt.savefig(results_path / f"{subject}/{subject}_cv_indices.png")
        plt.close(fig)

        ############### FEATURE IMPORTANCES ################
        if args.feat_importances is True:
            n_repeats = 5  # number of repeats for permutation importance

            clf_name = mtsmorf.get_params()["projection_matrix"]
            scores = clf_scores[clf_name]

            print("Starting feature importances...")

            best_ind = np.argmax(scores["test_accuracy"])
            best_estimator = scores["estimator"][best_ind]
            best_test_inds = scores["test_inds"][best_ind]

            X_test = X[best_test_inds]
            y_test = y[best_test_inds]

            # Run feat importance for Accuracy
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

                result = permutation_importance(
                    best_estimator,
                    X_test,
                    y_test,
                    scoring=scoring_method,
                    n_repeats=n_repeats,
                    n_jobs=1,
                    random_state=RNG,
                )

                imp_std = result.importances_std
                imp_vals = result.importances_mean
                scores[key_mean].append(list(imp_vals))
                scores[key_std].append(list(imp_std))

            clf_scores[clf_name] = scores

        ################ STORING RESULTS ################

        for clf in clfs:

            if clf.__class__.__name__ == "rerfClassifier":
                clf_name = clf.get_params()["projection_matrix"]
            elif clf.__class__.__name__ == "DummyClassifier":
                clf_name = clf.strategy
            else:
                clf_name = clf.__class__.__name__

            scores = clf_scores[clf_name]

            try:
                if "estimator" in scores.keys():
                    estimators = scores["estimator"]
                    del scores["estimator"]
                with open(
                    results_path / f"{subject}/{subject}_{clf_name}_results.json",
                    "w",
                ) as fout:
                    json.dump(scores, fout, cls=NumpyEncoder)
                    print(f"CV results for {clf_name} saved as json.")

            except Exception as e:
                traceback.print_exc()

            scores["estimator"] = estimators
            clf_scores[clf_name] = scores

    ######## Results already exists, so we read directly from JSON ########
    else:
        clf_scores = dict()

        for clf in clfs:
            if clf.__class__.__name__ == "rerfClassifier":
                clf_name = clf.get_params()["projection_matrix"]
            elif clf.__class__.__name__ == "DummyClassifier":
                clf_name = clf.strategy
            else:
                clf_name = clf.__class__.__name__

            with open(
                results_path / f"{subject}/{subject}_{clf_name}_results.json"
            ) as fout:
                scores = json.load(fout)

            clf_scores[clf_name] = scores

    try:
        clf_name = "MT-MORF"
        scoring_method = "roc_auc_ovr"
        result = dict(
            importances_mean=clf_scores[clf_name][
                f"validate_{scoring_method}_imp_mean"
            ],
            importances_std=clf_scores[clf_name][f"validate_{scoring_method}_imp_std"],
        )

        inds = np.where((power_times >= -0.5) & (power_times <= 1.0))[0]
        times = power_times[inds]

        image_height = nchs * nfreqs
        image_width = len(times)

        ## Plots to help summarize heatmap
        feat_importance_means = np.array(result["importances_mean"]).reshape(
            image_height, image_width
        )
        feat_importance_stds = np.array(result["importances_std"]).reshape(
            image_height, image_width
        )

        id_col = pd.Series(range(1, nchs * nfreqs + 1))
        ch_names_rep = np.repeat(epochs.ch_names, nfreqs)
        freqs_tiled = np.tile(freqs, nchs)
        thr = 0.003

        df_feat_importances = pd.DataFrame(
            {
                "ID": id_col,
                "channel": ch_names_rep,
                "frequency": freqs_tiled,
                "count": np.sum(np.absolute(feat_importance_means) >= thr, axis=1),
            }
        )

        fig, ax = plt.subplots(dpi=300, figsize=(8, 16))

        df_feat_importances['count_by_channel'] = df_feat_importances.groupby(['channel'])['count'].transform('sum')
        srtd_df = df_feat_importances.sort_values(by=["count_by_channel"], ascending=False)

        sns.barplot(x="count_by_channel", y="channel", data=srtd_df, ci=None, ax=ax)
        ax.set(title=f"{subject.upper()}: Count of Time Steps with Mean Importance >= {thr} by Channel", ylabel="channel")
        
        plt.savefig(
            results_path
            / f"{subject}/{clf_name}_feature_importances_freqs_{scoring_method}_summary_channels_thr={thr}.png"
        )
        print(f"Feature importance matrix {scoring_method} summary (channels) for {clf_name} saved.")
        plt.close(fig)

        fig, ax = plt.subplots(dpi=300, figsize=(8, 16))
        df_feat_importances['count_by_freq'] = df_feat_importances.groupby(['frequency'])['count'].transform('sum')
        srtd_df = df_feat_importances.sort_values(by=["count_by_freq"], ascending=False)

        sns.barplot(x="count_by_freq", y="frequency", orient="h", data=srtd_df, ci=None, ax=ax)
        ax.set(title=f"{subject.upper()}: Count of Time Steps with Mean Importance >= {thr} by Frequency", ylabel="frequency")
        plt.savefig(
            results_path
            / f"{subject}/{clf_name}_feature_importances_freqs_{scoring_method}_summary_freqs_thr={thr}.png"
        )
        print(f"Feature importance matrix {scoring_method} summary (frequencies) for {clf_name} saved.")
        plt.close(fig)


        id_col = pd.Series(range(1, nchs * nfreqs + 1))
        ch_names_rep = np.repeat(epochs.ch_names, nfreqs)
        freqs_tiled = np.tile(freqs, nchs)

        df_feat_importances = pd.DataFrame(
            {
                "ID": id_col,
                "channel": ch_names_rep,
                "frequency": freqs_tiled,
                "count": np.sum(np.absolute(feat_importance_means) > 0, axis=1),
            }
        )

        fig, ax = plt.subplots(dpi=300, figsize=(8, 16))

        df_feat_importances['count_by_channel'] = df_feat_importances.groupby(['channel'])['count'].transform('sum')
        srtd_df = df_feat_importances.sort_values(by=["count_by_channel"], ascending=False)

        sns.barplot(x="count_by_channel", y="channel", data=srtd_df, ci=None, ax=ax)
        ax.set(title=f"{subject.upper()}: Count of Time Steps with Nonzero Mean Importance by Channel", ylabel="channel")
        
        plt.savefig(
            results_path
            / f"{subject}/{clf_name}_feature_importances_freqs_{scoring_method}_summary_channels_nonzero.png"
        )
        print(f"Feature importance matrix {scoring_method} summary (channels) for {clf_name} saved.")
        plt.close(fig)


        fig, ax = plt.subplots(dpi=300, figsize=(8, 16))
        df_feat_importances['count_by_freq'] = df_feat_importances.groupby(['frequency'])['count'].transform('sum')
        srtd_df = df_feat_importances.sort_values(by=["count_by_freq"], ascending=False)

        sns.barplot(x="count_by_freq", y="frequency", orient="h", data=srtd_df, ci=None, ax=ax)
        ax.set(title=f"{subject.upper()}: Count of Time Steps with Nonzero Mean Importance by Frequency", ylabel="frequency")
        plt.savefig(
            results_path
            / f"{subject}/{clf_name}_feature_importances_freqs_{scoring_method}_summary_freqs_nonzero.png"
        )
        print(f"Feature importance matrix {scoring_method} summary (frequencies) for {clf_name} saved.")
        plt.close(fig)

        fig, ax = plt.subplots(dpi=300, figsize=(20, 80))

        ## Plot full feat importance heatmap
        ch_names = [f"{ch}-{freq:.1f}" for ch in epochs.ch_names for freq in freqs]
        plot_feature_importances(
            result, ch_names, times, nchs * nfreqs, len(times), ax=ax
        )
        ax.set(title=f"{subject.upper()}: Feature Importances {scoring_method}")
        fig.tight_layout()

        plt.savefig(
            results_path
            / f"{subject}/{clf_name}_feature_importances_freqs_{scoring_method}.png"
        )
        print(f"Feature importance matrix {scoring_method} for {clf_name} saved.")
        plt.close(fig)

    except Exception as e:
        traceback.print_exc()

    ######## Plot accuracies
    fig, ax = plt.subplots(dpi=100, figsize=(8, 6))

    accs = []
    acc_std = []
    for clf_name, scores in clf_scores.items():
        accs.append(np.mean(scores["test_accuracy"]))
        acc_std.append(np.std(scores["test_accuracy"]))

    accs = np.array(accs)
    acc_std = np.array(acc_std)

    ax.errorbar(
        list(clf_scores.keys()), accs, yerr=acc_std, fmt="o", markersize=8, capsize=15
    )
    ax.axhline(
        np.mean(clf_scores["MT-MORF"]["test_accuracy"]), lw=1, color="k", ls="--"
    )
    ax.set(ylabel="accuracy", title=f"{subject.upper()}: Accuracy of Classifiers")
    fig.tight_layout()

    plt.savefig(results_path / f"{subject}/{subject}_accuracies.png")
    print(f"Accuracies for {clf_name} saved.")
    plt.close(fig)

    ######## Plot ROC curves ########
    for clf in clfs:

        if clf.__class__.__name__ == "rerfClassifier":
            clf_name = clf.get_params()["projection_matrix"]
        elif clf.__class__.__name__ == "DummyClassifier":
            clf_name = clf.strategy
        else:
            clf_name = clf.__class__.__name__

        with open(
            results_path / f"{subject}/{subject}_{clf_name}_results.json"
        ) as fout:
            scores = json.load(fout)

        ######## Save ROC Curves ########
        fig, ax = plt.subplots(dpi=300, figsize=(12, 8))

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
        fig.tight_layout()

        try:
            plt.savefig(results_path / f"{subject}/{subject}_{clf_name}_roc_curves.png")
            print(f"{subject} ROC curve for {clf_name} saved.")

        except Exception as e:
            traceback.print_exc()

        plt.close(fig)

        ######## Plot confusion matrices for each fold ########
        for i, test_confusion_matrix in enumerate(scores["test_confusion_matrix"]):

            test_acc = np.sum(np.diagonal(test_confusion_matrix)) / np.sum(
                test_confusion_matrix
            )
            df_cm = pd.DataFrame(
                test_confusion_matrix,
                index=["down", "right", "up", "left"],
                columns=["down", "right", "up", "left"],
            )

            fig, ax = plt.subplots(dpi=300, figsize=(8, 6))

            sns.heatmap(df_cm, annot=True, cmap="Blues", ax=ax)
            ax.set(
                xlabel="Predicted label",
                ylabel="True label",
                title=f"{subject.upper()}: {clf_name} (Accuracy = {test_acc:.3f})",
            )

            try:
                plt.savefig(
                    results_path
                    / f"{subject}/{subject}_{clf_name}_confusion_matrix{str(i).zfill(2)}.png"
                )
                print(f"Confusion matrix {i} for {clf_name} saved.")

            except Exception as e:
                traceback.print_exc()

            plt.close(fig)
