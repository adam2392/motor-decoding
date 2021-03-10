from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from matplotlib.patches import Patch
from mne_bids import read_raw_bids
from mpl_toolkits.axes_grid1 import AxesGrid
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils import check_random_state
import dabest
from mne.time_frequency.tfr import AverageTFR, EpochsTFR

label_names = {0: "Down", 1: "Right", 2: "Up", 3: "Left"}
colors = cycle(["#26A7FF", "#7828FD", "#FF5126", "#FDF028"])
# colors = cycle(plt.cm.coolwarm(np.linspace(0,1,4)))
# colors = cycle(['#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF', '#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD'])
# colors = cycle(['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499', '#DDDDDD'])
# colors = cycle(['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB'])
# colors = cycle([['EE7733', '0077BB', '33BBEE', 'EE3377', 'CC3311', '009988', 'BBBBBB']])

try:
    plt.style.use(["science", "ieee", "no-latex"])
except Exception as e:
    print(e)

plt.rcParams["font.family"] = "sans-serif"


def _mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = a.shape[0]
    m, se = np.mean(a, axis=0), stats.sem(a, axis=0)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def _plot_signal(t, data, title="", ax=None, label="", ls="-", **plt_kwargs):
    if ax is None:
        _, ax = plt.subplots()

    avg_signal, lower_bound, upper_bound = _mean_confidence_interval(data)

    sns.lineplot(x=t, y=avg_signal, ax=ax, ls=ls, **plt_kwargs)
    ax.fill_between(t, lower_bound, upper_bound, alpha=0.25, label=label, **plt_kwargs)

    return ax


def plot_signals(epochs, labels, picks=None, axs=None):
    nchs = len(epochs)
    times = epochs.times

    if picks is None:
        picks = epochs.ch_names

    if axs is None:
        nrows = int(np.ceil(len(picks) / 4))
        ncols = min(len(picks), 4)
        width = 8 * ncols
        height = 5 * nrows
        _, axs = plt.subplots(nrows, ncols, dpi=200, figsize=(width, height))
        axs = axs.flatten()

    epochs_data = epochs.get_data(picks)
    for i, ch in enumerate(picks):
        ax = axs[i]
        ch_data = epochs_data[:, i] / 1e6

        # for each class label
        for label, color in zip(np.unique(labels), colors):

            _plot_signal(
                times,
                ch_data[labels == label],
                ax=ax,
                label=f"{label_names[label]}",
                color=color,
            )

            ax.legend(fontsize=10, ncol=2, loc="lower right")
            ax.set(title=f"{ch}", xlabel="Time (s)", ylabel="mV")

    return axs


def plot_epoch_time_series(epochs, picks=None, vmin=None, vmax=None, axs=None):
    """Plot heatmap similar to mne.viz.plot_epochs.

    Args:
        epochs (mne.Epochs): Epochs instance
        picks (list, optional): List of channel names from epochs to use. Defaults to None.
        vmin (int or list, optional): Smallest color value for each heatmap. Defaults to None.
        vmax (int or list, optional): Largest color value for each heatmap. Defaults to None.
        axs (matplotlib.axes.Axes, optional): Axes to plot. Defaults to None.
    """
    if isinstance(vmin, (int, float)):
        vmins = [vmin for _ in range(len(picks))]

    if isinstance(vmax, (int, float)):
        vmaxs = [vmax for _ in range(len(picks))]

    if picks is None:
        picks = epochs.ch_names

    if axs is None:
        nrows = len(picks)
        width = 6
        height = 3 * nrows
        _, axs = plt.subplots(nrows, figsize=(width, height), dpi=200)

    picks_data = epochs.get_data(picks).transpose(1, 0, 2) / 1e6
    ntrials = picks_data.shape[1]
    for ax, pick, data, vmin_, vmax_ in zip(axs, picks, picks_data, vmins, vmaxs):
        sns.heatmap(
            data,
            cmap=plt.cm.coolwarm,
            ax=ax,
            vmin=vmin_,
            vmax=vmax_,
            center=0.0,
            cbar_kws={"label": "mV"}
        )
        ax.invert_yaxis()

        times = epochs.times
        time_lock = np.argmin(np.abs(times))
        xticks = [0, time_lock, len(times) - 1]
        xticklabels = [f"{label:.1f}" for label in times[xticks]]
        yticks = np.arange(0, ntrials, 25)
        yticklabels = yticks + 1

        ax.axvline(x=time_lock, ls="--", c="r", label="'Left Center' onset")
        ax.legend(
            bbox_to_anchor=[0.79, 1], loc="lower center", fontsize=10, frameon=True
        )
        ax.set(
            xlabel="Time (s)",
            ylabel="Trials",
            title=pick,
            xticks=xticks,
            yticks=yticks,
        )
        ax.set_xticklabels(xticklabels, rotation=0),
        ax.set_yticklabels(yticklabels, rotation=0)
    pass


def plot_spectrogram(power, freqs, n_cycles, picks, func="average", db=False, axs=None):
    if picks is None:
        picks = power.ch_names

    if axs is None:
        nrows = int(np.ceil(len(picks) / 4))
        ncols = min(len(picks), 4)
        width = 8 * ncols
        height = 5 * nrows
        _, axs = plt.subplots(nrows, ncols, dpi=100, figsize=(width, height))
        axs = axs.flatten()

    if not isinstance(power, AverageTFR):
        if func == "average":
            avgpower = power.average()
            avgpower_data = avgpower.data
        elif func == "median":
            avgpower_data = np.median(power.data / 1e6, axis=0)
    elif isinstance(power, EpochsTFR):
        avgpower_data = power.data
    else:
        raise TypeError("power must be either an EpochsTFR or AverageTFR instance.")

    ch_names = power.ch_names
    avgpower_data = avgpower.data / 1e6

    times = power.times
    time_lock = np.argmin(np.abs(times))
    xticks = [0, time_lock, len(times) - 1]
    xticklabels = [f"{label:.1f}" for label in times[xticks]]
    yticks = np.arange(0, len(freqs), 2)
    yticklabels = [f"{freq:.1f}" for freq in freqs[yticks]]

    for ax, pick in zip(axs, picks):
        ind = ch_names.index(pick)
        data = avgpower_data[ind]
        if db:
            data = np.log10(data)
        sns.heatmap(data, cmap="viridis", cbar_kws={"label": "dB"}, ax=ax)
        ax.invert_yaxis()
        ax.axvline(x=time_lock, ls="--", c="r", label="'Left Center' onset")
        ax.legend(
            bbox_to_anchor=[0.79, 1], loc="lower center", fontsize=10, frameon=True
        )
        ax.set(
            xlabel="Time (s)",
            ylabel="Frequency (Hz)",
            title=pick,
            xticks=xticks,
            yticks=yticks,
        )
        ax.set_xticklabels(xticklabels, rotation=0),
        ax.set_yticklabels(yticklabels, rotation=0)


def plot_feature_importances(
    result, ch_names, times, image_height, image_width, vmin=None, vmax=None, ax=None
):
    nchs = len(ch_names)
    nsteps = len(times)

    if ax is None:
        _, ax = plt.subplots()

    feat_importance_means = np.array(result["importances_mean"]).reshape(
        image_height, image_width
    )

    df_feat_importances = pd.DataFrame(feat_importance_means)

    if vmin is None:
        vmin = np.min(feat_importance_means)

    if vmax is None:
        vmax = np.max(feat_importance_means)

    ax = sns.heatmap(
        df_feat_importances,
        vmin=vmin,
        vmax=vmax,
        center=0.0,
        cmap=plt.cm.coolwarm,
        yticklabels=ch_names,
        ax=ax,
    )

    time_lock = (np.abs(times)).argmin()

    ax.axvline(time_lock, ls="--")

    ax.set_xticks([0, time_lock, len(times)])
    ax.set_xticklabels(
        [f"{times[0]:.1f}", f"{times[time_lock]:.1f}", f"{times[-1]:.1f}"], rotation=0
    )
    ax.set(xlabel="Time (s)")

    return ax


def plot_roc_cv(y_pred_probas, X, y, test_inds, label="", show_chance=True, ax=None):

    if ax is None:
        _, ax = plt.subplots()

    fprs = []
    tprs = []
    aucs = []

    mean_fpr = np.linspace(0, 1, 100)

    # Compute ROC metrics for each fold
    for i, (y_pred_proba, test) in enumerate(zip(y_pred_probas, test_inds)):
        X_test, y_test = X[test], y[test]
        n_classes = len(np.unique(y_test))

        # y_score = estimator.predict_proba(X_test)
        y_pred_proba = np.array(y_pred_proba)

        # Compute ROC metrics
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0

        fprs.append(mean_fpr)
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

        # fprs.append(fpr)
        # tprs.append(tpr)
        # aucs.append(roc_auc)

    # n_folds x n_classes x 100
    fprs = np.array(fprs)
    tprs = np.array(tprs)
    aucs = np.array(aucs)

    mean_fpr = np.mean(fprs, axis=0)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    # For each class, compute ROC metrics averaged over the folds
    # for i, color in zip(range(n_classes), colors):
    #     mean_fpr = mean_fprs[i]
    #     mean_tpr = mean_tprs[i]
    #     mean_auc = auc(mean_fpr, mean_tpr)
    #     std_auc = np.std(aucs, axis=0)[i]
    #     ax.plot(
    #         mean_fpr,
    #         mean_tpr,
    #         color=color,
    #         label=r"{label_name}: Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})".format(
    #             label_name=label_names[i], mean_auc=mean_auc, std_auc=std_auc
    #         ),
    #     )

    #     std_tpr = np.std(tprs, axis=0)[i]
    #     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    #     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #     ax.fill_between(
    #         mean_fpr,
    #         tprs_lower,
    #         tprs_upper,
    #         color=color,
    #         alpha=0.2,
    #     )

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs, axis=0)
    ax.plot(
        mean_fpr,
        mean_tpr,
        label=label
        + r"Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})".format(
            mean_auc=mean_auc, std_auc=std_auc
        ),
        ls="-",
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        alpha=0.1,
    )

    if show_chance:
        ax.plot(
            [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
        )

    return ax


def _compute_roc_multiclass(y_true, y_score, n_classes):

    fprs = []
    tprs = []
    roc_auc = []

    for i in range(n_classes):
        mean_fpr = np.linspace(0, 1, 100)

        fpr, tpr, _ = roc_curve(
            label_binarize(y_true, classes=np.arange(n_classes))[:, i],
            y_score[:, i],
        )

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0

        fprs.append(mean_fpr)
        tprs.append(interp_tpr)
        roc_auc.append(auc(fprs[i], tprs[i]))

    return fprs, tprs, roc_auc


def plot_roc_multiclass_cv(
    y_pred_probas, X, y, test_inds, label="", show_chance=True, ax=None
):

    if ax is None:
        _, ax = plt.subplots()

    fprs = []
    tprs = []
    aucs = []

    # Compute ROC metrics for each fold
    for i, (y_pred_proba, test) in enumerate(zip(y_pred_probas, test_inds)):
        X_test, y_test = X[test], y[test]
        n_classes = len(np.unique(y_test))
        y_pred_proba = np.array(y_pred_proba)

        # Compute ROC metrics for each class
        fpr, tpr, roc_auc = _compute_roc_multiclass(y_test, y_pred_proba, n_classes)

        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)

    # n_folds x n_classes x 100
    fprs = np.array(fprs)
    tprs = np.array(tprs)
    aucs = np.array(aucs)

    mean_fprs = np.mean(fprs, axis=0)
    mean_tprs = np.mean(tprs, axis=0)
    mean_tprs[:, -1] = 1.0

    # For each class, compute ROC metrics averaged over the folds
    # for i, color in zip(range(n_classes), colors):
    #     mean_fpr = mean_fprs[i]
    #     mean_tpr = mean_tprs[i]
    #     mean_auc = auc(mean_fpr, mean_tpr)
    #     std_auc = np.std(aucs, axis=0)[i]
    #     ax.plot(
    #         mean_fpr,
    #         mean_tpr,
    #         color=color,
    #         label=r"{label_name}: Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})".format(
    #             label_name=label_names[i], mean_auc=mean_auc, std_auc=std_auc
    #         ),
    #     )

    #     std_tpr = np.std(tprs, axis=0)[i]
    #     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    #     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #     ax.fill_between(
    #         mean_fpr,
    #         tprs_lower,
    #         tprs_upper,
    #         color=color,
    #         alpha=0.2,
    #     )
    for i in range(n_classes):
        mean_fpr = mean_fprs[i]
        mean_tpr = mean_tprs[i]
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs, axis=0)[i]
        ax.plot(
            mean_fpr,
            mean_tpr,
            label=r"{label_name}: {clf_name} Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})".format(
                label_name=label_names[i],
                clf_name=label,
                mean_auc=mean_auc,
                std_auc=std_auc,
            ),
            ls="-",
        )

        std_tpr = np.std(tprs, axis=0)[i]
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            alpha=0.1,
        )

    if show_chance:
        ax.plot(
            [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
        )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )
    ax.legend(loc="lower right")

    return ax


def plot_cv_indices(cv, X, y, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object.
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html
    """

    cmap_data = plt.cm.Paired(np.linspace(0, 1, len(np.unique(y))))
    cmap_cv = plt.cm.coolwarm

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes at the end
    class_labels, counts = np.unique(y, return_counts=True)
    accumulated_counts = np.cumsum(counts)
    for i, class_label in enumerate(class_labels):
        if i == 0:
            class_inds = range(counts[i])
        else:
            class_inds = range(
                accumulated_counts[i - 1], counts[i] + accumulated_counts[i - 1]
            )

        ax.scatter(
            class_inds,
            [ii + 1.5] * len(class_inds),
            color=cmap_data[i],
            marker="_",
            lw=lw,
            label=label_names[class_label],
        )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class"]
    ax.set(
        yticks=np.arange(n_splits + 1) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + (0.3 * n_splits), -0.2],
        xlim=[0, len(X)],
    )
    ax.set_title(f"{type(cv).__name__}", fontsize=15)
    return ax


def plot_accuracies(clf_scores, ax=None, random_seed=1):
    if ax is None:
        _, ax = plt.subplots()

    id_col = pd.Series(range(1, len(clf_scores) + 1))

    df = pd.DataFrame(
        {clf_name: scores["test_accuracy"] for clf_name, scores in clf_scores.items()}
    )

    df["ID"] = id_col

    my_data = dabest.load(
        df,
        idx=list(clf_scores.keys()),
        id_col="ID",
        resamples=100,
        random_seed=random_seed,
    )

    my_data.mean_diff.plot(ax=ax)
    ax.set(
        title="Classifier Accuracy Comparison",
    )
    return ax


def plot_roc_aucs(clf_scores, ax=None, random_seed=1):

    if ax is None:
        _, ax = plt.subplots()

    id_col = pd.Series(range(1, len(clf_scores) + 1))

    df = pd.DataFrame(
        {
            clf_name: scores["test_roc_auc_ovr"]
            for clf_name, scores in clf_scores.items()
        }
    )

    df["ID"] = id_col

    my_data = dabest.load(
        df,
        idx=list(clf_scores.keys()),
        id_col="ID",
        resamples=100,
        random_seed=random_seed,
    )

    my_data.mean_diff.plot(ax=ax)
    ax.set(title="Classifier ROC AUC One vs. Rest Comparison")
    return ax


def plot_classifier_performance(clf_scores, X, y, axs=None):

    if axs is None:
        axs = plt.subplots(ncols=2)
        axs = axs.flatten()

    if len(axs) != 2:
        raise ValueError("Axes to be plotted to should have exactly 2 subplots.")


    # 1. Plot roc curves
    plot_roc_aucs(clf_scores, ax=axs[0])

    # 2. Plot accuracies
    plot_accuracies(clf_scores, ax=axs[1])

    return axs