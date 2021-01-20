from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import AxesGrid
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

label_names = {0: "Down", 1: "Right", 2: "Up", 3: "Left"}
colors = cycle(["#26A7FF", "#7828FD", "#FF5126", "#FDF028"])
# colors = cycle(plt.cm.coolwarm(np.linspace(0,1,4)))
# colors = cycle(['#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF', '#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD'])
# colors = cycle(['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499', '#DDDDDD'])
# colors = cycle(['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB'])
# colors = cycle([['EE7733', '0077BB', '33BBEE', 'EE3377', 'CC3311', '009988', 'BBBBBB']])

plt.style.use(["science", "ieee", "no-latex"])


def plot_event_durations(behav, events, jitter=0.025, ax=None):
    """
    docstring
    """
    if not isinstance(behav, pd.DataFrame):
        behav = pd.DataFrame(behav)

    if not isinstance(events, pd.DataFrame):
        events = pd.DataFrame(events)
    
    if ax is None:
        ax = plt.gca()

    # Convert column to numeric dtype
    events["onset"] = pd.to_numeric(events["onset"])

    # Get difference between Left Target onset and its preceding and succeeding events
    inds = (events.trial_type == "Left Target")
    go_cue_duration = events["onset"].diff(periods=-1).abs()[inds]
    left_target_duration = events["onset"].diff(periods=1)[inds]

    df = pd.DataFrame({'"Go Cue" duration': go_cue_duration,
                    '"Left Target" duration': left_target_duration})

    # Plot stripplot with some jitter in the x-coordinate
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), index=df.index, columns=df.columns)
    df_x_jitter += np.arange(len(df.columns))

    for col in df:
        ax.plot(df_x_jitter[col], df[col], 'o', alpha=.40, zorder=1, ms=8, mew=1)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_xlim(-0.5,len(df.columns)-0.5)
    ax.set_ylim(-0.5, 2.5)

    for idx in df.index:
        ax.plot(
            df_x_jitter.loc[idx,['"Go Cue" duration','"Left Target" duration']], 
            df.loc[idx,['"Go Cue" duration','"Left Target" duration']], 
            color = 'grey', linewidth = 0.5, alpha=0.75, linestyle = '--', zorder=-1
        )

    return ax


def _mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = a.shape[0]
    m, se = np.mean(a, axis=0), stats.sem(a, axis=0)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def _plot_signal(t, data, title="", ax=None, label="", ls="-", **plt_kwargs):
    if ax is None:
        ax = plt.gca()

    avg_signal, lower_bound, upper_bound = _mean_confidence_interval(data)

    sns.lineplot(x=t, y=avg_signal, ax=ax, ls=ls, **plt_kwargs)
    ax.fill_between(t, lower_bound, upper_bound, alpha=0.25, label=label, **plt_kwargs)

    return ax


def plot_signals(epochs, labels, ncols=4, axs=None):
    nchs = len(epochs.ch_names)
    t = epochs.times

    if axs is None:
        fig, axs = plt.subplots(
            dpi=200, nrows=int(np.ceil(nchs / ncols)), ncols=ncols, figsize=(15, 45)
        )

    for i, ch in enumerate(epochs.ch_names):
        ax = axs[i]

        epochs_data = epochs.get_data()
        data = epochs_data[:, i]

        # for each class label
        for j, (label, color) in enumerate(zip(np.unique(labels), colors)):
            if label in [1, 3]:
                continue

            _plot_signal(
                t,
                data[labels == label],
                ax=ax,
                label=f"{label_names[label]}",
                color=color,
            )

            ax.legend()
            ax.set(title=f"{ch}", xlabel="Time (s)", ylabel="LFP (mV)")

        # for each class label
        # for j, label in enumerate(np.unique(labels)):
        #     if label in [1, 3]:
        #         continue

        #     _plot_signal(
        #         t,
        #         data[labels == label],
        #         ax=ax,
        #         label=f"{label_names[label]}",
        #     )

        #     ax.legend()
        #     ax.set(title=f"{ch}", xlabel="Time (s)", ylabel="LFP (mV)")

    return axs


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


def plot_roc_multiclass_cv(y_pred_probas, X, y, test_inds, ax=None):

    if ax is None:
        ax = plt.gca()

    fprs = []
    tprs = []
    aucs = []

    # Compute ROC metrics for each fold
    for i, (y_pred_proba, test) in enumerate(zip(y_pred_probas, test_inds)):
        X_test, y_test = X[test], y[test]
        n_classes = len(np.unique(y_test))

        # y_score = estimator.predict_proba(X_test)

        y_pred_proba = np.array(y_pred_proba)

        # Compute ROC metrics for each class
        # fpr, tpr, roc_auc = _compute_roc_multiclass(y_test, y_score, n_classes)
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
            label=r"{label_name}: Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})".format(
                label_name=label_names[i], mean_auc=mean_auc, std_auc=std_auc
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

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    return ax


def plot_feature_importances(result, ch_names, times, image_height, image_width, ax=None):
    nchs = len(ch_names)
    nsteps = len(times)

    if ax is None:
        ax = plt.gca()

    feat_importance_means = np.array(result["importances_mean"]).reshape(image_height, image_width)
    feat_importance_stds = np.array(result["importances_std"]).reshape(image_height, image_width)

    df_feat_importances = pd.DataFrame(feat_importance_means)

    ax = sns.heatmap(
        df_feat_importances,
        vmin=np.min(feat_importance_means),
        vmax=np.max(feat_importance_means),
        center=0.0,
        cmap=plt.cm.coolwarm,
        yticklabels=ch_names,
        ax=ax,
    )

    time_lock = np.where(times == 0)[0][0]

    ax.axvline(time_lock, ls="--")

    ax.set_xticks([0, time_lock, len(times)])
    ax.set_xticklabels(
        [f"{times[0]:.1f}", f"{times[time_lock]:.1f}", f"{times[-1]:.1f}"], rotation=0
    )
    ax.set(xlabel="Time (s)")

    return ax


def plot_roc_cv(y_pred_probas, X, y, test_inds, ax=None):

    if ax is None:
        ax = plt.gca()

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
        label=r"Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})".format(
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

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

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


def plot_accuracies(clf_scores, ax=None):
    if ax is None:
        ax = plt.gca()

    accs = np.array([np.mean(scores["test_accuracy"]) for scores in clf_scores.values()])
    acc_std = np.array([np.std(scores["test_accuracy"]) for scores in clf_scores.values()])

    ax.errorbar(list(clf_scores.keys()), accs, yerr=acc_std, fmt='o', markersize=8, capsize=15)
    ax.axhline(np.mean(clf_scores["MT-MORF"]["test_accuracy"]), lw=1, color='k', ls='--')
    
    return ax


def plot_roc_aucs(clf_scores, ax=None):
    if ax is None:
        ax = plt.gca()

    roc_aucs = np.array([np.mean(scores["test_roc_auc_ovr"]) for scores in clf_scores.values()])
    roc_aucs_std = np.array([np.std(scores["test_roc_auc_ovr"]) for scores in clf_scores.values()])

    ax.errorbar(list(clf_scores.keys()), roc_aucs, yerr=roc_aucs_std, fmt='o', markersize=8, capsize=15)
    ax.axhline(np.mean(clf_scores["MT-MORF"]["test_roc_auc_ovr"]), lw=1, color='k', ls='--')
    
    return ax
