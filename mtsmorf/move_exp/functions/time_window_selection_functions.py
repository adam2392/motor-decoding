import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import sys
from matplotlib import pyplot as plt
from pathlib import Path
from ptitprince import PtitPrince as pt
from sklearn.utils import check_random_state

if not str(Path(__file__).parents[3]) in sys.path:
    sys.path.append(str(Path(__file__).parents[3]))

from mtsmorf.move_exp.cv import fit_classifiers_cv
from mtsmorf.move_exp.functions.move_experiment_functions import get_preprocessed_labels, get_event_data

from mtsmorf.io.read import get_trial_info_pd, get_unperturbed_trial_inds


label_names = {0: "Down", 1: "Right", 2: "Up", 3: "Left"}


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

    estimates = np.empty(
        num_bootstraps,
    )

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
    unperturbed_trial_inds = get_unperturbed_trial_inds(behav)
    durations = durations.iloc[unperturbed_trial_inds]

    return durations


def plot_event_durations(bids_path, jitter=0.025, ax=None, random_state=None):
    """
    docstring
    """
    rng = check_random_state(random_state)

    if ax is None:
        ax = plt.gca()

    subject = bids_path.subject

    # Compute durations for go cue and left target events
    go_cue_duration = get_event_durations(
        bids_path, event_key="Left Target", periods=-1
    )
    left_target_duration = get_event_durations(
        bids_path, event_key="Left Target", periods=1
    )

    ## Plot stripplot with random jitter in the x-coordinate
    df = pd.DataFrame(
        {
            '"Go Cue" duration': go_cue_duration,
            '"Left Target" duration': left_target_duration,
        }
    )

    df_x_jitter = pd.DataFrame(
        rng.normal(loc=0, scale=jitter, size=df.values.shape),
        index=df.index,
        columns=df.columns,
    )
    df_x_jitter += np.arange(len(df.columns))

    for col in df:
        ax.plot(df_x_jitter[col], df[col], "o", alpha=0.40, zorder=1, ms=8, mew=1)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_xlim(-0.5, len(df.columns) - 0.5)
    ax.set_ylim(-0.5, 2.5)

    for idx in df.index:
        ax.plot(
            df_x_jitter.loc[idx, ['"Go Cue" duration', '"Left Target" duration']],
            df.loc[idx, ['"Go Cue" duration', '"Left Target" duration']],
            color="grey",
            linewidth=0.5,
            alpha=0.75,
            linestyle="--",
            zorder=-1,
        )
    ax.set(ylabel="duration (s)", title=f"{subject.upper()}: Duration of Events")

    return ax


def plot_event_onsets(bids_path, jitter=0.025, ax=None, random_state=None):
    """
    docstring
    """
    rng = check_random_state(random_state)

    subject = bids_path.subject
    behav, events = get_trial_info_pd(bids_path)

    if not isinstance(behav, pd.DataFrame):
        behav = pd.DataFrame(behav)

    if not isinstance(events, pd.DataFrame):
        events = pd.DataFrame(events)

    if ax is None:
        ax = plt.gca()

    ## Convert columns to numeric dtype
    events.onset = pd.to_numeric(events.onset)
    behav[["successful_trial_flag", "force_magnitude"]] = behav[
        ["successful_trial_flag", "force_magnitude"]
    ].apply(pd.to_numeric)

    ## Get onsets for relevant events
    left_target_inds = events.index[events.trial_type == "Left Target"]

    go_cue_onset = events.onset.iloc[left_target_inds - 1]
    go_cue_onset.index = np.arange(len(go_cue_onset))
    left_target_onset = events.onset.iloc[left_target_inds]
    left_target_onset.index = np.arange(len(left_target_onset))
    hit_target_onset = events.onset.iloc[left_target_inds + 1]
    hit_target_onset.index = np.arange(len(hit_target_onset))

    ## Remove unsuccessful and perturbed trials
    successful_trials = behav[behav.successful_trial_flag == 1]
    successful_trials.index = go_cue_onset.index
    perturbed_trial_inds = successful_trials.force_magnitude > 0

    go_cue_onset = go_cue_onset[~perturbed_trial_inds]
    left_target_onset = left_target_onset[~perturbed_trial_inds]
    hit_target_onset = hit_target_onset[~perturbed_trial_inds]

    ## Plot data in strip plot
    df = pd.DataFrame(
        {
            '"Go Cue"': go_cue_onset - go_cue_onset,
            '"Left Target"': left_target_onset - go_cue_onset,
            '"Hit Target"': hit_target_onset - go_cue_onset,
        }
    )

    jitter = 0.025
    df_x_jitter = pd.DataFrame(
        rng.normal(loc=0, scale=jitter, size=df.values.shape),
        index=df.index,
        columns=df.columns,
    )
    df_x_jitter += np.arange(len(df.columns))

    for col in df:
        ax.plot(df_x_jitter[col], df[col], "o", alpha=0.40, zorder=1, ms=8, mew=1)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_xlim(-0.5, len(df.columns) - 0.5)
    ax.set_ylim(-0.5, 4)

    for idx in df.index:
        ax.plot(
            df_x_jitter.loc[idx, ['"Go Cue"', '"Left Target"', '"Hit Target"']],
            df.loc[idx, ['"Go Cue"', '"Left Target"', '"Hit Target"']],
            color="grey",
            linewidth=0.5,
            alpha=0.75,
            linestyle="--",
            zorder=-1,
        )

    ax.set(ylabel="duration (s)", title=f"{subject.upper()}: Duration of Events")

    return ax


def plot_durations_by_label_raincloud(bids_path, ax=None):
    if ax is None:
        ax = plt.gca()

    subject = bids_path.subject

    # compute length of time window for go cue to hit target
    go_cue_durations = get_event_durations(bids_path, event_key="Left Target", periods=-1)
    left_target_durations = get_event_durations(bids_path, event_key="Left Target", periods=1)
    total_durations = go_cue_durations + left_target_durations

    labels = get_preprocessed_labels(bids_path, label_keyword="target_direction")
    durations_df = pd.DataFrame(dict(durations=total_durations, labels=labels))
    
    # plot rain clouds
    pt.RainCloud(
        x="labels", y="durations", palette=None, data=durations_df, orient="h", alpha=0.6, ax=ax
    )
    ax.set(
        title=f"{subject.upper()}: 'Go Cue' to 'Hit Target' Durations", 
        xlabel="target direction",
        ylabel="duration (s)", 
        yticklabels=["Down", "Right", "Up", "Left"]
    )

    return ax


def plot_durations_by_label_kde(bids_path, ax=None):
    if ax is None:
        ax = plt.gca()

    subject = bids_path.subject

    # compute length of time window for go cue to hit target
    go_cue_durations = get_event_durations(bids_path, event_key="Left Target", periods=-1)
    left_target_durations = get_event_durations(bids_path, event_key="Left Target", periods=1)
    total_durations = go_cue_durations + left_target_durations

    labels = get_preprocessed_labels(bids_path, label_keyword="target_direction")
    durations_df = pd.DataFrame(dict(durations=total_durations, labels=labels))

    # plot kde plots
    for label, group in durations_df.groupby('labels'):
        sns.distplot(
            group.durations, 
            hist=False, hist_kws=dict(alpha=0.3), 
            kde=True, kde_kws=dict(fill=True, palette='crest'), 
            label=label_names[label], 
            ax=ax
        )
    
    ax.legend()
    ax.set(
        title=f"{subject.upper()}: 'Go Cue' to 'Hit Target' Durations", 
        xlabel="duration (s)", 
    )


def plot_durations_cv_split(bids_path, cv, ax=None):
    subject = bids_path.subject

    if ax is None:
        ax = plt.gca()

    # compute length of time window for go cue to hit target
    go_cue_durations = get_event_durations(bids_path, event_key="Left Target", periods=-1)
    left_target_durations = get_event_durations(bids_path, event_key="Left Target", periods=1)
    total_durations = go_cue_durations + left_target_durations

    epochs, labels = get_event_data(bids_path)
    epochs_data = epochs.get_data()
    ntrials, nchs, nsteps = epochs_data.shape

    X = epochs_data.reshape(ntrials, -1)
    y = labels

    # Get train and test indices for the first fold only
    *inds, = cv.split(X, y)
    train, test = inds[0]
    is_test = [1 if i in test else 0 for i in range(ntrials)]
    durations_df = pd.DataFrame(dict(durations=total_durations, labels=labels, is_test=is_test))
    
    # plot rain clouds
    pt.RainCloud(
        x="labels", y="durations", hue="is_test", palette=None, data=durations_df, orient="h", alpha=0.6, ax=ax
    )
    ax.set(
        title=f"{subject.upper()}: 'Go Cue' to 'Hit Target' Durations", 
        xlabel="target direction",
        ylabel="duration (s)", 
        yticklabels=["Down", "Right", "Up", "Left"]
    )

    return ax


def fit_classifiers_cv_time_window(bids_path, cv, metrics, time_window_method, random_state=None):
    """docstring."""
    #TODO: Optimize implementation of this function.
    if time_window_method not in ['trial_specific', 'patient_specific']:
        raise ValueError("time_window_method should be one of 'trial_specific' or 'patient_specific'")

    subject = bids_path.subject

    if time_window_method == 'trial_specific':
        go_cue_durations = get_event_durations(bids_path, event_key="Left Target", periods=-1)
        left_target_durations = get_event_durations(bids_path, event_key="Left Target", periods=1)

        tmin = -max(go_cue_durations)
        tmax = max(left_target_durations)

    elif time_window_method == 'patient_specific':
        go_cue_durations = get_event_durations(bids_path, event_key="Left Target", periods=-1)
        left_target_durations = get_event_durations(bids_path, event_key="Left Target", periods=1)
    
    epochs, labels = get_event_data(bids_path, tmin=tmin-0.2, tmax=tmax+0.2)
    epochs_data = epochs.get_data()
    
    ntrials, nchs, nsteps = epochs_data.shape
    print(f"{subject.upper()}: epochs_data.shape = ({epochs_data.shape})")

    t = epochs.times
    mask = (t >= -np.asarray(go_cue_durations)[:, None, None]) \
            & (t <= np.asarray(left_target_durations)[:, None, None])

    masked_data = epochs_data * mask

    X = masked_data.reshape(ntrials, -1)
    y = labels

    image_height = nchs
    image_width = nsteps

    clf_scores = fit_classifiers_cv(X, y, image_height, image_width, cv, metrics, random_state=random_state)
    return clf_scores