import json
from dataclasses import dataclass
from textwrap import shorten

import mne
import numpy as np
import pandas as pd
from mne import create_info
from mne.io import RawArray
from mne_bids import BIDSPath, read_raw_bids
from mne_bids.read import _handle_events_reading
from mne_bids.tsv_handler import _from_tsv

from mtsmorf.io.move.utils import _preprocess_raw, _preprocess_epochs, _pl, _get_bad_epochs


@dataclass()
class Trial:
    """Trial dataclass for the "move" efri task."""

    trial_idx: int  # index in the behav.tsv
    subject: str
    perturbed: bool
    fixation_trial: bool
    target_direction: bool
    success: bool

    reaction_time: float  # (in seconds; time between go cue and left center)
    correct_speed: bool  # (hard-flag on correct time)
    speed_instruction: float
    beh_fname: str
    event_fname: str

    # marks event markers of when this "trial"
    # starts and ends
    start_trial_id: str = None
    end_trial_id: str = None
    trial_duration: float = None
    avg_speed: float = None
    std_speed: float = None

    def __repr__(self):
        """Taken from mne-python."""
        MAX_WIDTH = 68
        strs = [f"<{self.subject} Move Trial {self.trial_idx} | %s non-empty values"]
        non_empty = 0
        for k, v in self.__dict__.items():
            if k == "ch_names":
                if v:
                    entr = shorten(", ".join(v), MAX_WIDTH, placeholder=" ...")
                else:
                    entr = "[]"  # always show
                    non_empty -= 1  # don't count as non-empty
            elif k in ['index', 'subject']:
                continue
            elif k in ["speed_instruction"]:
                entr = v
            else:
                try:
                    this_len = len(v)
                except TypeError:
                    entr = "{}".format(v) if v is not None else ""
                else:
                    if this_len > 0:
                        entr = "%d item%s (%s)" % (
                            this_len,
                            _pl(this_len),
                            type(v).__name__,
                        )
                    else:
                        entr = ""
            if entr != "":
                non_empty += 1
                strs.append("%s: %s" % (k, entr))
        st = "\n ".join(sorted(strs))
        st += "\n>"
        st %= non_empty
        return st

    def to_data_frame(self):
        df = pd.DataFrame.from_dict(self.__dict__.items(), orient='columns').T
        df.columns = df.iloc[0]

        # set datatype
        df['beh_fname'] = df['beh_fname'].astype(str)
        df['event_fname'] = df['event_fname'].astype(str)

        df.drop(df.index[0], inplace=True)
        return df

    def _add_xy_metrics(self, metrics_df):
        if not self.success or self.perturbed:
            return

        summ_df = metrics_df.groupby("trial_idx")["Speed"].agg([np.nanmean, np.nanstd])
        summ_df.reset_index(inplace=True)

        avg_speed = summ_df[summ_df['trial_idx'] == self.trial_idx]['nanmean']
        std_speed = summ_df[summ_df['trial_idx'] == self.trial_idx]['nanstd']
        self.avg_speed = avg_speed.values[0]
        self.std_speed = std_speed.values[0]
        self.trial_duration = metrics_df[metrics_df['trial_idx'] == self.trial_idx]['trial_duration'].values[0]


def compute_xy_metrics(raw, event_key_start, event_key_end,
                       remove_perturbed=True, verbose: bool = True):
    if (event_key_start is not None and event_key_end is None) \
            or (event_key_end is not None and event_key_start is None):
        raise RuntimeError(f'If event key start/end is set, then '
                           f'the other also must be set.')

    bids_path = raw.filenames[0].copy().update(
        recording=None
    )
    behav_path = bids_path.copy().update(
        suffix='behav', extension='.tsv'
    )
    events_path = bids_path.copy().update(
        suffix='events', extension='.tsv'
    )
    behav_df = pd.read_csv(behav_path, delimiter="\t", index_col=None)

    # apply preprocessing to obtain Epochs
    # get the events and events id structure
    start_key = "Reserved (Start Trial)"
    events, event_id = mne.events_from_annotations(raw)
    tlock_event_start = event_id[event_key_start]  # Change time locked event
    tlock_event_stop = event_id[event_key_end]
    id_start_key = event_id[start_key]

    drop_inds = _get_bad_epochs(behav_df,
                                remove_perturbed=remove_perturbed,
                                remove_unsuccessful=True)

    # indices in event array that Trials start
    start_indices = np.argwhere(events[:, 2] == id_start_key)

    # samples of the start/end to compute xy coordinate
    # metrics of interest
    start_samples = []
    stop_samples = []

    # now let's loop through these
    for trial_idx, start_idx in enumerate(start_indices):
        if trial_idx in drop_inds:
            start_samples.append(None)
            stop_samples.append(None)
            continue

        # get all the events between start of trial and end of trial
        idx = start_idx
        if trial_idx < len(start_indices):
            next_idx = start_indices[trial_idx + 1]
        else:
            next_idx = len(events)

        # move pointer to find start event and stop event
        found_start_sample = None
        found_end_sample = None
        while idx < next_idx:
            if events[idx, 2] == tlock_event_start:
                found_start_sample = int(events[idx, 0])
            elif events[idx, 2] == tlock_event_stop:
                found_end_sample = int(events[idx, 0])
            idx += 1
        if (found_start_sample is not None) and (found_end_sample is not None):
            start_samples.append(found_start_sample)
            stop_samples.append(found_end_sample)

    # loop through start/end sample points and compute
    # metrics
    dfs = []
    for idx, (start, stop) in enumerate(zip(start_samples, stop_samples)):
        if start is None or stop is None:
            continue

        # get the data
        df = raw.to_data_frame(start=start, stop=stop,
                               index='time')

        # compute the metrics
        # like speed
        df['Time'] = df.index.asi8
        dist = df.diff()  # .fillna(0.)
        dist['Dist'] = np.sqrt(dist['x'] ** 2 + dist['y'] ** 2)
        dist['Speed'] = dist.Dist / dist.Time / 2000.
        dist['trial_idx'] = idx
        dist['trial_duration'] = (stop - start) / raw.info['sfreq']
        dist.replace([np.inf, -np.inf], np.nan, inplace=True)
        dfs.append(dist)

    return pd.concat(dfs)

    # avg_speed = dist['Speed'].mean()
    # std_speed = dist['Speed'].std()
    # trial_metrics[idx] = {
    #     'avg_speed':
    # }
    # # make it a dict
    # epoch_event_id = {event_key: tlock_event_id}
    # # handle the case we want to double-trials,
    # # then we would pass in multiple event IDs
    # if double_trials:
    #     epoch_event_id.update({'Show Center': event_id['Show Center']})
    #
    # # obtain the trial data structure
    #
    # # get the epochs data structure
    # epochs = mne.Epochs(
    #     raw, events, epoch_event_id,
    #     tmin=tmin, tmax=tmax, baseline=None,
    #     verbose=verbose
    # )
    #
    # # drop unnecessary epochs
    # epochs.load_data()
    # if len(epochs) != len(behav_df['trial_id']):
    #     raise RuntimeError(f'Epochs length {len(epochs)} should match '
    #                        f'number of trials in behav.tsv.')

    # drop_inds = _get_bad_epochs(behav_df, remove_perturbed=remove_perturbed,
    #                             remove_unsuccessful=True)
    # epochs.drop(drop_inds, reason='unsuccessful, or perturbation')


def read_behav_xy_coords(root, subject, run='01'):
    """Read in xy coordinates for EFRI move task.

    Note this data is sampled at the same rate as the
    original EEG data.

    Parameters
    ----------
    root : str
        Root of the BIDS dataset
    subject : str
        subject identifier
    event_key_start :
    event_key_end :
    remove_perturbed :

    Returns
    -------

    """
    session = "efri"
    task = "move"
    recording = "xy"
    acquisition = "seeg"
    datatype = "ieeg"

    xy_physio_fpath = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        recording=recording,
        acquisition=acquisition,
        datatype=datatype,
        suffix="physio",
        extension=".tsv.gz",
        root=root,
        check=False,
    )
    sidecar_json = xy_physio_fpath.copy().update(extension='.json')

    # read in xy coordinate data
    xy_df = pd.read_csv(xy_physio_fpath, delimiter='\t',
                        index_col=None, compression='gzip')

    # read sidecar json
    with open(sidecar_json, 'r', encoding='utf-8') as fin:
        xy_json = json.load(fin)
    sfreq = xy_json['SamplingFrequency']

    ch_names = ['x', 'y']
    x = xy_df['x']
    y = xy_df['y']
    xy_arr = xy_df.to_numpy().T

    info = create_info(ch_names=ch_names, ch_types='bio', sfreq=sfreq)
    raw = RawArray(xy_arr, info=info, verbose=False)
    raw._filenames[0] = xy_physio_fpath

    bids_path = BIDSPath(
        subject=subject, session=session, task=task,
        acquisition=acquisition, datatype=datatype,
        run=run, suffix=datatype,
        root=root)
    behav_path = bids_path.copy().update(
        suffix='behav', extension='.tsv'
    )
    events_path = bids_path.copy().update(
        suffix='events', extension='.tsv'
    )
    behav_df = pd.read_csv(behav_path, delimiter="\t", index_col=None)

    # read in associated events
    raw = _handle_events_reading(events_path, raw=raw)

    return raw


def read_trial_metadata(root, subject, run='01'):
    """Read EFRI move trial behavioral metadata.

    Note: assumes that only 1 dataset ``run``
    for each subject.

    Parameters
    ----------
    root : str
        The root of the BIDS dataset
    subject : str
        The subject to load data from.

    Returns
    -------
    trials : list of Trial
        list of Trial dataclass.
    """
    session = "efri"
    task = "move"
    acquisition = "seeg"
    datatype = "ieeg"
    extension = '.tsv'

    bids_path = BIDSPath(
        subject=subject, session=session, task=task,
        acquisition=acquisition, datatype=datatype,
        run=run, suffix=datatype,
        extension=extension, root=root)

    behav_path = bids_path.copy().update(
        suffix='behav', extension='.tsv'
    )
    events_path = bids_path.copy().update(
        suffix='events', extension='.tsv'
    )

    # read in metadata as dataframes
    events_df = pd.DataFrame.from_dict(_from_tsv(events_path))
    behav_df = pd.read_csv(behav_path, delimiter="\t", index_col=None)

    # preprocess some columns
    behav_df['speed_instruction'] = behav_df['speed_instruction'].map({1. / 3: 'slow', 2. / 3: 'fast'})
    # behav_df['speed_instruction'] = behav_df['speed_instruction'].map({1. / 3: 'slow', 2. / 3: 'fast'})

    # initialize trial data structure
    trials_metadata = []
    for idx, row in behav_df.iterrows():
        # row = row
        # print(row)
        # print(row['force_magnitude'])
        perturbed = row['force_magnitude'] != 0.0
        target_direction = row['target_direction']
        reaction_time = row['reaction_time']
        correct_speed = row['correct_speed_flag']
        speed_instruction = row['speed_instruction']
        success = row['successful_trial_flag']

        # create data structure for Trial
        trial = Trial(subject=subject, trial_idx=idx,
                      success=bool(success),
                      perturbed=perturbed,
                      fixation_trial=False,
                      target_direction=target_direction,
                      reaction_time=reaction_time,
                      speed_instruction=speed_instruction,
                      correct_speed=correct_speed,
                      beh_fname=behav_path,
                      event_fname=events_path)
        trials_metadata.append(trial)
    return trials_metadata


def read_move_trial_epochs(root, subject, run='01',
                           event_key: str = 'Left Target',
                           event_key_end = 'Hit Target',
                           tmin: float = -0.2,
                           tmax: float = 0.5,
                           l_freq: float = 1.,
                           h_freq: float = None,
                           notch_filter: bool = True,
                           resample_rate: int = None,
                           remove_perturbed: bool = True,
                           double_trials: bool = False,
                           intermediate_fpath: str = None,
                           verbose: bool = True):
    """Read move trials.

    Assumes that all subjects only have exactly 1 ``run``
    for a specific task (e.g. move, or war).

    Will remove unsuccessful trials, depicted by the
    ``successful_trial_flag`` inside the behavioral.tsv
    file.

    Parameters
    ----------
    root : str
        The root of the BIDS dataset
    subject : str
        The specified subject to read in.
    event_key : str
        Where to time-lock raw data to get Epochs.
    l_freq : float
        The lower frequency to apply bandpass
    h_freq : float
        The higher frequency to apply bandpass
    notch_filter : bool
        Whether to apply notch filtering.
    resample_rate : int | None
        Whether to resample the Epochs to a new sampling frequency.
    remove_perturbed : bool
        Whether to remove the perturbed trials or not.
    double_trials : bool
        Whether or not to augment the trials when subjects do the
        fixation.
    intermediate_fpath : str | None
        The intermediate file path to save the Epochs to.
        Saves time, but notch and time-locking will have
        occurred already.

    Returns
    -------
    trial_data_list : list
        A list of trial data classes.

    Notes
    -----
    Move trials follow this sequence in events
    for all successful trials that finish:
        - Reserved (Start Trial)
        - Speed Instruction
        - Show Center
        - At Center
        - Go Cue
        - Left Target
        - Hit Target
        - Held Target
        - Speed Feedback
        - Reward / Speed Fail
        - Reserved (End Trial)

    To double trials, we time lock also to
    Show Center, and add the trials there.
    """
    # readin in epoch hyperparameters

    # BIDS Path entities
    session = 'efri'
    task = 'move'
    acquisition = 'seeg'
    datatype = 'ieeg'
    extension = '.vhdr'

    bids_path = BIDSPath(
        subject=subject, session=session, task=task,
        acquisition=acquisition, datatype=datatype,
        run=run, suffix=datatype,
        extension=extension, root=root)

    if verbose:
        print(f'Analyzing bids dataset: {bids_path} '
              f'with tmin={tmin} and tmax={tmax} '
              f'time-locked to {event_key}.')

    behav_path = bids_path.copy().update(
        suffix='behav', extension='.tsv'
    )
    events_path = bids_path.copy().update(
        suffix='events', extension='.tsv'
    )

    # read in metadata as dataframes
    events_df = pd.DataFrame.from_dict(_from_tsv(events_path))
    behav_df = pd.read_csv(behav_path, delimiter="\t",
                           index_col=None)

    if verbose:
        print(f'Loaded in behavioral df: {behav_df.columns}')

    '''Read in raw dataset'''
    raw = read_raw_bids(bids_path)

    # preprocess the raw data
    raw = _preprocess_raw(raw, bids_path, notch_filter=notch_filter)

    # apply preprocessing to obtain Epochs
    # get the events and events id structure
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    tlock_event_id = event_id[event_key]  # Change time locked event

    # make it a dict
    epoch_event_id = {event_key: tlock_event_id}

    # trim the events data structure for only the events we care about
    epoch_events_idx = np.argwhere(events[:, 2] == tlock_event_id)
    epoch_events = events[epoch_events_idx, ...].squeeze()
    # epoch_events = events

    # handle the case we want to double-trials,
    # then we would pass in multiple event IDs
    if double_trials:
        epoch_event_id.update({'Show Center': event_id['Show Center']})

    # load metadata for each epoch
    trials_list = read_trial_metadata(root, subject)

    # get the xy coordinate data and compute metrics from it
    xy_raw = read_behav_xy_coords(root, subject=subject)
    metrics_df = compute_xy_metrics(
        xy_raw, event_key_start=event_key, event_key_end=event_key_end
    )

    # append information about the specific trial endpoints
    # and also the xy metrics to each trial
    for trial in trials_list:
        trial.start_trial_id = event_key
        trial.end_trial_id = event_key_end
        trial._add_xy_metrics(metrics_df)

    # convert entirely to a dataframe
    trials_df_metadata = pd.concat([trial.to_data_frame() for trial in trials_list])
    trials_df_metadata.reset_index(drop=True, inplace=True)
    drop_inds = _get_bad_epochs(behav_df, remove_perturbed=False,
                                remove_unsuccessful=True)

    # drop unsuccessful trial rows
    if len(trials_df_metadata) == len(epoch_events):
        epoch_events = np.delete(epoch_events, drop_inds, axis=0)
    trials_df_metadata.drop(drop_inds, inplace=True)
    trials_df_metadata.reset_index(drop=True, inplace=True)

    # get the epochs data structure
    epochs = mne.Epochs(
        raw, epoch_events, epoch_event_id,
        tmin=tmin, tmax=tmax, baseline=None,
        metadata=trials_df_metadata,
        verbose=verbose
    )
    if remove_perturbed:
        drop_inds = trials_df_metadata.index[trials_df_metadata['perturbed'] == True].tolist()
        print('new dropped indices', drop_inds)

        # drop unnecessary epochs - e.g. perturbed
        epochs.load_data()
        # if len(epochs) != len(behav_df['trial_id']):
        #     raise RuntimeError(f'Epochs length {len(epochs)} should match '
        #                        f'number of trials in behav.tsv.')
        epochs.drop(drop_inds, reason='unsuccessful, or perturbation')

    # now preprocess the epochs
    epochs = _preprocess_epochs(epochs, resample_rate=resample_rate,
                                l_freq=l_freq, h_freq=h_freq)

    if intermediate_fpath:
        # raise NotImplementedError('didnt figure this out yet..')
        epochs.save(intermediate_fpath,
                    overwrite=True)

    return epochs
