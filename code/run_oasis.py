
import json
import logging
import os
from datetime import datetime as dt
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Union
import h5py
import numpy as np

from oasis.functions import deconvolve
from oasis.oasis_methods import oasisAR1, oasisAR1_f32, oasisAR2


def get_args_params():
    args = {'estimate_parameters': True, 'g': (None,), 's_min': 0.0, 'lam': None, 'tau_rise': 0, 'b': None,
    'optimize_tau': 0, 'sn': None, 'b_nonneg': True, 'penalty': 1, 'tau': None}
    params = args.copy()

    if args['tau'] is None or args['tau_rise'] is None:  # automatically estimate tau
        if args['tau_rise'] == 0:  # negligible rise time -> AR1
            params["g"] = (None,)
        else:  # automatically estimate rise time too -> AR2
            params["g"] = (None, None)
    else:
        if args['tau_rise'] == 0:  # negligible rise time -> AR1
            params["g"] = (np.exp(-1 / (args['tau'] * frame_rate)),)
        else:  # AR2
            d, r = (
                np.exp(-1 / (args['tau'] * frame_rate)),
                np.exp(-1 / (args['tau_rise'] * frame_rate)),
            )
            params["g"] = (d + r, -d * r)

    if not args['estimate_parameters']:
        if args['tau'] is None:
            raise UserWarning(
                "'estimate_parameters' is False, but no value for decay time "
                "constant 'tau' has been provided."
            )
        if args['lam'] is None:
            params["lam"] = 0
            logging.info(
                "'estimate_parameters' is False, but no value for sparsity penalty "
                " 'lam' has been provided,thus automatically setting 'lam' to 0."
            )
        if args['b'] is None:
            params["b"] = 0
            logging.info(
                "'estimate_parameters' is False, but no value for baseline 'b' has been provided, "
                "thus automatically setting 'b' to 0."
            )
    return args, params


def _deconv(t):
    args, params = get_args_params()
    if np.isnan(t).any():  # check if trace has any nans, if so return all nans
        c, s = np.full_like(t, np.nan), np.full_like(t, np.nan)
        return (
            (c, s, np.nan, np.nan, np.nan) if args['estimate_parameters'] else (c, s)
        )
    else:
        if args['estimate_parameters']:
            relevant_params = {
                k: args[k] for k in ["g", "sn", "b", "b_nonneg", "penalty"]
            }
            relevant_params["optimize_g"] = params["optimize_tau"]
            return deconvolve(t, **relevant_params)
        elif args['tau_rise'] == 0:
            return (oasisAR1_f32 if t.dtype == np.float32 else oasisAR1)(
                t - params["b"], args['g'][0], s_min=args['s_min'], lam=args['lam']
            )
        else:
            return oasisAR2(
                t.astype(float) - params["b"],
                args.g[0],
                args.g[1],
                s_min=args['s_min'],
                lam=args['lam'],
            )

def run_oasis(traces, frame_rate, save_fn):
    ''' Run OASIS on traces and save results to h5 files

    Parameters
    ----------
    traces : np.ndarray
        Calcium traces of shape (N, T)
    frame_rate : float
        Frame rate of the recording
    '''

    N, T = traces.shape
    nans = np.where(np.isnan(traces))[0]
    if len(nans) > 0:
        logging.info(f"Traces have nans: {len(nans)} in {experiment_id}")

    args, params = get_args_params()

    # run oasis on each trace in parallel
    if N:
        pool = Pool(int(tmp) if (tmp := os.environ.get("CO_CPUS")) else tmp)
        res = pool.map(_deconv, traces)
        calcium, spikes = [
            np.array([r[i] for r in res], dtype="f4") for i in (0, 1)
        ]
        if args['estimate_parameters']:
            b_hat, g_hat, lam_hat = [
                np.array([r[i] for r in res], dtype="f4") for i in (2, 3, 4)
            ]
            # convert parameters of the auto-regressive (AR) process to time constants
            if g_hat.ndim == 1:  # AR1
                tau_hat = -1 / np.log(g_hat) / frame_rate
            else:  # AR2
                tmp = np.sqrt(g_hat[:, 0] ** 2 + 4 * g_hat[:, 1]) / 2
                tau_hat = -1 / np.log(g_hat[:, 0] / 2 + tmp) / frame_rate
                tau_rise_hat = -1 / np.log(g_hat[:, 0] / 2 - tmp) / frame_rate
    else:  # no ROIs detected
        calcium, spikes = [np.empty((0, T), dtype="f4")] * 2
        if args['estimate_parameters']:
            b_hat, tau_hat, lam_hat = [], [], []
            if args.tau_rise != 0:
                tau_rise_hat = []

    # save to h5
    with h5py.File(save_fn, "w") as f:
        f.create_dataset("events", data=spikes, compression="gzip")
        f.create_dataset("denoised", data=calcium, compression="gzip")
        if args['estimate_parameters']:
            f.create_dataset("b_hat", data=b_hat)
            f.create_dataset("tau_hat", data=tau_hat)
            if args['tau_rise'] != 0:
                f.create_dataset("tau_rise_hat", data=tau_rise_hat)
            f.create_dataset("lam_hat", data=lam_hat)