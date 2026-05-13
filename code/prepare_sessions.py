"""Prepare session input folders (F_all_array.npy, baselines, metrics, images)
from processed + zdrift-qc data on disk.

Usage:
    python prepare_sessions.py --subject_id 804670 --save_dir /results/runs/0000_first_try
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

from lamf_analysis.code_ocean import capsule_data_utils as cdu, docdb_utils
import aind_ophys_utils.dff as dff_utils
from aind_ophys_utils.signal_utils import noise_std


DATA_DIR = Path("/root/capsule/data")

FILES_REQUIRED = [
    "F_all_array.npy", "timestamps.npy",
    "baseline_long_window_all_array.npy", "baseline_short_window_all_array.npy",
    "dff_short_window_all_array.npy", "dff_long_window_all_array.npy",
    "sczdrift_df_all.csv",
    "F_noise.npy", "F_signal.npy", "F_snr.npy", "F_skewness.npy",
    "bleaching_metric.npy", "sustained_metric.npy",
]


def get_merged_df(subject_id: int) -> pd.DataFrame:
    session_infos   = docdb_utils.get_session_infos_from_docdb(subject_id, filter_test_data=True)
    processed_infos = (docdb_utils.get_processed_data_info(subject_id)
                       .sort_values("long_window")
                       .drop_duplicates(subset=["raw_name"]))
    merged = processed_infos.merge(session_infos, left_on="raw_name",
                                   right_on="raw_asset_name", how="left")
    sczdrift = docdb_utils.get_derived_data_assets(subject_id, "single-cell-zdrift-qc")
    sczdrift = sczdrift.rename(columns={
        "derived_name":     "single_cell_zdrift_derived_name",
        "derived_asset_id": "single_cell_zdrift_derived_asset_id",
    })
    merged = merged.merge(
        sczdrift[["raw_name", "single_cell_zdrift_derived_name",
                  "single_cell_zdrift_derived_asset_id"]],
        on="raw_name", how="left",
    )
    # De-duplicate: keep latest zdrift-qc per session_key
    merged = (merged.sort_values("single_cell_zdrift_derived_name")
                    .drop_duplicates(subset=["session_key"], keep="last")
                    .reset_index(drop=True))
    # Only sessions where both processed and zdrift dirs exist locally
    merged = merged[
        merged["processed_name"].apply(lambda n: (DATA_DIR / n).exists() if pd.notna(n) else False) &
        merged["single_cell_zdrift_derived_name"].apply(lambda n: (DATA_DIR / n).exists() if pd.notna(n) else False)
    ].reset_index(drop=True)
    return merged


def prepare_one_session(row: pd.Series, save_dir: Path) -> None:
    session_key  = row["session_key"]
    sess_dir     = save_dir / session_key
    sess_dir.mkdir(parents=True, exist_ok=True)

    if all((sess_dir / fn).exists() for fn in FILES_REQUIRED):
        print(f"  [{session_key}] already complete — skipping")
        return

    proc_path    = DATA_DIR / row["processed_name"]
    zdrift_path  = DATA_DIR / row["single_cell_zdrift_derived_name"]
    plane_ids    = cdu.get_plane_ids_from_processed_path(proc_path)

    # ---------- images + roi tables ----------
    for plane_id in plane_ids:
        plane_path = proc_path / plane_id
        roi_table  = cdu.get_roi_table_from_plane_path(plane_path)
        mean_img   = cdu.load_projection_image(plane_path, "mean")
        max_img    = cdu.load_projection_image(plane_path, "max")
        np.save(sess_dir / f"{plane_id}_mean_img.npy", mean_img)
        np.save(sess_dir / f"{plane_id}_max_img.npy",  max_img)
        roi_table.to_pickle(sess_dir / f"{plane_id}_roi_table.pkl")

    # ---------- F, baselines, metrics ----------
    F_all_list = []
    dff_short_list = []
    dff_long_list  = []
    bl_short_list  = []
    bl_long_list   = []
    sczdrift_list  = []
    frame_rate     = None

    for plane_id in plane_ids:
        plane_path  = proc_path / plane_id
        roi_table   = cdu.get_roi_table_from_plane_path(plane_path)
        valid_ids   = roi_table.query("valid_roi")["cell_roi_id"].values
        fr          = cdu.get_frame_rate_from_plane_path(plane_path)
        if frame_rate is None:
            frame_rate = fr
        plane_depth = cdu.get_intended_depth(plane_path)

        zdrift_csv  = next((zdrift_path / plane_id).glob("*roi_time_profile.csv"))
        zdrift_df   = pd.read_csv(zdrift_csv)
        z_range_um  = (zdrift_df["matched_plane_index_smoothed"].max()
                       - zdrift_df["matched_plane_index_smoothed"].min()) * 0.75
        max_frame   = zdrift_df["smoothed_frame_index"].max()
        zdrift_last = zdrift_df.query("smoothed_frame_index == @max_frame")[
            ["cell_roi_id", "fractional_change_from_first_frame"]
        ].copy()
        zdrift_last["plane_id"]       = plane_id
        zdrift_last["intended_depth"] = plane_depth
        zdrift_last["z_drift_um"]     = z_range_um

        F            = cdu.load_corrected_fluorescence(plane_path=plane_path)
        F_valid      = F[valid_ids, :]
        dff_short    = cdu.load_dff_from_plane_path(plane_path)
        bl_short     = cdu.get_baseline_traces(plane_path)
        dff_long, bl_long, _ = dff_utils.dff(F_valid, long_window=60 * 30, fs=fr)

        F_all_list.append(F_valid)
        dff_short_list.append(dff_short[valid_ids, :])
        dff_long_list.append(dff_long)
        bl_short_list.append(bl_short[valid_ids, :])
        bl_long_list.append(bl_long)
        sczdrift_list.append(zdrift_last)

    F_all    = np.concatenate(F_all_list,    axis=0)
    dff_sh   = np.concatenate(dff_short_list, axis=0)
    dff_lg   = np.concatenate(dff_long_list,  axis=0)
    bl_sh    = np.concatenate(bl_short_list,  axis=0)
    bl_lg    = np.concatenate(bl_long_list,   axis=0)
    scz_df   = pd.concat(sczdrift_list, ignore_index=True)

    timestamps      = np.arange(F_all.shape[1]) / frame_rate
    bleach_win      = int(frame_rate * 60 * 5)
    bl_diff         = bl_sh - bl_lg
    bl_diff_z       = bl_diff / np.std(bl_diff, axis=1, keepdims=True)
    bleaching_metric = np.mean(bl_diff_z[:, :bleach_win], axis=1)
    sustained_metric = np.mean(bl_diff_z[:, bleach_win * 2:], axis=1)

    F_noise_arr   = noise_std(F_all, "mad")
    F_signal_arr  = np.percentile(F_all - bl_sh, 99, axis=1)
    F_snr_arr     = F_signal_arr / F_noise_arr
    F_skewness_arr = skew(F_all, axis=1)

    np.save(sess_dir / "F_all_array.npy",                   F_all.astype(np.float32))
    np.save(sess_dir / "timestamps.npy",                     timestamps.astype(np.float32))
    np.save(sess_dir / "baseline_short_window_all_array.npy", bl_sh.astype(np.float32))
    np.save(sess_dir / "baseline_long_window_all_array.npy",  bl_lg.astype(np.float32))
    np.save(sess_dir / "dff_short_window_all_array.npy",     dff_sh.astype(np.float32))
    np.save(sess_dir / "dff_long_window_all_array.npy",      dff_lg.astype(np.float32))
    np.save(sess_dir / "bleaching_metric.npy",               bleaching_metric.astype(np.float32))
    np.save(sess_dir / "sustained_metric.npy",               sustained_metric.astype(np.float32))
    np.save(sess_dir / "F_noise.npy",                        F_noise_arr.astype(np.float32))
    np.save(sess_dir / "F_signal.npy",                       F_signal_arr.astype(np.float32))
    np.save(sess_dir / "F_snr.npy",                          F_snr_arr.astype(np.float32))
    np.save(sess_dir / "F_skewness.npy",                     F_skewness_arr.astype(np.float32))
    scz_df.to_csv(sess_dir / "sczdrift_df_all.csv", index=False)

    print(f"  [{session_key}] done — {F_all.shape[0]} ROIs × {F_all.shape[1]} frames")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject_id", type=int, required=True)
    p.add_argument("--save_dir",   type=Path, default=Path("/results/runs/0000_first_try"))
    args = p.parse_args()

    df = get_merged_df(args.subject_id)
    print(f"Preparing {len(df)} sessions for subject {args.subject_id} → {args.save_dir}")
    for _, row in df.iterrows():
        print(f"\nProcessing {row['session_key']} ...")
        try:
            prepare_one_session(row, args.save_dir)
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
