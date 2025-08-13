# Mostly copied from:
# https://github.com/Peter-MacCallum-Cancer-Centre/DEEP-PSMA/blob/main/evaluation.py

import os
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import scipy.ndimage
import SimpleITK as sitk
from dotenv import load_dotenv
from rich import print
from rich.progress import track
from rich.table import Table

load_dotenv(override=True)

# Default configuration
INPUT_DIR = os.getenv("INPUT_DIR", "input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} entries from {input_csv}")

    # Resolve inputs / outputs paths
    df["psma_pt_ttb_path"] = df["psma_pt_ttb_path"].apply(lambda path: Path(args.input_dir, path) if path else None)
    df["fdg_pt_ttb_path"] = df["fdg_pt_ttb_path"].apply(lambda path: Path(args.input_dir, path) if path else None)
    df["psma_pred_path"] = df["psma_pred_path"].apply(lambda path: Path(args.output_dir, path) if path else None)
    df["fdg_pred_path"] = df["fdg_pred_path"].apply(lambda path: Path(args.output_dir, path) if path else None)

    results = []
    for _, row in track(df.iterrows(), total=len(df), description="Evaluating..."):
        psma_gt_path = row["psma_pt_ttb_path"]
        psma_pred_path = row["psma_pred_path"]
        fdg_gt_path = row["fdg_pt_ttb_path"]
        fdg_pred_path = row["fdg_pred_path"]

        print(f"[PSMA] Evaluating {psma_gt_path} and {psma_pred_path}")
        psma_scores = compute_scores(psma_gt_path, psma_pred_path) if (psma_gt_path and psma_pred_path) else {}
        print(f"[PSMA] Scores: \t {' | '.join(f'{k}: {v:.4f}' for k, v in psma_scores.items())}")

        print(f"[PSMA] Evaluating {fdg_gt_path} and {fdg_pred_path}")
        fdg_scores = compute_scores(fdg_gt_path, fdg_pred_path) if (fdg_gt_path and fdg_pred_path) else {}
        print(f"[FDG ] Scores: \t {' | '.join(f'{k}: {v:.4f}' for k, v in fdg_scores.items())}")

        result = {
            **{f"psma_{k}": v for k, v in psma_scores.items()},
            **{f"fdg_{k}": v for k, v in fdg_scores.items()},
        }
        results.append(result)

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)

    df_scores = df_results.describe()
    df_scores["metric"] = df_scores.index
    df_scores = df_scores[["metric"] + [col for col in df_scores.columns if col != "metric"]]

    table = Table()
    for col in df_scores.columns:
        table.add_column(col, justify="right", style="cyan" if col != "metric" else "", no_wrap=True)
    for row in df_scores.values:
        table.add_row(*[f"{value:.4f}" if isinstance(value, float) else str(value) for value in row])
    print(table)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Evaluate segmentation results against ground truth labels.")
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help="Directory containing the ground truth labels.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory containing the predicted segmentation results.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path of the CSV format.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path of the output CSV file.",
    )
    return parser.parse_args()


def label_mask(mask: np.ndarray) -> np.ndarray:
    structure = scipy.ndimage.generate_binary_structure(3, 2)
    return scipy.ndimage.label(mask > 0, structure=structure)


def fp_volume_score(gt_array: np.ndarray, pred_array: np.ndarray) -> float:
    pred_label, num_labels = label_mask(pred_array)
    gt_array = gt_array > 0  # Ensure target is binary

    if num_labels == 0:
        return 0.0

    labels = np.arange(1, num_labels + 1)
    overlap_counts = scipy.ndimage.sum_labels(gt_array, pred_label, labels)
    component_sizes = scipy.ndimage.sum_labels(np.ones_like(pred_label), pred_label, labels)
    fp_mask = overlap_counts == 0
    num_fps = component_sizes[fp_mask].sum()
    return num_fps


def fn_volume_score(gt_array: np.ndarray, pred_array: np.ndarray) -> float:
    target_label, num_labels = label_mask(gt_array)
    pred_array = pred_array > 0  # Ensure pred is binary

    if num_labels == 0:
        return 0.0

    labels = np.arange(1, num_labels + 1)
    overlap_counts = scipy.ndimage.sum_labels(pred_array, target_label, labels)
    component_sizes = scipy.ndimage.sum_labels(np.ones_like(target_label), target_label, labels)
    fn_mask = overlap_counts == 0
    num_fns = component_sizes[fn_mask].sum()
    return num_fns


def dice_score(mask1: np.ndarray, mask2: np.ndarray) -> float:
    if mask1.sum() == mask2.sum() == 0:
        return 0.0

    numerator = 2 * (mask1 * mask2).sum()
    denominator = mask1.sum() + mask2.sum()
    return float(numerator / denominator)


def surface_dice_score(gt_image: sitk.Image, pred_image: sitk.Image) -> float:
    distance_map = partial(sitk.SignedMaurerDistanceMap, squaredDistance=False, useImageSpacing=True)

    gt_surface = sitk.LabelContour(gt_image == 1, False)
    pred_surface = sitk.LabelContour(pred_image == 1, False)
    pred_distance_map = sitk.Abs(distance_map(pred_surface))
    gt_to_pred_array = sitk.GetArrayViewFromImage(pred_distance_map)[sitk.GetArrayViewFromImage(gt_surface) == 1]
    matching_surface_voxels = (gt_to_pred_array == 0).sum()
    gold_surface_voxels = (sitk.GetArrayViewFromImage(gt_surface) == 1).sum()
    prediction_surface_voxels = (sitk.GetArrayViewFromImage(pred_surface) == 1).sum()
    surface_dice = (2.0 * matching_surface_voxels) / (gold_surface_voxels + prediction_surface_voxels)
    return surface_dice


def read_label(label_path: Union[str, Path]) -> tuple[np.ndarray, float]:
    label = sitk.ReadImage(label_path)
    ar = sitk.GetArrayFromImage(label)
    voxel_volume = np.prod(np.array(label.GetSpacing())) / 1000.0
    return ar, voxel_volume


def compute_scores(gt_path: Union[str, Path], pred_path: Union[str, Path]) -> dict[str, float]:
    gt_ar, voxel_volume = read_label(gt_path)
    pred_ar, _ = read_label(pred_path)
    dice = dice_score(gt_ar, pred_ar)
    fp_volume = fp_volume_score(gt_ar, pred_ar) * voxel_volume
    fn_volume = fn_volume_score(gt_ar, pred_ar) * voxel_volume
    surface_dice = surface_dice_score(sitk.ReadImage(gt_path), sitk.ReadImage(pred_path))

    return {
        "dice": dice,
        "fp_volume": fp_volume,
        "fn_volume": fn_volume,
        "surface_dice": surface_dice,
    }


if __name__ == "__main__":
    main()
