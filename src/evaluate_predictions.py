"""Evaluate saved PANDA ensemble predictions with patient-cluster bootstrap CIs."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


METRICS = ("AUROC", "AUPRC", "F1")


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=root / "reference" / "tcga" / "predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "evaluation",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=250)
    return parser.parse_args()


def identify_columns(frame):
    label = "y_true" if "y_true" in frame.columns else "label"
    if label not in frame.columns:
        raise KeyError("Predictions require a y_true or label column.")
    for candidate in ("normalized_sample_id", "sample_id", "ID", "id"):
        if candidate in frame.columns:
            patient = candidate
            break
    else:
        raise KeyError("Predictions require a patient identifier column.")

    pattern = re.compile(r"^pred_fold_(\d+)(?:_model)?$")
    fold_columns = []
    for column in frame.columns:
        match = pattern.match(column)
        if match:
            fold_columns.append((int(match.group(1)), column))
    fold_columns = [column for _, column in sorted(fold_columns)]
    return label, patient, fold_columns


def point_metrics(y, scores, threshold):
    return np.asarray(
        [
            roc_auc_score(y, scores),
            average_precision_score(y, scores),
            f1_score(y, scores >= threshold, zero_division=0),
        ],
        dtype=np.float64,
    )


def weighted_metrics(y, scores, counts, threshold):
    counts = np.asarray(counts, dtype=np.float64)
    positives = counts @ y
    negatives = counts.sum(axis=1) - positives
    if (positives == 0).any() or (negatives == 0).any():
        raise ValueError("Bootstrap samples must contain both outcome classes.")

    order = np.argsort(-scores, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order]) != 0) + 1]
    sorted_counts = counts[:, order]
    tp_step = np.add.reduceat(sorted_counts * y[order], starts, axis=1)
    fp_step = np.add.reduceat(sorted_counts * (1 - y[order]), starts, axis=1)
    tp = np.cumsum(tp_step, axis=1)
    fp = np.cumsum(fp_step, axis=1)
    auroc = np.sum(fp_step * (tp - 0.5 * tp_step), axis=1) / (positives * negatives)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    auprc = np.sum(tp_step * precision, axis=1) / positives

    predicted_positive = (scores >= threshold).astype(np.float64)
    true_positive = counts @ (y * predicted_positive)
    f1 = np.divide(
        2 * true_positive,
        positives + counts @ predicted_positive,
        out=np.zeros_like(true_positive),
        where=(positives + counts @ predicted_positive) > 0,
    )
    return np.column_stack([auroc, auprc, f1])


def draw_patient_counts(rng, batch_size, observation_patient_index, n_patients):
    sampled = rng.integers(0, n_patients, size=(batch_size, n_patients))
    offsets = np.arange(batch_size)[:, None] * n_patients
    patient_counts = np.bincount(
        (sampled + offsets).ravel(), minlength=batch_size * n_patients
    ).reshape(batch_size, n_patients)
    return patient_counts[:, observation_patient_index]


def patient_cluster_bootstrap(y, scores, patients, n_bootstrap, seed, threshold, batch_size):
    _, patient_index = np.unique(patients, return_inverse=True)
    n_patients = int(patient_index.max() + 1)
    rng = np.random.default_rng(seed)
    values = np.empty((n_bootstrap, len(METRICS)), dtype=np.float64)
    completed = 0
    rejected = 0
    while completed < n_bootstrap:
        current = min(batch_size, n_bootstrap - completed)
        counts = draw_patient_counts(rng, current, patient_index, n_patients)
        valid = ((counts @ y) > 0) & ((counts @ (1 - y)) > 0)
        rejected += int((~valid).sum())
        counts = counts[valid]
        if not len(counts):
            continue
        take = min(len(counts), n_bootstrap - completed)
        values[completed : completed + take] = weighted_metrics(
            y, scores, counts[:take], threshold
        )
        completed += take
    return values, n_patients, rejected


def evaluate_file(path, n_bootstrap, seed, threshold, batch_size):
    frame = pd.read_csv(path)
    label_column, patient_column, fold_columns = identify_columns(frame)
    y = frame[label_column].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both binary outcome classes are required.")

    if fold_columns:
        fold_values = frame[fold_columns].to_numpy(dtype=np.float64)
        scores = fold_values.mean(axis=1)
        if "pred_ensemble_mean" in frame.columns:
            np.testing.assert_allclose(
                scores,
                frame["pred_ensemble_mean"].to_numpy(dtype=np.float64),
                rtol=0,
                atol=2e-7,
            )
    elif "pred_ensemble_mean" in frame.columns:
        scores = frame["pred_ensemble_mean"].to_numpy(dtype=np.float64)
    elif "ensemble_probability" in frame.columns:
        scores = frame["ensemble_probability"].to_numpy(dtype=np.float64)
    else:
        raise KeyError("No fold predictions or ensemble prediction column was found.")

    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Predictions must be finite values in [0, 1].")

    patients = frame[patient_column].astype(str).to_numpy()
    estimates = point_metrics(y, scores, threshold)
    bootstrap, n_patients, rejected = patient_cluster_bootstrap(
        y, scores, patients, n_bootstrap, seed, threshold, batch_size
    )
    low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    rows = []
    for index, metric in enumerate(METRICS):
        rows.append(
            {
                "metric": metric,
                "estimate": estimates[index],
                "ci_lower": low[index],
                "ci_upper": high[index],
                "bootstrap_se": bootstrap[:, index].std(ddof=1),
            }
        )
    metadata = {
        "n_pairs": len(frame),
        "n_patients": n_patients,
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "n_checkpoints": len(fold_columns),
        "n_bootstrap": n_bootstrap,
        "rejected_single_class_resamples": rejected,
        "threshold": threshold,
    }
    return frame, scores, pd.DataFrame(rows), bootstrap, metadata


def main():
    args = parse_args()
    frame, scores, metrics, bootstrap, metadata = evaluate_file(
        args.predictions,
        args.n_bootstrap,
        args.seed,
        args.threshold,
        args.batch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics_patient_cluster_ci.csv", index=False)
    pd.DataFrame(bootstrap, columns=METRICS).to_csv(
        args.output_dir / "bootstrap_distributions.csv", index=False
    )
    output = frame.copy()
    output["ensemble_probability_recomputed"] = scores
    output.to_csv(args.output_dir / "ensemble_predictions.csv", index=False)
    pd.Series(metadata).to_json(args.output_dir / "evaluation_metadata.json", indent=2)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(metadata)


if __name__ == "__main__":
    main()
