"""Generate TCGA predictions directly from the ten seed-0 PANDA checkpoints."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from predict import PANCDRInferenceModel, find_checkpoints, resolve_device
from run_PANCDR_THERAPI_csg2a_compound_da_light_regularized import (
    build_instances_with_rows,
    extract_graph_expression_csg2a_arrays,
    find_csg2a_cache_triple,
    load_csg2a_cache_pair,
)
from therapi_data_utils import build_shared_expression_matrices, build_smiles_feature_bank


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "data" / "THERAPI")
    parser.add_argument(
        "--cache-dir", type=Path, default=root / "data" / "THERAPI_CSG2A_zscore"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=root / "checkpoints" / "tcga_seed0"
    )
    parser.add_argument("--checkpoint", action="append", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=root / "outputs" / "tcga_checkpoint_predictions.csv"
    )
    return parser.parse_args()


def checkpoint_args(args):
    class Values:
        pass

    values = Values()
    values.checkpoint = args.checkpoint
    values.checkpoint_dir = str(args.checkpoint_dir)
    return values


def predict_checkpoint(model, arrays, device, batch_size):
    drug_feat, drug_adj, raw_gex, csg2a_pert, csg2a_comp = arrays
    predictions = np.empty(len(raw_gex), dtype=np.float32)
    for start in range(0, len(raw_gex), batch_size):
        stop = min(start + batch_size, len(raw_gex))
        pred, _ = model.predict(
            torch.as_tensor(drug_feat[start:stop], dtype=torch.float32, device=device),
            torch.as_tensor(drug_adj[start:stop], dtype=torch.float32, device=device),
            torch.as_tensor(raw_gex[start:stop], dtype=torch.float32, device=device),
            torch.as_tensor(csg2a_pert[start:stop], dtype=torch.float32, device=device),
            torch.as_tensor(csg2a_comp[start:stop], dtype=torch.float32, device=device),
        )
        predictions[start:stop] = pred.detach().cpu().numpy()
    return predictions


def main():
    args = parse_args()
    device = resolve_device(args.device)
    gdsc_expr, _, tcga_expr = build_shared_expression_matrices(args.data_root)
    gdsc_response = pd.read_csv(args.data_root / "GDSC_Durg_SMILES_Response.csv")
    tcga_response = pd.read_csv(args.data_root / "TCGA_Durg_SMILES_Response.csv")
    pert_path, comp_path, meta_path = find_csg2a_cache_triple(
        args.cache_dir, "TCGA_LABELED"
    )
    tcga_pert, tcga_comp, _ = load_csg2a_cache_pair(
        pert_path, comp_path, meta_path, tcga_response, "TCGA labeled"
    )
    drug_features, smiles_to_id, skipped = build_smiles_feature_bank(
        gdsc_response, tcga_response
    )
    if skipped:
        raise ValueError(f"Invalid SMILES encountered: {len(skipped)}")
    (
        sample_ids,
        drug_ids,
        labels,
        row_indices,
        missing_expression,
        missing_smiles,
    ) = build_instances_with_rows(
        tcga_response, tcga_expr, smiles_to_id, tcga=True
    )
    if missing_expression or missing_smiles or len(labels) != 383:
        raise ValueError(
            f"Expected 383 complete TCGA pairs; found {len(labels)} "
            f"(missing expression={missing_expression}, missing SMILES={missing_smiles})."
        )
    drug_feat, drug_adj, raw_gex, csg2a_rows, y = extract_graph_expression_csg2a_arrays(
        sample_ids,
        drug_ids,
        labels,
        row_indices,
        tcga_expr,
        drug_features,
        tcga_pert,
        "TCGA",
    )
    arrays = (
        drug_feat,
        drug_adj,
        raw_gex,
        np.asarray(tcga_pert[csg2a_rows], dtype=np.float32),
        np.asarray(tcga_comp[csg2a_rows], dtype=np.float32),
    )

    checkpoints = find_checkpoints(checkpoint_args(args))
    fold_predictions = []
    for checkpoint in checkpoints:
        print(f"Evaluating {checkpoint.name}")
        model = PANCDRInferenceModel(checkpoint, device).to(device)
        fold_predictions.append(
            predict_checkpoint(model, arrays, device, args.batch_size)
        )

    selected = tcga_response.iloc[row_indices].reset_index(drop=True)
    drug_column = "Drug" if "Drug" in selected.columns else "Drug name"
    output = pd.DataFrame(
        {
            "observation_index": np.arange(len(y)),
            "ID": selected["ID"].astype(str),
            "normalized_sample_id": np.asarray(sample_ids, dtype=str),
            "Drug": selected[drug_column].astype(str),
            "Canonical_SMILES": selected["Canonical_SMILES"].astype(str),
            "y_true": y.astype(int),
        }
    )
    for fold, predictions in enumerate(fold_predictions):
        output[f"pred_fold_{fold}_model"] = predictions
    output["pred_ensemble_mean"] = np.mean(np.vstack(fold_predictions), axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output)} predictions to {args.output}")


if __name__ == "__main__":
    main()
