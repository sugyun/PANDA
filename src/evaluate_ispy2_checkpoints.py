"""Generate I-SPY2 predictions from the ten seed-0 transductive PANDA checkpoints."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from predict import (
    build_pancdr_drug_tensors,
    find_csg2a_gene_order_csv,
    read_gene_order_from_zscore_csv,
    resolve_device,
)
from therapi_data_utils import build_shared_expression_matrices
from train_ispy2 import (
    PAC_SMILES,
    align_expression_df,
    generate_csg2a_rows,
    load_external_labels,
    load_wide_expression_df,
    predict_external_checkpoint,
    samplewise_zscore,
)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    external = root / "data" / "External"
    csg2a_root = root / "external" / "CSG2A_asg_minimal"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "data" / "THERAPI")
    parser.add_argument(
        "--cache-dir", type=Path, default=root / "data" / "THERAPI_CSG2A_zscore"
    )
    parser.add_argument(
        "--expression-csv", type=Path, default=external / "External_labeled_gex.csv"
    )
    parser.add_argument(
        "--response-csv", type=Path, default=external / "External_labeled_response.csv"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=root / "checkpoints" / "ispy2_seed0"
    )
    parser.add_argument("--smiles", default=PAC_SMILES)
    parser.add_argument("--csg2a-root", type=Path, default=csg2a_root)
    parser.add_argument(
        "--csg2a-ckpt",
        type=Path,
        default=csg2a_root / "ckpts" / "260207_CSG2A_finetune_GDSC_MOSA_platform.pt",
    )
    parser.add_argument("--csg2a-pretrain", default=None)
    parser.add_argument("--dose", type=float, default=0.1)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gene-hdim", type=int, default=64)
    parser.add_argument("--finetune-hdim1", type=int, default=512)
    parser.add_argument("--finetune-hdim2", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=root / "outputs" / "ispy2_checkpoint_predictions.csv"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    gdsc_expression, _, _ = build_shared_expression_matrices(args.data_root)
    raw_gene_order = gdsc_expression.columns.astype(str).tolist()
    csg2a_gene_order = read_gene_order_from_zscore_csv(
        find_csg2a_gene_order_csv(args.cache_dir)
    )

    expression = load_wide_expression_df(args.expression_csv, "ID")
    response_args = argparse.Namespace(
        external_response_csv=str(args.response_csv),
        external_response_id_col="id",
        external_label_col="pCR",
    )
    labels = load_external_labels(response_args)
    patient_ids = labels["ID"].astype(str).tolist()
    y = labels["Label"].to_numpy(dtype=np.float32)
    raw_gex = align_expression_df(
        expression.reindex(patient_ids), raw_gene_order
    ).to_numpy(dtype=np.float32)
    csg2a_expression = align_expression_df(
        expression.reindex(patient_ids), csg2a_gene_order
    )
    csg2a_z = samplewise_zscore(
        csg2a_expression, csg2a_gene_order, "I-SPY2 labeled expression"
    )
    csg2a_pert, csg2a_comp = generate_csg2a_rows(
        args, device, csg2a_z, args.smiles
    )
    drug_feat, drug_adj = build_pancdr_drug_tensors(
        [args.smiles] * len(patient_ids), device
    )
    drug_feat = drug_feat.detach().cpu().numpy()
    drug_adj = drug_adj.detach().cpu().numpy()

    checkpoints = sorted(args.checkpoint_dir.glob("fold_*_model.pt"))
    if len(checkpoints) != 10:
        raise FileNotFoundError(
            f"Expected 10 checkpoints in {args.checkpoint_dir}; found {len(checkpoints)}."
        )
    fold_predictions = []
    for checkpoint in checkpoints:
        print(f"Evaluating {checkpoint.name}")
        fold_predictions.append(
            predict_external_checkpoint(
                checkpoint,
                device,
                drug_feat,
                drug_adj,
                raw_gex,
                csg2a_pert,
                csg2a_comp,
                args.batch_size,
            )
        )

    output = pd.DataFrame(
        {"sample_id": patient_ids, "label": y.astype(int), "smiles": args.smiles}
    )
    for fold, predictions in enumerate(fold_predictions):
        output[f"pred_fold_{fold}"] = predictions
    output["pred_ensemble_mean"] = np.mean(np.vstack(fold_predictions), axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output)} predictions to {args.output}")


if __name__ == "__main__":
    main()
