import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from ModelTraining.PANCDR_csg2a_compound_da_light_regularized_cached import (
    train_PANCDR_csg2a_compound_da_light_regularized,
)
from therapi_data_utils import (
    build_shared_expression_matrices,
    build_smiles_feature_bank,
)
from run_PANCDR_THERAPI_csg2a_compound_da_light_regularized import (
    build_instances_with_rows,
    extract_graph_expression_csg2a_arrays,
    find_csg2a_cache_triple,
    find_csg2a_gene_order_csv,
    load_csg2a_cache_pair,
    load_csg2a_gene_order,
    samplewise_zscore,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Train PANDA with seed-0 10-fold GDSC cross-validation and evaluate "
            "the resulting checkpoints on TCGA."
        )
    )
    parser.add_argument("--data-root", default=str(root / "data" / "THERAPI"))
    parser.add_argument(
        "--csg2a-cache-dir",
        default=str(root / "data" / "THERAPI_CSG2A_zscore"),
    )
    parser.add_argument("--gdsc-csg2a-pert", default=None)
    parser.add_argument("--gdsc-csg2a-comp", default=None)
    parser.add_argument("--gdsc-csg2a-meta", default=None)
    parser.add_argument("--tcga-csg2a-pert", default=None)
    parser.add_argument("--tcga-csg2a-comp", default=None)
    parser.add_argument("--tcga-csg2a-meta", default=None)
    parser.add_argument("--csg2a-gene-order-csv", default=None)
    parser.add_argument(
        "--params-file",
        default=str(root / "src" / "tuned_hyperparameters" / "TCGA_CV_params.csv"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(root / "outputs" / "tcga_seed0" / "checkpoints"),
    )
    parser.add_argument(
        "--results-file",
        default=str(root / "outputs" / "tcga_seed0" / "fold_metrics.csv"),
    )
    parser.add_argument(
        "--split-file",
        default=str(root / "outputs" / "tcga_seed0" / "gdsc_fold_assignments.csv"),
    )
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--csg2a-state-hidden-dim", type=int, default=128)
    parser.add_argument("--csg2a-state-out-dim", type=int, default=32)
    parser.add_argument("--csg2a-comp-hidden-dim", type=int, default=128)
    parser.add_argument("--csg2a-comp-out-dim", type=int, default=32)
    parser.add_argument("--csg2a-dropout", type=float, default=0.2)
    parser.add_argument("--fusion-dropout", type=float, default=0.2)
    parser.add_argument("--use-gcn-dropout", dest="use_gcn_dropout", action="store_true")
    parser.add_argument("--no-gcn-dropout", dest="use_gcn_dropout", action="store_false")
    parser.set_defaults(use_gcn_dropout=True)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adv-weight-decay", type=float, default=1e-4)
    parser.add_argument("--lam-csg2a-adv", type=float, default=0.001)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--da-start-epoch", type=int, default=0)
    parser.add_argument("--da-warmup-epochs", type=int, default=50)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--d-dim", type=int, default=None)
    parser.add_argument("--batch-source", type=int, default=None)
    parser.add_argument("--batch-target", type=int, default=None)
    parser.add_argument("--lam", type=float, default=None)
    return parser.parse_args()


def load_existing_results(results_file):
    existing = {}
    if not os.path.exists(results_file):
        return existing
    try:
        df = pd.read_csv(results_file)
    except Exception as exc:
        print("Warning: failed to read existing results:", exc)
        return existing
    required = {"fold", "TCGA AUC"}
    if not required.issubset(df.columns):
        return existing
    fold_num = pd.to_numeric(df["fold"], errors="coerce")
    auc_num = pd.to_numeric(df["TCGA AUC"], errors="coerce")
    valid = fold_num.notna() & auc_num.notna()
    for _, row in df.loc[valid].iterrows():
        existing[int(row["fold"])] = float(row["TCGA AUC"])
    return existing


def save_results(results, results_file):
    result_df = pd.DataFrame(results).sort_values("fold").reset_index(drop=True)
    numeric_scores = pd.to_numeric(result_df["TCGA AUC"], errors="coerce")
    summary = pd.DataFrame(
        [
            {"fold": "mean", "TCGA AUC": numeric_scores.mean()},
            {"fold": "std", "TCGA AUC": numeric_scores.std(ddof=1)},
        ]
    )
    pd.concat([result_df, summary], ignore_index=True).to_csv(results_file, index=False)


if __name__ == "__main__":
    args = parse_args()
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.split_file).parent.mkdir(parents=True, exist_ok=True)
    set_all_seeds(args.seed)
    print("Using device:", device)
    print(
        "Fold policy: KFold(n_splits=%d, shuffle=True, random_state=%d)"
        % (args.n_folds, args.cv_seed)
    )
    print(
        "Seed policy: one initial training seed=%d, no per-fold seed reset."
        % args.seed
    )

    if (
        args.gdsc_csg2a_pert is None
        or args.gdsc_csg2a_comp is None
        or args.gdsc_csg2a_meta is None
    ):
        (
            args.gdsc_csg2a_pert,
            args.gdsc_csg2a_comp,
            args.gdsc_csg2a_meta,
        ) = find_csg2a_cache_triple(args.csg2a_cache_dir, "GDSC")
    if (
        args.tcga_csg2a_pert is None
        or args.tcga_csg2a_comp is None
        or args.tcga_csg2a_meta is None
    ):
        (
            args.tcga_csg2a_pert,
            args.tcga_csg2a_comp,
            args.tcga_csg2a_meta,
        ) = find_csg2a_cache_triple(args.csg2a_cache_dir, "TCGA_LABELED")
    if args.csg2a_gene_order_csv is None:
        args.csg2a_gene_order_csv = find_csg2a_gene_order_csv(args.csg2a_cache_dir)

    gdsc_expr, tcga_unlabeled_expr, tcga_labeled_expr = build_shared_expression_matrices(
        args.data_root
    )
    print("Shared THERAPI genes:", gdsc_expr.shape[1])
    print("GDSC expr shape:", gdsc_expr.shape)
    print("TCGA unlabeled expr shape:", tcga_unlabeled_expr.shape)
    print("TCGA labeled expr shape:", tcga_labeled_expr.shape)

    csg2a_gene_order = load_csg2a_gene_order(args.csg2a_gene_order_csv)
    tcga_unlabeled_csg2a_z = samplewise_zscore(
        tcga_unlabeled_expr, csg2a_gene_order, "TCGA unlabeled expression"
    )
    print("TCGA unlabeled CSG2A-state DA target:", tcga_unlabeled_csg2a_z.shape)

    gdsc_resp = pd.read_csv(os.path.join(args.data_root, "GDSC_Durg_SMILES_Response.csv"))
    tcga_resp = pd.read_csv(os.path.join(args.data_root, "TCGA_Durg_SMILES_Response.csv"))
    gdsc_csg2a_pert, gdsc_csg2a_comp, _ = load_csg2a_cache_pair(
        args.gdsc_csg2a_pert,
        args.gdsc_csg2a_comp,
        args.gdsc_csg2a_meta,
        gdsc_resp,
        "THERAPI GDSC",
    )
    tcga_csg2a_pert, tcga_csg2a_comp, _ = load_csg2a_cache_pair(
        args.tcga_csg2a_pert,
        args.tcga_csg2a_comp,
        args.tcga_csg2a_meta,
        tcga_resp,
        "THERAPI TCGA labeled",
    )

    drug_feature, smiles_to_drug_id, skipped_smiles = build_smiles_feature_bank(
        gdsc_resp, tcga_resp
    )
    print("Drug graphs built:", len(drug_feature))
    print("Skipped invalid SMILES:", len(skipped_smiles))

    (
        gdsc_sample_ids,
        gdsc_drug_ids,
        gdsc_labels,
        gdsc_response_row_idx,
        gdsc_missing_expr,
        gdsc_missing_smiles,
    ) = build_instances_with_rows(gdsc_resp, gdsc_expr, smiles_to_drug_id, tcga=False)
    (
        tcga_sample_ids,
        tcga_drug_ids,
        tcga_labels,
        tcga_response_row_idx,
        tcga_missing_expr,
        tcga_missing_smiles,
    ) = build_instances_with_rows(tcga_resp, tcga_labeled_expr, smiles_to_drug_id, tcga=True)
    print(
        "GDSC instances: %d | missing expr: %d | missing SMILES graph: %d"
        % (len(gdsc_labels), gdsc_missing_expr, gdsc_missing_smiles)
    )
    print(
        "TCGA instances: %d | missing expr: %d | missing SMILES graph: %d"
        % (len(tcga_labels), tcga_missing_expr, tcga_missing_smiles)
    )

    (
        X_drug_feat_data,
        X_drug_adj_data,
        X_gexpr_data,
        X_csg2a_row_idx,
        Y,
    ) = extract_graph_expression_csg2a_arrays(
        gdsc_sample_ids,
        gdsc_drug_ids,
        gdsc_labels,
        gdsc_response_row_idx,
        gdsc_expr,
        drug_feature,
        gdsc_csg2a_pert,
        "THERAPI GDSC",
    )
    (
        TX_drug_feat_data_test,
        TX_drug_adj_data_test,
        TX_gexpr_data_test,
        TX_csg2a_row_idx_test,
        TY_test,
    ) = extract_graph_expression_csg2a_arrays(
        tcga_sample_ids,
        tcga_drug_ids,
        tcga_labels,
        tcga_response_row_idx,
        tcga_labeled_expr,
        drug_feature,
        tcga_csg2a_pert,
        "THERAPI TCGA",
    )

    test_data = [
        TX_drug_feat_data_test,
        TX_drug_adj_data_test,
        TX_gexpr_data_test,
        tcga_csg2a_pert,
        tcga_csg2a_comp,
        TX_csg2a_row_idx_test,
        TY_test,
    ]

    df = pd.read_csv(args.params_file)
    best_params = eval(
        df.loc[
            (df["Model"] == "PANCDR") & (df["Classification"] == "T"),
            "Best_params",
        ].values[0]
    )
    run_params = dict(best_params)
    if args.nz is not None:
        run_params["nz"] = args.nz
    if args.d_dim is not None:
        run_params["d_dim"] = args.d_dim
    if args.batch_source is not None or args.batch_target is not None:
        run_params["batch_size"] = [
            args.batch_source
            if args.batch_source is not None
            else run_params["batch_size"][0],
            args.batch_target
            if args.batch_target is not None
            else run_params["batch_size"][1],
        ]
    if args.lam is not None:
        run_params["lam"] = args.lam
    print("Run params:", run_params)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.cv_seed)
    split_rows = []
    for fold_idx, (_, val_idx) in enumerate(kf.split(np.arange(len(Y)))):
        for idx in val_idx:
            split_rows.append(
                {
                    "instance_index": int(idx),
                    "fold": int(fold_idx),
                    "sample_id": gdsc_sample_ids[idx],
                    "drug_id": gdsc_drug_ids[idx],
                    "label": float(Y[idx]),
                }
            )
    pd.DataFrame(split_rows).to_csv(args.split_file, index=False)
    print("Saved GDSC 10-fold split to:", args.split_file)

    existing_results = load_existing_results(args.results_file) if args.resume_existing else {}
    results = [
        {"fold": fold, "TCGA AUC": auc}
        for fold, auc in sorted(existing_results.items())
    ]
    if existing_results:
        print("Loaded existing fold/run results from:", args.results_file)

    print(
        "Training light regularized CSG2A compound DA with THERAPI 10-fold "
        "GDSC CV and TCGA external test..."
    )
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(Y)))):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        print(
            "Fold %d | train instances=%d | val instances=%d"
            % (fold_idx, len(train_idx), len(val_idx))
        )

        train_data = [
            X_drug_feat_data[train_idx],
            X_drug_adj_data[train_idx],
            X_gexpr_data[train_idx],
            gdsc_csg2a_pert,
            gdsc_csg2a_comp,
            X_csg2a_row_idx[train_idx],
            Y[train_idx],
            tcga_unlabeled_expr.T,
            tcga_unlabeled_csg2a_z,
        ]
        source_val_data = [
            X_drug_feat_data[val_idx],
            X_drug_adj_data[val_idx],
            X_gexpr_data[val_idx],
            gdsc_csg2a_pert,
            gdsc_csg2a_comp,
            X_csg2a_row_idx[val_idx],
            Y[val_idx],
        ]

        if args.resume_existing and fold_idx in existing_results:
            print(
                "Skipping fold %d because an existing result was found."
                % fold_idx
            )
            continue

        model = train_PANCDR_csg2a_compound_da_light_regularized(
            train_data,
            test_data,
            csg2a_state_hidden_dim=args.csg2a_state_hidden_dim,
            csg2a_state_out_dim=args.csg2a_state_out_dim,
            csg2a_comp_hidden_dim=args.csg2a_comp_hidden_dim,
            csg2a_comp_out_dim=args.csg2a_comp_out_dim,
            csg2a_dropout=args.csg2a_dropout,
            fusion_dropout=args.fusion_dropout,
            use_gcn_dropout=args.use_gcn_dropout,
            weight_decay=args.weight_decay,
            adv_weight_decay=args.adv_weight_decay,
            lam_csg2a_adv=args.lam_csg2a_adv,
            da_start_epoch=args.da_start_epoch,
            da_warmup_epochs=args.da_warmup_epochs,
            max_epochs=args.max_epochs,
            patience=args.patience,
            eval_batch_size=args.eval_batch_size,
            source_val_data=source_val_data,
        )
        weight_path = os.path.join(args.checkpoint_dir, f"fold_{fold_idx}_model.pt")
        auc_tcga = model.train(run_params, weight_path)
        print("fold %d - roc-TCGA: %.4f" % (fold_idx, auc_tcga))
        results = [row for row in results if int(row["fold"]) != fold_idx]
        results.append({"fold": fold_idx, "TCGA AUC": auc_tcga})
        save_results(results, args.results_file)

    save_results(results, args.results_file)
    print("Saved results to:", args.results_file)
