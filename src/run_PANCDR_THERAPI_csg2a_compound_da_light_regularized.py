import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ModelTraining.PANCDR_csg2a_compound_da_light_regularized_cached import (
    train_PANCDR_csg2a_compound_da_light_regularized,
)
from therapi_data_utils import (
    build_shared_expression_matrices,
    build_smiles_feature_bank,
    calculate_graph_feat_numpy,
    tcga_patient_id,
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
    parser = argparse.ArgumentParser(
        description=(
            "Light regularized THERAPI PANCDR with raw GEX DA, CSG2A "
            "perturbed-GEX state DA, and CSG2A compound concat. "
            "scFoundation gene embedding branch is removed."
        )
    )
    parser.add_argument("--data-root", default="../data/THERAPI")
    parser.add_argument("--csg2a-cache-dir", default="../data/THERAPI_CSG2A_zscore")
    parser.add_argument("--gdsc-csg2a-pert", default=None)
    parser.add_argument("--gdsc-csg2a-comp", default=None)
    parser.add_argument("--gdsc-csg2a-meta", default=None)
    parser.add_argument("--tcga-csg2a-pert", default=None)
    parser.add_argument("--tcga-csg2a-comp", default=None)
    parser.add_argument("--tcga-csg2a-meta", default=None)
    parser.add_argument("--csg2a-gene-order-csv", default=None)
    parser.add_argument("--params-file", default="tuned_hyperparameters/TCGA_CV_params.csv")
    parser.add_argument(
        "--checkpoint-dir",
        default="../checkpoint/THERAPI_csg2a_compound_da_light_regularized",
    )
    parser.add_argument(
        "--results-file",
        default="THERAPI_TCGA_results_csg2a_compound_da_light_regularized.csv",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=5)
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
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--d-dim", type=int, default=None)
    parser.add_argument("--batch-source", type=int, default=None)
    parser.add_argument("--batch-target", type=int, default=None)
    parser.add_argument("--lam", type=float, default=None)
    return parser.parse_args()


def find_csg2a_cache_triple(cache_dir, dataset_name):
    cache_dir = Path(cache_dir)
    pert_matches = sorted(
        cache_dir.glob(f"THERAPI_{dataset_name}_csg2a_zscore_perturbed_gex_*.npy")
    )
    comp_matches = sorted(
        cache_dir.glob(f"THERAPI_{dataset_name}_csg2a_zscore_compound_*.npy")
    )
    if not pert_matches:
        raise FileNotFoundError(
            f"Missing THERAPI_{dataset_name} CSG2A perturbed GEX cache in {cache_dir}."
        )
    if not comp_matches:
        raise FileNotFoundError(
            f"Missing THERAPI_{dataset_name} CSG2A compound cache in {cache_dir}."
        )
    meta_path = cache_dir / f"THERAPI_{dataset_name}_csg2a_zscore_row_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing CSG2A row metadata: {meta_path}")
    return str(pert_matches[-1]), str(comp_matches[-1]), str(meta_path)


def load_csg2a_cache_pair(pert_path, comp_path, meta_csv, response_df, label):
    pert = np.load(pert_path, mmap_mode="r")
    comp = np.load(comp_path, mmap_mode="r")
    meta = pd.read_csv(meta_csv)
    if pert.ndim != 2 or comp.ndim != 2:
        raise ValueError(
            f"{label} CSG2A caches must be 2D. pert={pert.shape}, comp={comp.shape}"
        )
    if pert.shape != comp.shape:
        raise ValueError(
            f"{label} CSG2A pert/compound shape mismatch: {pert.shape} vs {comp.shape}"
        )
    if len(meta) != pert.shape[0]:
        raise ValueError(
            f"{label} CSG2A metadata mismatch: meta={len(meta)}, cache_rows={pert.shape[0]}"
        )
    if len(response_df) != pert.shape[0]:
        raise ValueError(
            f"{label} response/cache row mismatch: response={len(response_df)}, "
            f"cache_rows={pert.shape[0]}"
        )
    print(f"{label} CSG2A perturbed cache: shape={pert.shape}, dtype={pert.dtype}")
    print(f"{label} CSG2A compound cache: shape={comp.shape}, dtype={comp.dtype}")
    return pert, comp, meta


def find_csg2a_gene_order_csv(cache_dir):
    cache_dir = Path(cache_dir)
    matches = sorted(cache_dir.glob("THERAPI_GDSC_samplewise_zscore_*genes.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Missing THERAPI_GDSC_samplewise_zscore_*genes.csv in {cache_dir}. "
            "Run generate_therapi_csg2a_zscore_perturbation.py first."
        )
    return str(matches[-1])


def load_csg2a_gene_order(csv_path):
    columns = pd.read_csv(csv_path, index_col=0, nrows=0).columns.astype(str).tolist()
    if not columns:
        raise ValueError(f"No gene columns found in CSG2A z-score CSV: {csv_path}")
    return columns


def samplewise_zscore(expr_df, gene_order, label):
    missing = [gene for gene in gene_order if gene not in expr_df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing {len(missing)} CSG2A genes. "
            f"First missing genes: {missing[:20]}"
        )
    values = expr_df.loc[:, gene_order].to_numpy(dtype=np.float32, copy=True)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    std = np.maximum(std, 1e-6)
    return (values - mean) / std


def build_instances_with_rows(response_df, expr_df, smiles_to_drug_id, tcga=False):
    sample_ids = []
    drug_ids = []
    labels = []
    row_indices = []

    working_df = response_df.copy()
    working_df["ID"] = working_df["ID"].astype(str)
    if tcga:
        working_df["ID"] = working_df["ID"].map(tcga_patient_id)

    skipped_missing_expr = 0
    skipped_missing_smiles = 0
    for row_idx, row in enumerate(working_df.itertuples(index=False)):
        sample_id = row.ID
        smiles = str(row.Canonical_SMILES)
        if sample_id not in expr_df.index:
            skipped_missing_expr += 1
            continue
        if smiles not in smiles_to_drug_id:
            skipped_missing_smiles += 1
            continue

        sample_ids.append(sample_id)
        drug_ids.append(smiles_to_drug_id[smiles])
        labels.append(float(row.Label))
        row_indices.append(row_idx)

    return (
        sample_ids,
        drug_ids,
        np.asarray(labels, dtype=np.float32),
        np.asarray(row_indices, dtype=np.int64),
        skipped_missing_expr,
        skipped_missing_smiles,
    )


def extract_graph_expression_csg2a_arrays(
    sample_ids,
    drug_ids,
    labels,
    response_row_idx,
    expr_df,
    drug_feature,
    csg2a_pert,
    label,
):
    print(f"Extracting {label} graph/raw/CSG2A indices...", flush=True)
    graph_cache = {}
    n_samples = len(sample_ids)
    n_genes = expr_df.shape[1]
    drug_feat_data = []
    drug_adj_data = []
    gexpr_data = np.zeros((n_samples, n_genes), dtype=np.float32)
    csg2a_row_idx = np.zeros(n_samples, dtype=np.int64)
    y = np.asarray(labels, dtype=np.float32)

    for idx, (sample_id, drug_id, csg2a_row) in enumerate(
        zip(sample_ids, drug_ids, response_row_idx)
    ):
        if csg2a_row < 0 or csg2a_row >= csg2a_pert.shape[0]:
            raise IndexError(f"{label}: CSG2A row index out of range: {csg2a_row}")
        if drug_id not in graph_cache:
            feat_mat, adj_list, _ = drug_feature[drug_id]
            graph_cache[drug_id] = calculate_graph_feat_numpy(feat_mat, adj_list)
        drug_feat, drug_adj = graph_cache[drug_id]

        drug_feat_data.append(drug_feat)
        drug_adj_data.append(drug_adj)
        gexpr_data[idx, :] = expr_df.loc[str(sample_id)].to_numpy(
            dtype=np.float32, copy=False
        )
        csg2a_row_idx[idx] = csg2a_row

        if (idx + 1) % 25000 == 0:
            print(f"  {label}: scanned {idx + 1}/{n_samples}", flush=True)

    print(
        f"Finished {label}: {n_samples} instances, {len(graph_cache)} drug graphs",
        flush=True,
    )
    return (
        np.asarray(drug_feat_data, dtype=np.float32),
        np.asarray(drug_adj_data, dtype=np.float32),
        gexpr_data,
        csg2a_row_idx,
        y,
    )


if __name__ == "__main__":
    args = parse_args()
    set_all_seeds(args.seed_base)
    print("Using device:", device)

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

    train_data = [
        X_drug_feat_data,
        X_drug_adj_data,
        X_gexpr_data,
        gdsc_csg2a_pert,
        gdsc_csg2a_comp,
        X_csg2a_row_idx,
        Y,
        tcga_unlabeled_expr.T,
        tcga_unlabeled_csg2a_z,
    ]
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

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print("Run params:", run_params)
    print(
        "CSG2A state branch:",
        f"hidden_dim={args.csg2a_state_hidden_dim}, "
        f"out_dim={args.csg2a_state_out_dim}, lam_adv={args.lam_csg2a_adv}",
    )
    print(
        "CSG2A compound branch:",
        f"hidden_dim={args.csg2a_comp_hidden_dim}, out_dim={args.csg2a_comp_out_dim}",
    )
    print(
        "Regularization:",
        f"csg2a_dropout={args.csg2a_dropout}, fusion_dropout={args.fusion_dropout}, "
        f"use_gcn_dropout={args.use_gcn_dropout}, weight_decay={args.weight_decay}, "
        f"adv_weight_decay={args.adv_weight_decay}",
    )
    print(
        "DA warmup:",
        f"start_epoch={args.da_start_epoch}, warmup_epochs={args.da_warmup_epochs}",
    )
    print(
        "Training light THERAPI PANCDR + CSG2A state/compound "
        "with raw DA and CSG2A-state DA..."
    )

    results = []
    for iter_idx in range(args.runs):
        seed = args.seed_base + iter_idx
        set_all_seeds(seed)
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
        )
        weight_path = os.path.join(args.checkpoint_dir, f"{iter_idx}_model.pt")
        auc_tcga = model.train(run_params, weight_path)
        print("iter %d - roc-TCGA: %.4f" % (iter_idx, auc_tcga))
        results.append(auc_tcga)

        result_df = pd.DataFrame(results, columns=["TCGA AUC"])
        result_df.loc["mean",] = result_df.mean().values
        result_df.to_csv(args.results_file)

    result_df = pd.DataFrame(results, columns=["TCGA AUC"])
    result_df.loc["mean",] = result_df.mean().values
    result_df.to_csv(args.results_file)
    print("Saved results to:", args.results_file)
