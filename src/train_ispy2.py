import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold
from torch import nn

from ModelTraining.PANCDR_csg2a_compound_da_ablation import (
    FusedOnlyPredictionHead,
    train_PANCDR_csg2a_compound_da_ablation,
)
from ModelTraining.csg2a_utils import CSG2AFeatureEncoder
from ModelTraining.model import Encoder, GCN
from predict import (
    CSG2AGenerator,
    build_pancdr_drug_tensors,
    find_csg2a_gene_order_csv,
    read_gene_order_from_zscore_csv,
    resolve_device,
)
from run_PANCDR_THERAPI_csg2a_compound_da_light_regularized import (
    build_instances_with_rows,
    extract_graph_expression_csg2a_arrays,
    find_csg2a_cache_triple,
    load_csg2a_cache_pair,
    samplewise_zscore,
)
from therapi_data_utils import build_shared_expression_matrices, build_smiles_feature_bank


PAC_SMILES = (
    "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)"
    "(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)"
    "OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
)


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
    external_root = root / "data" / "External"
    parser = argparse.ArgumentParser(
        description="Seed-0 10-fold PANDA training and transductive I-SPY2 evaluation."
    )
    parser.add_argument("--data-root", default=str(root / "data" / "THERAPI"))
    parser.add_argument(
        "--gdsc-info-csv",
        default=str(root / "data" / "THERAPI" / "GDSC_info.csv"),
    )
    parser.add_argument("--gdsc-tissue", default="Breast")
    parser.add_argument(
        "--external-labeled-gex-csv",
        default=str(external_root / "External_labeled_gex.csv"),
    )
    parser.add_argument(
        "--external-unlabeled-gex-csv",
        default=str(external_root / "External_unlabeled_gex.csv"),
    )
    parser.add_argument(
        "--external-response-csv",
        default=str(external_root / "External_labeled_response.csv"),
    )
    parser.add_argument("--external-id-col", default="ID")
    parser.add_argument("--external-response-id-col", default="id")
    parser.add_argument("--external-label-col", default="pCR")
    parser.add_argument("--smiles", default=PAC_SMILES)
    parser.add_argument(
        "--csg2a-cache-dir", default=str(root / "data" / "THERAPI_CSG2A_zscore")
    )
    parser.add_argument(
        "--csg2a-root", default=str(root / "external" / "CSG2A_asg_minimal")
    )
    parser.add_argument(
        "--csg2a-ckpt",
        default=str(
            root
            / "external"
            / "CSG2A_asg_minimal"
            / "ckpts"
            / "260207_CSG2A_finetune_GDSC_MOSA_platform.pt"
        ),
    )
    parser.add_argument("--csg2a-pretrain", default=None)
    parser.add_argument("--dose", type=float, default=0.1)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gene-hdim", type=int, default=64)
    parser.add_argument("--finetune-hdim1", type=int, default=512)
    parser.add_argument("--finetune-hdim2", type=int, default=64)
    parser.add_argument(
        "--params-file",
        default=str(root / "src" / "tuned_hyperparameters" / "TCGA_CV_params.csv"),
    )
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=60)
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
    parser.add_argument("--da-start-epoch", type=int, default=0)
    parser.add_argument("--da-warmup-epochs", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--d-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-adv", type=float, default=None)
    parser.add_argument("--batch-source", type=int, default=None)
    parser.add_argument("--batch-target", type=int, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        default=str(root / "outputs" / "ispy2_seed0" / "checkpoints"),
    )
    parser.add_argument(
        "--results-file",
        default=str(root / "outputs" / "ispy2_seed0" / "fold_metrics.csv"),
    )
    parser.add_argument(
        "--preds-csv",
        default=str(root / "outputs" / "ispy2_seed0" / "predictions.csv"),
    )
    parser.add_argument(
        "--split-file",
        default=str(root / "outputs" / "ispy2_seed0" / "gdsc_fold_assignments.csv"),
    )
    parser.add_argument(
        "--boxplot-png",
        default=str(root / "outputs" / "ispy2_seed0" / "score_boxplot.png"),
    )
    parser.add_argument(
        "--roc-png",
        default=str(root / "outputs" / "ispy2_seed0" / "roc_curve.png"),
    )
    return parser.parse_args()


def load_wide_expression_df(path, id_col):
    df = pd.read_csv(path)
    if id_col not in df.columns:
        first_col = str(df.columns[0])
        if first_col.startswith("Unnamed:") or first_col == "":
            df = df.rename(columns={df.columns[0]: id_col})
    if id_col not in df.columns:
        raise KeyError(f"Missing ID column '{id_col}' in {path}")
    df[id_col] = df[id_col].astype(str)
    gene_cols = [col for col in df.columns if col != id_col]
    work = df.set_index(id_col)[gene_cols]
    work.columns = work.columns.astype(str)
    return work


def align_expression_df(expr_df, gene_order):
    return expr_df.reindex(columns=[str(gene) for gene in gene_order], fill_value=0.0).astype(
        np.float32
    )


def load_external_labels(args):
    response_df = pd.read_csv(args.external_response_csv).copy()
    response_df[args.external_response_id_col] = response_df[
        args.external_response_id_col
    ].astype(str)
    labels = pd.to_numeric(response_df[args.external_label_col], errors="raise").astype(np.float32)
    merged = pd.DataFrame(
        {
            "ID": response_df[args.external_response_id_col].astype(str),
            "Label": labels,
        }
    )
    return merged


def filter_gdsc_by_tissue(gdsc_resp, args):
    if not args.gdsc_tissue:
        return gdsc_resp
    info_df = pd.read_csv(args.gdsc_info_csv).copy()
    if "ID" not in info_df.columns or "tissue_type" not in info_df.columns:
        raise KeyError("GDSC_info.csv must contain columns: ID, tissue_type")
    info_df["ID"] = info_df["ID"].astype(str)
    keep_ids = set(
        info_df.loc[
            info_df["tissue_type"].astype(str).str.lower() == str(args.gdsc_tissue).lower(),
            "ID",
        ].astype(str)
    )
    filtered = gdsc_resp.loc[gdsc_resp["ID"].astype(str).isin(keep_ids)].copy()
    print(
        "Filtered GDSC by tissue '%s': %d -> %d rows"
        % (args.gdsc_tissue, len(gdsc_resp), len(filtered))
    )
    return filtered


def build_instances_with_preserved_rows(response_df, expr_df, smiles_to_drug_id):
    sample_ids = []
    drug_ids = []
    labels = []
    row_indices = []
    skipped_missing_expr = 0
    skipped_missing_smiles = 0
    working_df = response_df.copy()
    working_df["ID"] = working_df["ID"].astype(str)
    for row in working_df.itertuples(index=False):
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
        row_indices.append(int(getattr(row, "orig_row_idx")))
    return (
        sample_ids,
        drug_ids,
        np.asarray(labels, dtype=np.float32),
        np.asarray(row_indices, dtype=np.int64),
        skipped_missing_expr,
        skipped_missing_smiles,
    )


def build_external_response_df(label_df, smiles):
    out = label_df.copy()
    out["Canonical_SMILES"] = smiles
    out["Drug name"] = "Paclitaxel"
    return out[["ID", "Drug name", "Canonical_SMILES", "Label"]]


def generate_csg2a_rows(args, device, z_expr, smiles):
    generator = CSG2AGenerator(args, device)
    pert_rows = []
    comp_rows = []
    smiles_list = smiles if isinstance(smiles, (list, tuple, np.ndarray, pd.Series)) else [smiles] * z_expr.shape[0]
    if len(smiles_list) != z_expr.shape[0]:
        raise ValueError(
            "SMILES length mismatch: got %d smiles for %d samples"
            % (len(smiles_list), z_expr.shape[0])
        )
    for idx in range(z_expr.shape[0]):
        pert, comp = generator.generate(z_expr[idx : idx + 1], [str(smiles_list[idx])])
        pert_rows.append(pert[0])
        comp_rows.append(comp[0])
    return np.asarray(pert_rows, dtype=np.float32), np.asarray(comp_rows, dtype=np.float32)


class SavedAblationPredictor(nn.Module):
    def __init__(self, checkpoint_path, device, raw_dim, drug_feat_dim, csg2a_dim):
        super().__init__()
        self.state = torch.load(checkpoint_path, map_location=device)
        params = self.state["params"]
        self.use_raw_gex = bool(self.state.get("use_raw_gex", True))
        self.use_drug_gcn = bool(self.state.get("use_drug_gcn", True))
        self.use_csg2a_state = bool(self.state.get("use_csg2a_state", True))
        self.use_csg2a_compound = bool(self.state.get("use_csg2a_compound", True))
        self.nz = int(params["nz"])
        self.d_dim = int(params["d_dim"])
        self.csg2a_state_out_dim = int(self.state["csg2a_state_out_dim"])
        self.csg2a_comp_out_dim = int(self.state["csg2a_comp_out_dim"])
        self.device = device

        self.EN_model = Encoder(raw_dim, self.nz, device).to(device) if self.use_raw_gex else None
        self.CSG2AState_model = (
            CSG2AFeatureEncoder(
                csg2a_dim,
                hidden_dim=int(self.state["csg2a_state_hidden_dim"]),
                out_dim=self.csg2a_state_out_dim,
                dropout=float(self.state["csg2a_dropout"]),
            ).to(device)
            if self.use_csg2a_state
            else None
        )
        self.CSG2ACompound_model = (
            CSG2AFeatureEncoder(
                csg2a_dim,
                hidden_dim=int(self.state["csg2a_comp_hidden_dim"]),
                out_dim=self.csg2a_comp_out_dim,
                dropout=float(self.state["csg2a_dropout"]),
            ).to(device)
            if self.use_csg2a_compound
            else None
        )
        fused_dim = 0
        if self.use_raw_gex:
            fused_dim += self.nz
        if self.use_csg2a_state:
            fused_dim += self.csg2a_state_out_dim
        if self.use_csg2a_compound:
            fused_dim += self.csg2a_comp_out_dim
        if self.use_drug_gcn:
            self.Response_model = GCN(
                drug_feat_dim,
                [256, 256, 256],
                h_dims=[self.d_dim, fused_dim + self.d_dim],
                use_dropout=bool(self.state["use_gcn_dropout"]),
            ).to(device)
        else:
            self.Response_model = FusedOnlyPredictionHead(
                fused_dim, dropout=float(self.state.get("fusion_dropout", 0.2))
            ).to(device)
        if self.EN_model is not None:
            self.EN_model.load_state_dict(self.state["EN_model"])
            self.EN_model.eval()
        if self.CSG2AState_model is not None:
            self.CSG2AState_model.load_state_dict(self.state["CSG2AState_model"])
            self.CSG2AState_model.eval()
        if self.CSG2ACompound_model is not None:
            self.CSG2ACompound_model.load_state_dict(self.state["CSG2ACompound_model"])
            self.CSG2ACompound_model.eval()
        self.Response_model.load_state_dict(self.state["Response_model"])
        self.Response_model.eval()

    @torch.no_grad()
    def predict(self, drug_feat, drug_adj, raw_gex, csg2a_pert, csg2a_comp):
        latents = []
        raw_mu = None
        if self.EN_model is not None:
            _, raw_mu, _ = self.EN_model(raw_gex)
            latents.append(raw_mu)
        if self.CSG2AState_model is not None:
            latents.append(self.CSG2AState_model(csg2a_pert))
        if self.CSG2ACompound_model is not None:
            latents.append(self.CSG2ACompound_model(csg2a_comp))
        fused = torch.cat(latents, dim=1)
        pred = (
            self.Response_model(drug_feat, drug_adj, fused)
            if self.use_drug_gcn else self.Response_model(fused)
        )
        return pred.view(-1)


def predict_external_checkpoint(
    checkpoint_path,
    device,
    drug_feat,
    drug_adj,
    raw_gex,
    csg2a_pert,
    csg2a_comp,
    batch_size,
):
    model = SavedAblationPredictor(
        checkpoint_path,
        device,
        raw_gex.shape[1],
        drug_feat.shape[2],
        csg2a_pert.shape[1],
    ).to(device)
    preds = np.zeros(raw_gex.shape[0], dtype=np.float32)
    for start in range(0, raw_gex.shape[0], batch_size):
        end = min(start + batch_size, raw_gex.shape[0])
        pred = model.predict(
            torch.as_tensor(drug_feat[start:end], dtype=torch.float32, device=device),
            torch.as_tensor(drug_adj[start:end], dtype=torch.float32, device=device),
            torch.as_tensor(raw_gex[start:end], dtype=torch.float32, device=device),
            torch.as_tensor(csg2a_pert[start:end], dtype=torch.float32, device=device),
            torch.as_tensor(csg2a_comp[start:end], dtype=torch.float32, device=device),
        )
        preds[start:end] = pred.detach().cpu().numpy().astype(np.float32, copy=False)
    return preds


def save_boxplot(pred_df, out_path):
    plot_df = pred_df.copy()
    plot_df["group"] = plot_df["label"].map(
        {0: "failed complete response", 1: "complete response"}
    )
    order = ["failed complete response", "complete response"]
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=plot_df, x="group", y="pred_ensemble_mean", order=order)
    sns.stripplot(
        data=plot_df,
        x="group",
        y="pred_ensemble_mean",
        order=order,
        color="black",
        alpha=0.5,
        size=3,
    )
    plt.xlabel("")
    plt.ylabel("Predicted response probability")
    plt.title("External Paclitaxel")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_roc(labels, preds, auc_value, out_path):
    fpr, tpr, _ = roc_curve(labels, preds)
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(fpr, tpr, label=f"Ensemble ROC (AUROC={auc_value:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("External Paclitaxel ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_ensemble_stats(labels, preds):
    precision, recall, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)
    best = None
    thresholds = np.unique(preds.astype(np.float32))
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))
    for threshold in thresholds:
        pred_label = (preds >= threshold).astype(np.int32)
        f1_num = 2 * np.sum((pred_label == 1) & (labels == 1))
        f1_den = 2 * np.sum((pred_label == 1) & (labels == 1)) + np.sum(
            (pred_label == 1) & (labels == 0)
        ) + np.sum((pred_label == 0) & (labels == 1))
        f1 = float(f1_num / f1_den) if f1_den > 0 else 0.0
        bal_acc = float(balanced_accuracy_score(labels, pred_label))
        acc = float(accuracy_score(labels, pred_label))
        tn, fp, fn, tp = confusion_matrix(labels, pred_label, labels=[0, 1]).ravel()
        row = (f1, bal_acc, acc, float(threshold), int(tn), int(fp), int(fn), int(tp), int(pred_label.sum()))
        if best is None or row[:3] > best[:3]:
            best = row

    r_scores = preds[labels == 1]
    nr_scores = preds[labels == 0]
    mw_u, mw_p = mannwhitneyu(r_scores, nr_scores, alternative="two-sided")
    return {
        "pr_auc": float(pr_auc),
        "best_threshold": float(best[3]),
        "best_f1": float(best[0]),
        "best_balanced_accuracy": float(best[1]),
        "best_accuracy": float(best[2]),
        "best_tn": int(best[4]),
        "best_fp": int(best[5]),
        "best_fn": int(best[6]),
        "best_tp": int(best[7]),
        "best_pred_positive": int(best[8]),
        "mannwhitney_u": float(mw_u),
        "mannwhitney_p": float(mw_p),
        "r_mean": float(r_scores.mean()),
        "nr_mean": float(nr_scores.mean()),
        "r_median": float(np.median(r_scores)),
        "nr_median": float(np.median(nr_scores)),
    }


def main():
    args = parse_args()
    set_all_seeds(0)
    device = resolve_device(args.device)

    (
        args.gdsc_csg2a_pert,
        args.gdsc_csg2a_comp,
        args.gdsc_csg2a_meta,
    ) = find_csg2a_cache_triple(args.csg2a_cache_dir, "GDSC")
    csg2a_gene_order = read_gene_order_from_zscore_csv(find_csg2a_gene_order_csv(args.csg2a_cache_dir))

    gdsc_expr, _, _ = build_shared_expression_matrices(args.data_root)
    raw_gene_order = gdsc_expr.columns.astype(str).tolist()

    external_labeled_wide = load_wide_expression_df(args.external_labeled_gex_csv, args.external_id_col)
    external_unlabeled_wide = load_wide_expression_df(
        args.external_unlabeled_gex_csv, args.external_id_col
    )
    external_label_df = load_external_labels(args)

    external_labeled_ids = external_label_df["ID"].astype(str).tolist()
    external_labels = external_label_df["Label"].to_numpy(dtype=np.float32)

    external_labeled_raw_df = align_expression_df(
        external_labeled_wide.reindex(external_labeled_ids), raw_gene_order
    )
    external_unlabeled_raw_df = align_expression_df(external_unlabeled_wide, raw_gene_order)
    external_labeled_csg2a_df = align_expression_df(
        external_labeled_wide.reindex(external_labeled_ids), csg2a_gene_order
    )
    external_unlabeled_csg2a_df = align_expression_df(external_unlabeled_wide, csg2a_gene_order)
    external_labeled_csg2a_z = samplewise_zscore(
        external_labeled_csg2a_df, csg2a_gene_order, "External labeled expression"
    )
    external_unlabeled_csg2a_z = samplewise_zscore(
        external_unlabeled_csg2a_df, csg2a_gene_order, "External unlabeled expression"
    )

    external_resp = build_external_response_df(external_label_df, args.smiles)
    gdsc_resp_full = pd.read_csv(os.path.join(args.data_root, "GDSC_Durg_SMILES_Response.csv"))
    gdsc_resp_full["orig_row_idx"] = np.arange(len(gdsc_resp_full), dtype=np.int64)
    gdsc_resp = filter_gdsc_by_tissue(gdsc_resp_full, args)
    gdsc_csg2a_pert, gdsc_csg2a_comp, _ = load_csg2a_cache_pair(
        args.gdsc_csg2a_pert,
        args.gdsc_csg2a_comp,
        args.gdsc_csg2a_meta,
        gdsc_resp_full,
        "THERAPI GDSC",
    )

    drug_feature, smiles_to_drug_id, skipped_smiles = build_smiles_feature_bank(gdsc_resp, external_resp)
    print("Using device:", device)
    print("Drug graphs built:", len(drug_feature))
    print("Skipped invalid SMILES:", len(skipped_smiles))

    (
        gdsc_sample_ids,
        gdsc_drug_ids,
        gdsc_labels,
        gdsc_response_row_idx,
        gdsc_missing_expr,
        gdsc_missing_smiles,
    ) = build_instances_with_preserved_rows(gdsc_resp, gdsc_expr, smiles_to_drug_id)
    (
        external_sample_ids,
        external_drug_ids,
        _,
        external_response_row_idx,
        external_missing_expr,
        external_missing_smiles,
    ) = build_instances_with_rows(external_resp, external_labeled_raw_df, smiles_to_drug_id, tcga=False)
    print(
        "GDSC instances: %d | missing expr: %d | missing SMILES graph: %d"
        % (len(gdsc_labels), gdsc_missing_expr, gdsc_missing_smiles)
    )
    print(
        "External instances: %d | missing expr: %d | missing SMILES graph: %d"
        % (len(external_sample_ids), external_missing_expr, external_missing_smiles)
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
        external_sample_ids,
        external_drug_ids,
        external_labels,
        external_response_row_idx,
        external_labeled_raw_df,
        drug_feature,
        np.zeros((len(external_resp), 1), dtype=np.float32),
        "External Paclitaxel",
    )

    external_labeled_pert, external_labeled_comp = generate_csg2a_rows(
        args, device, external_labeled_csg2a_z, args.smiles
    )

    test_data = [
        TX_drug_feat_data_test,
        TX_drug_adj_data_test,
        TX_gexpr_data_test,
        external_labeled_pert,
        external_labeled_comp,
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
    if args.lr is not None:
        run_params["lr"] = args.lr
    if args.lr_adv is not None:
        run_params["lr_adv"] = args.lr_adv
    if args.batch_source is not None or args.batch_target is not None:
        run_params["batch_size"] = [
            args.batch_source if args.batch_source is not None else run_params["batch_size"][0],
            args.batch_target if args.batch_target is not None else run_params["batch_size"][1],
        ]
    if args.lam is not None:
        run_params["lam"] = args.lam

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)
    print("External labeled shape:", external_labeled_raw_df.shape)
    print("External unlabeled shape:", external_unlabeled_raw_df.shape)
    print("Run params:", run_params)

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
    print("Saved GDSC validation split:", args.split_file)

    fold_rows = []
    pred_df = pd.DataFrame({"sample_id": external_sample_ids, "label": TY_test.astype(np.int32)})
    all_fold_preds = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(Y)))):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        print(
            "Fold %d | GDSC train=%d | GDSC val=%d | External eval=%d"
            % (fold_idx, len(train_idx), len(val_idx), len(TY_test))
        )

        train_data = [
            X_drug_feat_data[train_idx],
            X_drug_adj_data[train_idx],
            X_gexpr_data[train_idx],
            gdsc_csg2a_pert,
            gdsc_csg2a_comp,
            X_csg2a_row_idx[train_idx],
            Y[train_idx],
            external_unlabeled_raw_df.T,
            external_unlabeled_csg2a_z,
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

        model = train_PANCDR_csg2a_compound_da_ablation(
            train_data,
            test_data,
            use_raw_gex=True,
            use_csg2a_state=True,
            use_csg2a_compound=True,
            use_drug_gcn=True,
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
            ablation_name="external_paclitaxel_full",
        )
        checkpoint_path = os.path.join(args.checkpoint_dir, f"fold_{fold_idx}_model.pt")
        auc_external = model.train(run_params, checkpoint_path)
        pred_external = predict_external_checkpoint(
            checkpoint_path,
            device,
            TX_drug_feat_data_test,
            TX_drug_adj_data_test,
            TX_gexpr_data_test,
            external_labeled_pert,
            external_labeled_comp,
            args.eval_batch_size,
        )
        auc_external_recomputed = roc_auc_score(TY_test, pred_external)
        pred_df[f"pred_fold_{fold_idx}"] = pred_external
        all_fold_preds.append(pred_external)
        fold_rows.append(
            {
                "fold": int(fold_idx),
                "external_auroc": float(auc_external),
                "external_auroc_recomputed": float(auc_external_recomputed),
            }
        )
        print(
            "fold %d - external AUROC(return): %.6f | recomputed: %.6f"
            % (fold_idx, auc_external, auc_external_recomputed)
        )

    fold_aurocs = np.asarray([row["external_auroc_recomputed"] for row in fold_rows], dtype=np.float64)
    ensemble_pred = np.mean(np.stack(all_fold_preds, axis=0), axis=0).astype(np.float32)
    ensemble_auc = roc_auc_score(TY_test, ensemble_pred)
    ensemble_stats = compute_ensemble_stats(TY_test.astype(np.int32), ensemble_pred)
    pred_df["pred_ensemble_mean"] = ensemble_pred

    results_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(
        [
            {"fold": "fold_mean", "external_auroc": float(fold_aurocs.mean())},
            {
                "fold": "fold_std",
                "external_auroc": float(fold_aurocs.std(ddof=1)) if len(fold_aurocs) > 1 else 0.0,
            },
            {"fold": "ensemble_mean", "external_auroc": float(ensemble_auc)},
            {"fold": "ensemble_pr_auc", "external_auroc": ensemble_stats["pr_auc"]},
            {"fold": "ensemble_best_threshold", "external_auroc": ensemble_stats["best_threshold"]},
            {"fold": "ensemble_best_f1", "external_auroc": ensemble_stats["best_f1"]},
            {
                "fold": "ensemble_best_balanced_accuracy",
                "external_auroc": ensemble_stats["best_balanced_accuracy"],
            },
            {"fold": "ensemble_best_accuracy", "external_auroc": ensemble_stats["best_accuracy"]},
            {"fold": "ensemble_best_tn", "external_auroc": ensemble_stats["best_tn"]},
            {"fold": "ensemble_best_fp", "external_auroc": ensemble_stats["best_fp"]},
            {"fold": "ensemble_best_fn", "external_auroc": ensemble_stats["best_fn"]},
            {"fold": "ensemble_best_tp", "external_auroc": ensemble_stats["best_tp"]},
            {
                "fold": "ensemble_best_pred_positive",
                "external_auroc": ensemble_stats["best_pred_positive"],
            },
            {"fold": "ensemble_mannwhitney_u", "external_auroc": ensemble_stats["mannwhitney_u"]},
            {"fold": "ensemble_mannwhitney_p", "external_auroc": ensemble_stats["mannwhitney_p"]},
            {"fold": "ensemble_r_mean", "external_auroc": ensemble_stats["r_mean"]},
            {"fold": "ensemble_nr_mean", "external_auroc": ensemble_stats["nr_mean"]},
            {"fold": "ensemble_r_median", "external_auroc": ensemble_stats["r_median"]},
            {"fold": "ensemble_nr_median", "external_auroc": ensemble_stats["nr_median"]},
        ]
    )
    pd.concat([results_df, summary_df], ignore_index=True).to_csv(args.results_file, index=False)
    pred_df.to_csv(args.preds_csv, index=False)
    save_boxplot(pred_df, args.boxplot_png)
    save_roc(TY_test, ensemble_pred, ensemble_auc, args.roc_png)

    print("fold_mean: External AUROC=%.6f" % fold_aurocs.mean())
    print("ensemble_mean: External AUROC=%.6f" % ensemble_auc)
    print("Saved:", args.results_file)
    print("Saved:", args.preds_csv)
    print("Saved:", args.boxplot_png)
    print("Saved:", args.roc_png)


if __name__ == "__main__":
    main()
