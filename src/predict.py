import argparse
import importlib.util
import json
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from rdkit import RDLogger

from ModelTraining.csg2a_utils import CSG2AFeatureEncoder
from ModelTraining.model import Encoder, GCN
from therapi_data_utils import (
    build_shared_expression_matrices,
    calculate_graph_feat_numpy,
    smiles_to_graph,
)


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def package_root():
    return Path(__file__).resolve().parents[1]


def default_csg2a_root():
    bundled = package_root() / "external" / "CSG2A_asg_minimal"
    if bundled.exists():
        return bundled
    return package_root().parent.parent / "CSG2A_asg"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    root = package_root()
    csg2a_root = default_csg2a_root()
    parser = argparse.ArgumentParser(
        description=(
            "Inference for PANDA. "
            "Input: one patient transcriptome and one or more SMILES. "
            "Output: drug response probability."
        )
    )
    parser.add_argument("--expression-csv", required=True)
    parser.add_argument(
        "--expression-format",
        choices=["auto", "wide", "long"],
        default="auto",
        help="wide: one row with gene columns; long: columns gene/expression.",
    )
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--gene-col", default="gene")
    parser.add_argument("--value-col", default="expression")
    parser.add_argument("--smiles", nargs="*", default=None)
    parser.add_argument("--smiles-csv", default=None)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--drug-name-col", default=None)
    parser.add_argument("--data-root", default=str(root / "data" / "THERAPI"))
    parser.add_argument(
        "--csg2a-cache-dir",
        default=str(root / "data" / "THERAPI_CSG2A_zscore"),
        help="Used only for the CSG2A 978-gene order.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(
            root
            / "checkpoints"
            / "tcga_seed0"
        ),
        help="Directory containing fold_*.pt checkpoints. Used as an ensemble.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Single checkpoint path. Can be provided multiple times. Overrides --checkpoint-dir.",
    )
    parser.add_argument(
        "--csg2a-mode",
        choices=["exact", "zscore_approx"],
        default="exact",
        help=(
            "exact uses the external CSG2A_asg model to generate perturbed_gex and compound. "
            "zscore_approx uses raw sample-wise z-score as perturbed_gex and zeros as compound; "
            "this is not equivalent to the trained experiment."
        ),
    )
    parser.add_argument("--csg2a-root", default=str(csg2a_root))
    parser.add_argument(
        "--csg2a-ckpt",
        default=str(csg2a_root / "ckpts" / "260207_CSG2A_finetune_GDSC_MOSA_platform.pt"),
    )
    parser.add_argument(
        "--csg2a-pretrain",
        default=None,
    )
    parser.add_argument("--dose", type=float, default=0.1)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gene-hdim", type=int, default=64)
    parser.add_argument("--finetune-hdim1", type=int, default=512)
    parser.add_argument("--finetune-hdim2", type=int, default=64)
    parser.add_argument("--allow-missing-zero", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", default="THERAPI_inference_predictions.csv")
    parser.add_argument("--save-latents", action="store_true")
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def read_gene_order_from_zscore_csv(path):
    return pd.read_csv(path, index_col=0, nrows=0).columns.astype(str).tolist()


def find_csg2a_gene_order_csv(cache_dir):
    matches = sorted(Path(cache_dir).glob("THERAPI_GDSC_samplewise_zscore_*genes.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Missing THERAPI_GDSC_samplewise_zscore_*genes.csv in {cache_dir}."
        )
    return matches[-1]


def load_expression_vector(
    path,
    gene_order,
    expression_format="auto",
    sample_id=None,
    id_col=None,
    gene_col="gene",
    value_col="expression",
    allow_missing_zero=False,
):
    def normalize_tcga_id(value):
        parts = str(value).split("-")
        return "-".join(parts[:3]) if len(parts) >= 3 else str(value)

    df = pd.read_csv(path)
    lower_cols = {col.lower(): col for col in df.columns}

    if expression_format == "auto":
        if gene_col in df.columns and value_col in df.columns:
            expression_format = "long"
        elif "gene" in lower_cols and (
            "expression" in lower_cols or "value" in lower_cols
        ):
            expression_format = "long"
            gene_col = lower_cols["gene"]
            value_col = lower_cols.get("expression", lower_cols.get("value"))
        else:
            expression_format = "wide"

    if expression_format == "long":
        if gene_col not in df.columns or value_col not in df.columns:
            raise KeyError(
                f"Long expression CSV needs columns '{gene_col}' and '{value_col}'."
            )
        series = df.set_index(gene_col)[value_col].astype(float)
    else:
        working = df.copy()
        if sample_id is not None:
            if id_col is None:
                candidate_id_cols = [
                    col
                    for col in ["Sample", "sample", "ID", "id", "patient_id", "Patient"]
                    if col in working.columns
                ]
                if not candidate_id_cols:
                    raise ValueError("--sample-id was provided but no ID column was found.")
                id_col = candidate_id_cols[0]
            working = working.loc[working[id_col].astype(str) == str(sample_id)]
            if working.empty:
                sample_patient_id = normalize_tcga_id(sample_id)
                working = df.loc[
                    df[id_col].astype(str).map(normalize_tcga_id) == sample_patient_id
                ]
            if working.empty:
                raise ValueError(f"sample_id not found in expression CSV: {sample_id}")
            if len(working) > 1:
                gene_cols = [col for col in working.columns if col != id_col]
                working = pd.DataFrame(
                    [working[gene_cols].astype(float).mean(axis=0)],
                    columns=gene_cols,
                )
        if len(working) != 1:
            raise ValueError(
                f"Wide expression CSV must resolve to exactly one row, got {len(working)}."
            )
        drop_cols = [
            col
            for col in ["Sample", "sample", "ID", "id", "patient_id", "Patient", "DepMap_ID"]
            if col in working.columns
        ]
        if id_col is not None and id_col in working.columns and id_col not in drop_cols:
            drop_cols.append(id_col)
        series = working.drop(columns=drop_cols, errors="ignore").iloc[0].astype(float)

    series.index = series.index.astype(str)
    missing = [gene for gene in gene_order if gene not in series.index]
    if missing and not allow_missing_zero:
        raise ValueError(
            f"Expression is missing {len(missing)} required genes. "
            f"First missing genes: {missing[:20]}. "
            "Use --allow-missing-zero only if zero-fill is intentional."
        )
    aligned = series.reindex(gene_order, fill_value=0.0).astype(np.float32)
    return aligned.to_numpy(dtype=np.float32).reshape(1, -1)


def samplewise_zscore(values, eps=1e-6):
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=1, keepdims=True)
    std = np.maximum(values.std(axis=1, keepdims=True), eps)
    return (values - mean) / std


def load_smiles_inputs(args):
    rows = []
    if args.smiles:
        for idx, smiles in enumerate(args.smiles):
            rows.append({"drug_name": f"input_{idx}", "smiles": smiles})
    if args.smiles_csv:
        df = pd.read_csv(args.smiles_csv)
        if args.smiles_col not in df.columns:
            raise KeyError(f"SMILES CSV has no column: {args.smiles_col}")
        for idx, row in df.iterrows():
            drug_name = (
                row[args.drug_name_col]
                if args.drug_name_col is not None and args.drug_name_col in df.columns
                else f"row_{idx}"
            )
            rows.append({"drug_name": drug_name, "smiles": row[args.smiles_col]})
    if not rows:
        raise ValueError("Provide --smiles or --smiles-csv.")
    return pd.DataFrame(rows)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def infer_checkpoint_shapes(state):
    en_state = state["EN_model"]
    gcn_state = state["GCN_model"]
    csg_state = state["CSG2AState_model"]
    comp_state = state["CSG2ACompound_model"]
    raw_input_dim = int(en_state["encoder.0.weight"].shape[1])
    nz = int(en_state["fc1.weight"].shape[0])
    csg2a_input_dim = int(csg_state["net.0.weight"].shape[1])
    csg2a_hidden_dim = int(csg_state["net.0.weight"].shape[0])
    csg2a_out_dim = int(csg_state["net.4.weight"].shape[0])
    comp_input_dim = int(comp_state["net.0.weight"].shape[1])
    comp_hidden_dim = int(comp_state["net.0.weight"].shape[0])
    comp_out_dim = int(comp_state["net.4.weight"].shape[0])
    drug_in_channels = int(gcn_state["GC1.weight"].shape[0])
    d_dim = int(gcn_state["GC4.weight"].shape[1])
    fused_dim = nz + csg2a_out_dim + comp_out_dim
    return {
        "raw_input_dim": raw_input_dim,
        "nz": nz,
        "csg2a_input_dim": csg2a_input_dim,
        "csg2a_hidden_dim": csg2a_hidden_dim,
        "csg2a_out_dim": csg2a_out_dim,
        "comp_input_dim": comp_input_dim,
        "comp_hidden_dim": comp_hidden_dim,
        "comp_out_dim": comp_out_dim,
        "drug_in_channels": drug_in_channels,
        "d_dim": d_dim,
        "fused_dim": fused_dim,
    }


class PANCDRInferenceModel(nn.Module):
    def __init__(self, checkpoint_path, device):
        super().__init__()
        self.checkpoint_path = str(checkpoint_path)
        self.state = load_checkpoint(checkpoint_path, device)
        self.shapes = infer_checkpoint_shapes(self.state)

        self.EN_model = Encoder(
            self.shapes["raw_input_dim"], self.shapes["nz"], device
        ).to(device)
        self.CSG2AState_model = CSG2AFeatureEncoder(
            self.shapes["csg2a_input_dim"],
            hidden_dim=self.shapes["csg2a_hidden_dim"],
            out_dim=self.shapes["csg2a_out_dim"],
            dropout=float(self.state.get("csg2a_dropout", 0.0)),
        ).to(device)
        self.CSG2ACompound_model = CSG2AFeatureEncoder(
            self.shapes["comp_input_dim"],
            hidden_dim=self.shapes["comp_hidden_dim"],
            out_dim=self.shapes["comp_out_dim"],
            dropout=float(self.state.get("csg2a_dropout", 0.0)),
        ).to(device)
        self.GCN_model = GCN(
            self.shapes["drug_in_channels"],
            [256, 256, 256],
            h_dims=[
                self.shapes["d_dim"],
                self.shapes["fused_dim"] + self.shapes["d_dim"],
            ],
            use_dropout=bool(self.state.get("use_gcn_dropout", True)),
        ).to(device)

        self.EN_model.load_state_dict(self.state["EN_model"])
        self.CSG2AState_model.load_state_dict(self.state["CSG2AState_model"])
        self.CSG2ACompound_model.load_state_dict(self.state["CSG2ACompound_model"])
        self.GCN_model.load_state_dict(self.state["GCN_model"])

        self.eval()

    @torch.no_grad()
    def predict(self, drug_feat, drug_adj, raw_gex, csg2a_pert, csg2a_comp):
        self.eval()
        _, raw_mu, _ = self.EN_model(raw_gex)
        csg2a_state_latent = self.CSG2AState_model(csg2a_pert)
        csg2a_comp_latent = self.CSG2ACompound_model(csg2a_comp)
        fused = torch.cat((raw_mu, csg2a_state_latent, csg2a_comp_latent), dim=1)
        pred = self.GCN_model(drug_feat, drug_adj, fused)
        return pred.view(-1), {
            "raw_mu": raw_mu,
            "csg2a_state_latent": csg2a_state_latent,
            "csg2a_comp_latent": csg2a_comp_latent,
        }


def find_checkpoints(args):
    if args.checkpoint:
        paths = [Path(path) for path in args.checkpoint]
    else:
        paths = sorted(Path(args.checkpoint_dir).glob("fold_*_model.pt"))
    if not paths:
        raise FileNotFoundError(
            f"No checkpoints found. checkpoint={args.checkpoint}, checkpoint_dir={args.checkpoint_dir}"
        )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return paths


def build_pancdr_drug_tensors(smiles_list, device):
    feats = []
    adjs = []
    for smiles in smiles_list:
        feat_mat, adj_list, _ = smiles_to_graph(smiles)
        feat, adj = calculate_graph_feat_numpy(feat_mat, adj_list)
        feats.append(feat)
        adjs.append(adj)
    return (
        torch.FloatTensor(np.asarray(feats, dtype=np.float32)).to(device),
        torch.FloatTensor(np.asarray(adjs, dtype=np.float32)).to(device),
    )


def add_csg2a_to_path(csg2a_root):
    csg2a_root = Path(csg2a_root).resolve()
    if str(csg2a_root) not in sys.path:
        sys.path.insert(0, str(csg2a_root))


def load_csg2a_modules(csg2a_root):
    csg2a_root = Path(csg2a_root)
    add_csg2a_to_path(csg2a_root)
    from models.CSG2A_net_therapi import CSG2A_finetune

    utils_data_path = csg2a_root / "utils" / "utils_data.py"
    spec = importlib.util.spec_from_file_location("csg2a_utils_data", utils_data_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CSG2A utils_data from {utils_data_path}")
    utils_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_data)

    def numpy_pairwise_distances(x):
        x = np.asarray(x, dtype=np.float32)
        diff = x[:, None, :] - x[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    # Avoid sklearn/scipy pairwise_distances hangs seen in this environment.
    # This preserves the same Euclidean distance matrix semantics for CSG2A.
    utils_data.pairwise_distances = numpy_pairwise_distances
    return CSG2A_finetune, utils_data.load_data_from_smiles


def load_csg2a_genes(csg2a_root):
    path = Path(csg2a_root) / "data" / "LINCS_L1000_landmark_genes.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing CSG2A 978-gene reference: {path}")
    genes = pd.Index(
        [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    )
    if len(genes) != 978:
        raise ValueError(f"Expected 978 CSG2A Landmark genes, found {len(genes)}.")
    return genes


def build_ppi_adj(csg2a_root, genes, device):
    string_path = Path(csg2a_root) / "data" / "STRING_edges.csv"
    if not string_path.exists():
        raise FileNotFoundError(f"Missing STRING edges: {string_path}")
    string_df = pd.read_csv(string_path)
    string_df = string_df[
        string_df["source"].isin(genes) & string_df["target"].isin(genes)
    ].copy()
    string_df["source_idx"] = string_df["source"].map(lambda x: genes.get_loc(x))
    string_df["target_idx"] = string_df["target"].map(lambda x: genes.get_loc(x))
    ppi_adj = torch.eye(len(genes), dtype=torch.float32)
    for target_idx, source_idx in string_df[["target_idx", "source_idx"]].values:
        ppi_adj[target_idx, source_idx] = 1.0
        ppi_adj[source_idx, target_idx] = 1.0
    return ppi_adj.to(device)


class CSG2AGenerator:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.csg2a_root = Path(args.csg2a_root).resolve()
        self.csg2a_ckpt = Path(args.csg2a_ckpt).resolve()
        self.csg2a_pretrain = (
            Path(args.csg2a_pretrain).resolve() if args.csg2a_pretrain else None
        )
        CSG2A_finetune, self.load_data_from_smiles = load_csg2a_modules(
            self.csg2a_root
        )
        self.genes = load_csg2a_genes(self.csg2a_root)
        ppi_adj = build_ppi_adj(self.csg2a_root, self.genes, device)
        csg2a_params = {
            "gex_dim": len(self.genes),
            "hdim": args.gene_hdim,
            "dropout": args.dropout,
            "ppi_adj": ppi_adj,
            "bias": False,
        }
        self.model = CSG2A_finetune(
            gex_dim=len(self.genes),
            CSG2A_params=csg2a_params,
            finetune_hdim1=args.finetune_hdim1,
            finetune_hdim2=args.finetune_hdim2,
            dropout=args.dropout,
        ).to(device)

        if self.csg2a_pretrain is not None and self.csg2a_pretrain.exists():
            pretrained_state = torch.load(self.csg2a_pretrain, map_location=device)
            model_state = self.model.CSG2A.state_dict()
            for name, param in pretrained_state.items():
                if isinstance(param, torch.nn.Parameter):
                    param = param.data
                if name in model_state:
                    model_state[name].copy_(param)

        for param in self.model.CSG2A.parameters():
            param.requires_grad = False

        if not self.csg2a_ckpt.exists():
            raise FileNotFoundError(f"Missing CSG2A checkpoint: {self.csg2a_ckpt}")
        self.model.load_state_dict(torch.load(self.csg2a_ckpt, map_location=device))
        self.model.eval()
        self.graph_cache = {}

    def graph_from_smiles(self, smiles):
        if smiles not in self.graph_cache:
            try:
                features = self.load_data_from_smiles(
                    [smiles], add_dummy_node=True, one_hot_formal_charge=True
                )
            except BaseException:
                print("CSG2A SMILES featurization failed for:", smiles, file=sys.stderr)
                traceback.print_exc()
                raise
            if not features:
                raise ValueError(f"CSG2A could not featurize SMILES: {smiles}")
            node_feat_np, adj_np, dist_np = features[0]
            self.graph_cache[smiles] = (
                torch.from_numpy(node_feat_np).float().unsqueeze(0),
                torch.from_numpy(adj_np).float().unsqueeze(0),
                torch.from_numpy(dist_np).float().unsqueeze(0),
            )
        return self.graph_cache[smiles]

    @torch.no_grad()
    def generate(self, z_expr, smiles_list):
        pert_rows = []
        comp_rows = []
        b_gex_base = torch.from_numpy(z_expr.astype(np.float32, copy=False)).to(
            self.device
        )
        for smiles in smiles_list:
            node_feat, adj_mat, dist_mat = self.graph_from_smiles(smiles)
            node_feat = node_feat.to(self.device)
            adj_mat = adj_mat.to(self.device)
            dist_mat = dist_mat.to(self.device)
            mask = (torch.sum(torch.abs(node_feat), dim=-1) != 0).to(self.device)
            dose = torch.full((1, 1), float(self.args.dose), dtype=torch.float32).to(
                self.device
            )
            time = torch.full((1, 1), float(self.args.time), dtype=torch.float32).to(
                self.device
            )
            try:
                pred_gex, comp = self.model(
                    b_gex_base, node_feat, mask, adj_mat, dist_mat, dose, time
                )
            except BaseException:
                print("CSG2A forward failed for:", smiles, file=sys.stderr)
                print(
                    "Shapes:",
                    {
                        "gex": tuple(b_gex_base.shape),
                        "node_feat": tuple(node_feat.shape),
                        "mask": tuple(mask.shape),
                        "adj": tuple(adj_mat.shape),
                        "dist": tuple(dist_mat.shape),
                        "dose": tuple(dose.shape),
                        "time": tuple(time.shape),
                    },
                    file=sys.stderr,
                )
                traceback.print_exc()
                raise
            pert_rows.append(pred_gex.cpu().numpy()[0])
            comp_rows.append(comp.cpu().numpy()[0])
        return (
            np.asarray(pert_rows, dtype=np.float32),
            np.asarray(comp_rows, dtype=np.float32),
        )


def make_approx_csg2a_inputs(z_expr, n_smiles):
    pert = np.repeat(z_expr.astype(np.float32), repeats=n_smiles, axis=0)
    comp = np.zeros_like(pert, dtype=np.float32)
    return pert, comp


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print("Using device:", device)

    gdsc_expr, _, _ = build_shared_expression_matrices(args.data_root)
    raw_gene_order = gdsc_expr.columns.astype(str).tolist()
    csg2a_gene_order_csv = find_csg2a_gene_order_csv(args.csg2a_cache_dir)
    csg2a_gene_order = read_gene_order_from_zscore_csv(csg2a_gene_order_csv)
    print("Raw GEX genes:", len(raw_gene_order))
    print("CSG2A genes:", len(csg2a_gene_order))

    raw_gex = load_expression_vector(
        args.expression_csv,
        raw_gene_order,
        expression_format=args.expression_format,
        sample_id=args.sample_id,
        id_col=args.id_col,
        gene_col=args.gene_col,
        value_col=args.value_col,
        allow_missing_zero=args.allow_missing_zero,
    )
    csg2a_expr = load_expression_vector(
        args.expression_csv,
        csg2a_gene_order,
        expression_format=args.expression_format,
        sample_id=args.sample_id,
        id_col=args.id_col,
        gene_col=args.gene_col,
        value_col=args.value_col,
        allow_missing_zero=args.allow_missing_zero,
    )
    csg2a_z = samplewise_zscore(csg2a_expr)

    smiles_df = load_smiles_inputs(args)
    smiles_list = smiles_df["smiles"].astype(str).tolist()
    n_drugs = len(smiles_list)
    print("Input drugs:", n_drugs)

    drug_feat, drug_adj = build_pancdr_drug_tensors(smiles_list, device)
    raw_gex_batch = torch.FloatTensor(np.repeat(raw_gex, n_drugs, axis=0)).to(device)

    if args.csg2a_mode == "exact":
        csg2a_generator = CSG2AGenerator(args, device)
        csg2a_pert, csg2a_comp = csg2a_generator.generate(csg2a_z, smiles_list)
        csg2a_note = "exact_csg2a_asg"
    else:
        csg2a_pert, csg2a_comp = make_approx_csg2a_inputs(csg2a_z, n_drugs)
        csg2a_note = "approx_zscore_pert_zero_compound_not_equivalent_to_training"
        print("WARNING:", csg2a_note)

    csg2a_pert_t = torch.FloatTensor(csg2a_pert).to(device)
    csg2a_comp_t = torch.FloatTensor(csg2a_comp).to(device)

    checkpoint_paths = find_checkpoints(args)
    print("Loaded checkpoints:", len(checkpoint_paths))
    fold_preds = []
    latent_dump = {}
    for checkpoint_path in checkpoint_paths:
        model = PANCDRInferenceModel(checkpoint_path, device).to(device)
        pred, latents = model.predict(
            drug_feat, drug_adj, raw_gex_batch, csg2a_pert_t, csg2a_comp_t
        )
        fold_preds.append(pred.detach().cpu().numpy())
        if args.save_latents:
            latent_dump[Path(checkpoint_path).stem] = {
                name: value.detach().cpu().numpy().tolist()
                for name, value in latents.items()
            }

    pred_matrix = np.vstack(fold_preds)
    out_df = smiles_df.copy()
    out_df["response_probability"] = pred_matrix.mean(axis=0)
    out_df["response_probability_std_across_checkpoints"] = pred_matrix.std(axis=0, ddof=1) if len(checkpoint_paths) > 1 else 0.0
    out_df["n_checkpoints"] = len(checkpoint_paths)
    out_df["csg2a_mode"] = csg2a_note
    for idx, checkpoint_path in enumerate(checkpoint_paths):
        out_df[f"pred_{Path(checkpoint_path).stem}"] = pred_matrix[idx]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True) if out_path.parent != Path(".") else None
    out_df.to_csv(out_path, index=False)
    print("Saved predictions:", out_path)

    if args.save_latents:
        latent_path = out_path.with_suffix(".latents.json")
        latent_path.write_text(json.dumps(latent_dump), encoding="utf-8")
        print("Saved latents:", latent_path)


if __name__ == "__main__":
    main()
