# Data schema adapted from THERAPI: https://github.com/Sunginyoung/THERAPI
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from rdkit import Chem


MAX_ATOMS = 100

ATOM_SYMBOLS = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "Unknown",
]


def one_hot_with_unknown(value, choices):
    if value not in choices:
        value = choices[-1]
    return [float(value == choice) for choice in choices]


def atom_features(atom):
    feature = []
    feature += one_hot_with_unknown(atom.GetSymbol(), ATOM_SYMBOLS)
    feature += one_hot_with_unknown(atom.GetDegree(), list(range(11)))
    feature += one_hot_with_unknown(atom.GetTotalNumHs(), list(range(11)))
    feature += one_hot_with_unknown(atom.GetImplicitValence(), list(range(11)))
    feature.append(float(atom.GetFormalCharge()))
    feature.append(float(atom.GetNumRadicalElectrons()))
    feature.append(float(atom.GetIsAromatic()))
    return np.asarray(feature, dtype=np.float32)


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    feat_mat = np.stack([atom_features(atom) for atom in mol.GetAtoms()], axis=0)
    adj_list = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        adj_list[begin_idx].append(end_idx)
        adj_list[end_idx].append(begin_idx)

    degree_list = [len(neighbors) for neighbors in adj_list]
    return feat_mat, adj_list, degree_list


def tcga_patient_id(sample_id):
    parts = str(sample_id).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else str(sample_id)


def load_expression_table(path, id_col, normalize_tcga=False):
    df = pd.read_csv(path)
    df[id_col] = df[id_col].astype(str)
    if normalize_tcga:
        df[id_col] = df[id_col].map(tcga_patient_id)

    gene_cols = [col for col in df.columns if col != id_col]
    expr_df = df.groupby(id_col, as_index=True)[gene_cols].mean()
    return expr_df


def build_shared_expression_matrices(data_root):
    gdsc_expr = load_expression_table(
        os.path.join(data_root, "GDSC_gex.csv"),
        id_col="DepMap_ID",
        normalize_tcga=False,
    )
    tcga_unlabeled_expr = load_expression_table(
        os.path.join(data_root, "TCGA_unlabeled_gex.csv"),
        id_col="Sample",
        normalize_tcga=True,
    )
    tcga_labeled_expr = load_expression_table(
        os.path.join(data_root, "TCGA_labeled_gex.csv"),
        id_col="Sample",
        normalize_tcga=True,
    )

    common_genes = [
        gene
        for gene in gdsc_expr.columns
        if gene in tcga_unlabeled_expr.columns and gene in tcga_labeled_expr.columns
    ]
    if not common_genes:
        raise ValueError("No common genes were found across THERAPI expression tables.")

    return (
        gdsc_expr[common_genes],
        tcga_unlabeled_expr[common_genes],
        tcga_labeled_expr[common_genes],
    )


def build_smiles_feature_bank(gdsc_resp_df, tcga_resp_df):
    all_smiles = pd.concat(
        [gdsc_resp_df["Canonical_SMILES"], tcga_resp_df["Canonical_SMILES"]],
        axis=0,
    ).dropna()
    unique_smiles = list(OrderedDict.fromkeys(all_smiles.astype(str).tolist()))

    drug_feature = {}
    smiles_to_drug_id = {}
    skipped = []
    for idx, smiles in enumerate(unique_smiles):
        try:
            feat_mat, adj_list, degree_list = smiles_to_graph(smiles)
        except Exception:
            skipped.append(smiles)
            continue

        drug_id = f"SMI_{idx:05d}"
        smiles_to_drug_id[smiles] = drug_id
        drug_feature[drug_id] = [feat_mat, adj_list, degree_list]

    return drug_feature, smiles_to_drug_id, skipped


def normalize_adj_numpy(adj):
    adj = adj.astype("float32", copy=False) + np.eye(adj.shape[0], dtype="float32")
    degree = adj.sum(axis=1)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.power(degree, -0.5).astype("float32")
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    return d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]


def calculate_graph_feat_numpy(feat_mat, adj_list):
    if feat_mat.shape[0] != len(adj_list):
        raise ValueError(
            f"Feature rows ({feat_mat.shape[0]}) and adjacency list ({len(adj_list)}) mismatch."
        )
    if feat_mat.shape[0] > MAX_ATOMS:
        raise ValueError(
            f"Drug graph has {feat_mat.shape[0]} atoms, larger than MAX_ATOMS={MAX_ATOMS}."
        )

    feat = np.zeros((MAX_ATOMS, feat_mat.shape[-1]), dtype="float32")
    adj_mat = np.zeros((MAX_ATOMS, MAX_ATOMS), dtype="float32")
    feat[: feat_mat.shape[0], :] = feat_mat
    for i, nodes in enumerate(adj_list):
        for node in nodes:
            adj_mat[i, int(node)] = 1.0
    if not np.allclose(adj_mat, adj_mat.T):
        raise ValueError("Drug adjacency matrix is not symmetric.")

    n_atoms = len(adj_list)
    adj_mat[:n_atoms, :n_atoms] = normalize_adj_numpy(adj_mat[:n_atoms, :n_atoms])
    if n_atoms < MAX_ATOMS:
        adj_mat[n_atoms:, n_atoms:] = normalize_adj_numpy(adj_mat[n_atoms:, n_atoms:])
    return feat, adj_mat
