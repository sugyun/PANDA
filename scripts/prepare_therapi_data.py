"""Install required tables from a downloaded THERAPI data directory."""

import argparse
import shutil
from pathlib import Path


MAPPINGS = {
    "GDSC/GDSC_gex.csv": "THERAPI/GDSC_gex.csv",
    "GDSC/GDSC_info.csv": "THERAPI/GDSC_info.csv",
    "TCGA/TCGA_unlabeled_gex.csv": "THERAPI/TCGA_unlabeled_gex.csv",
    "TCGA/TCGA_unlabeled_info.csv": "THERAPI/TCGA_unlabeled_info.csv",
    "TCGA/TCGA_labeled_gex.csv": "THERAPI/TCGA_labeled_gex.csv",
    "TCGA/TCGA_labeled_info.csv": "THERAPI/TCGA_labeled_info.csv",
    "External/External_unlabeled_gex.csv": "External/External_unlabeled_gex.csv",
    "External/External_unlabeled_info.csv": "External/External_unlabeled_info.csv",
    "External/External_labeled_gex.csv": "External/External_labeled_gex.csv",
    "External/External_labeled_info.csv": "External/External_labeled_info.csv",
}


def resolve_response(source, dataset):
    candidates = [
        source / dataset / f"{dataset}_Drug_SMILES_Response.csv",
        source / dataset / f"{dataset}_Durg_SMILES_Response.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing {dataset} drug-response table. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=root / "data")
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)

    resolved = [(source / relative, destination / target) for relative, target in MAPPINGS.items()]
    resolved.extend(
        [
            (resolve_response(source, "GDSC"), destination / "THERAPI" / "GDSC_Durg_SMILES_Response.csv"),
            (resolve_response(source, "TCGA"), destination / "THERAPI" / "TCGA_Durg_SMILES_Response.csv"),
        ]
    )
    missing = [str(path) for path, _ in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required THERAPI files:\n" + "\n".join(missing))

    for input_path, output_path in resolved:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        print(f"{input_path} -> {output_path}")

    print("THERAPI base data installed. PANDA-specific CSG2A caches are not copied.")


if __name__ == "__main__":
    main()
