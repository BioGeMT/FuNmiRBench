"""Validate that experiment DE tables referenced by metadata are benchmark-ready."""

import argparse
import logging
import math
import pathlib
import re
import sys
from dataclasses import dataclass

import pandas as pd

from funmirbench.de_table import find_gene_id_column, read_de_table
from funmirbench.logger import parse_log_level, setup_logging


logger = logging.getLogger(__name__)
REGISTRY_ISSUE_ID = "<registry>"
REQUIRED_REGISTRY_COLUMNS = (
    "id",
    "mirna_name",
    "organism",
    "experiment_type",
    "de_table_path",
)
REQUIRED_DE_COLUMNS = ("gene_id", "logFC", "FDR")
VALID_PERTURBATIONS = {"OE", "KO", "KD"}
MAX_LOGGED_ISSUES = 20
_ENSEMBL_GENE_ID = re.compile(r"^ENS[A-Z]*G\d+(?:\.\d+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class ValidationIssue:
    dataset_id: str
    check: str
    message: str
    path: str = ""


@dataclass(frozen=True)
class ValidationSummary:
    total: int
    files_present: int
    benchmark_ready: int
    issues: list[ValidationIssue]

    @property
    def ok(self) -> bool:
        return not self.issues


def resolve_de_table_path(
    de_table_path: str | pathlib.Path,
    *,
    root: pathlib.Path | None = None,
) -> pathlib.Path:
    path = pathlib.Path(de_table_path)
    if path.is_absolute():
        return path

    if root is not None:
        return (root / path).resolve()

    return (pathlib.Path.cwd() / path).resolve()


def _text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _duplicate_values(values: pd.Series) -> list[str]:
    normalized = values.map(_text)
    normalized = normalized[normalized != ""]
    return sorted(normalized[normalized.duplicated()].unique().tolist())


def _normalize_de_table(path: pathlib.Path) -> pd.DataFrame:
    de = read_de_table(path)
    gene_src = find_gene_id_column(de)
    if gene_src == "__index__":
        de = de.copy()
        de.insert(0, "gene_id", de.index.astype(str))
    else:
        de = de.rename(columns={gene_src: "gene_id"})
    return de


def _finite_numeric(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.map(lambda value: pd.notna(value) and math.isfinite(float(value)))
    return numeric, finite


def _expected_effect(logfc: pd.Series, perturbation: str) -> pd.Series:
    perturbation = perturbation.upper()
    if perturbation == "OE":
        return -logfc
    return logfc


def _validate_de_table(
    *,
    dataset_id: str,
    perturbation: str,
    path: pathlib.Path,
    fdr_threshold: float,
    abs_logfc_threshold: float,
) -> list[ValidationIssue]:
    issues = []
    try:
        de = _normalize_de_table(path)
    except Exception as exc:
        return [
            ValidationIssue(
                dataset_id,
                "readable_de_table",
                f"DE table could not be read: {exc}",
                str(path),
            )
        ]

    missing = [column for column in REQUIRED_DE_COLUMNS if column not in de.columns]
    if missing:
        return [
            ValidationIssue(
                dataset_id,
                "required_de_columns",
                f"DE table is missing required columns: {missing}",
                str(path),
            )
        ]

    if de.empty:
        issues.append(
            ValidationIssue(
                dataset_id,
                "nonempty_de_table",
                "DE table has no rows.",
                str(path),
            )
        )
        return issues

    gene_ids = de["gene_id"].map(_text)
    blank_gene_count = int((gene_ids == "").sum())
    if blank_gene_count:
        issues.append(
            ValidationIssue(
                dataset_id,
                "gene_id_values",
                f"DE table has {blank_gene_count} blank gene_id value(s).",
                str(path),
            )
        )

    duplicate_gene_count = int(gene_ids[gene_ids != ""].duplicated().sum())
    if duplicate_gene_count:
        issues.append(
            ValidationIssue(
                dataset_id,
                "duplicate_gene_ids",
                f"DE table has {duplicate_gene_count} duplicate gene_id value(s).",
                str(path),
            )
        )

    non_ensembl_count = int(
        (~gene_ids[gene_ids != ""].str.match(_ENSEMBL_GENE_ID)).sum()
    )
    if non_ensembl_count:
        issues.append(
            ValidationIssue(
                dataset_id,
                "ensembl_gene_ids",
                f"DE table has {non_ensembl_count} non-Ensembl gene_id value(s).",
                str(path),
            )
        )

    logfc, valid_logfc = _finite_numeric(de["logFC"])
    invalid_logfc_count = int((~valid_logfc).sum())
    if invalid_logfc_count:
        issues.append(
            ValidationIssue(
                dataset_id,
                "numeric_logfc",
                f"DE table has {invalid_logfc_count} non-numeric or non-finite logFC value(s).",
                str(path),
            )
        )

    fdr, valid_fdr = _finite_numeric(de["FDR"])
    invalid_fdr_count = int((~valid_fdr).sum())
    if invalid_fdr_count:
        issues.append(
            ValidationIssue(
                dataset_id,
                "numeric_fdr",
                f"DE table has {invalid_fdr_count} non-numeric or non-finite FDR value(s).",
                str(path),
            )
        )

    valid_fdr_values = fdr[valid_fdr]
    out_of_range_fdr_count = int(
        ((valid_fdr_values <= 0.0) | (valid_fdr_values > 1.0)).sum()
    )
    if out_of_range_fdr_count:
        issues.append(
            ValidationIssue(
                dataset_id,
                "fdr_range",
                f"DE table has {out_of_range_fdr_count} FDR value(s) outside (0, 1].",
                str(path),
            )
        )

    if issues:
        return issues

    effect = _expected_effect(logfc.astype(float), perturbation)
    positives = int((
        (fdr.astype(float) < float(fdr_threshold))
        & (effect > float(abs_logfc_threshold))
    ).sum())
    negatives = int(len(de) - positives)
    if positives == 0:
        issues.append(
            ValidationIssue(
                dataset_id,
                "ground_truth_classes",
                (
                    "DE table has no positive genes under "
                    f"FDR<{fdr_threshold} and effect>{abs_logfc_threshold}."
                ),
                str(path),
            )
        )
    if negatives == 0:
        issues.append(
            ValidationIssue(
                dataset_id,
                "ground_truth_classes",
                (
                    "DE table has no negative genes under "
                    f"FDR<{fdr_threshold} and effect>{abs_logfc_threshold}."
                ),
                str(path),
            )
        )
    return issues


def _registry_issues(df: pd.DataFrame) -> list[ValidationIssue]:
    issues = []
    missing_columns = [
        column for column in REQUIRED_REGISTRY_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        issues.append(
            ValidationIssue(
                REGISTRY_ISSUE_ID,
                "registry_columns",
                f"Experiment registry is missing required columns: {missing_columns}",
            )
        )
        return issues

    duplicate_ids = _duplicate_values(df["id"])
    if duplicate_ids:
        issues.append(
            ValidationIssue(
                REGISTRY_ISSUE_ID,
                "duplicate_dataset_ids",
                f"Experiment registry has duplicate id value(s): {duplicate_ids}",
            )
        )

    duplicate_paths = _duplicate_values(df["de_table_path"])
    if duplicate_paths:
        issues.append(
            ValidationIssue(
                REGISTRY_ISSUE_ID,
                "duplicate_de_table_paths",
                f"Experiment registry has duplicate de_table_path value(s): {duplicate_paths}",
            )
        )
    return issues


def _validate_registry_row(row: pd.Series) -> list[ValidationIssue]:
    dataset_id = _text(row.get("id")) or REGISTRY_ISSUE_ID
    issues = []
    for column in REQUIRED_REGISTRY_COLUMNS:
        if not _text(row.get(column)):
            issues.append(
                ValidationIssue(
                    dataset_id,
                    "required_registry_value",
                    f"Registry row is missing required value {column!r}.",
                )
            )

    perturbation = _text(row.get("experiment_type")).upper()
    if perturbation and perturbation not in VALID_PERTURBATIONS:
        issues.append(
            ValidationIssue(
                dataset_id,
                "experiment_type",
                (
                    f"Unsupported experiment_type {row.get('experiment_type')!r}; "
                    f"expected one of {sorted(VALID_PERTURBATIONS)}."
                ),
            )
        )
    return issues


def validate_experiments(
    experiments_tsv: str | pathlib.Path,
    *,
    root: pathlib.Path | None = None,
    fdr_threshold: float = 0.05,
    abs_logfc_threshold: float = 1.0,
) -> ValidationSummary:
    experiments_tsv = pathlib.Path(experiments_tsv).expanduser().resolve()
    root = pathlib.Path(root).expanduser().resolve() if root is not None else None
    df = pd.read_csv(experiments_tsv, sep="\t", dtype=str).fillna("")

    issues = _registry_issues(df)
    if any(issue.check == "registry_columns" for issue in issues):
        return ValidationSummary(
            total=int(len(df)),
            files_present=0,
            benchmark_ready=0,
            issues=issues,
        )

    files_present = 0
    invalid_dataset_ids = set()
    for _, row in df.iterrows():
        dataset_id = _text(row.get("id")) or REGISTRY_ISSUE_ID
        row_issues = _validate_registry_row(row)
        issues.extend(row_issues)

        path_value = _text(row.get("de_table_path"))
        if not path_value:
            invalid_dataset_ids.add(dataset_id)
            continue

        path = resolve_de_table_path(path_value, root=root)
        if not path.is_file():
            issues.append(
                ValidationIssue(
                    dataset_id,
                    "de_table_file",
                    f"DE table path does not point to a file: {path}",
                    str(path),
                )
            )
            invalid_dataset_ids.add(dataset_id)
            continue

        files_present += 1
        perturbation = _text(row.get("experiment_type")).upper()
        if perturbation in VALID_PERTURBATIONS:
            table_issues = _validate_de_table(
                dataset_id=dataset_id,
                perturbation=perturbation,
                path=path,
                fdr_threshold=fdr_threshold,
                abs_logfc_threshold=abs_logfc_threshold,
            )
            issues.extend(table_issues)
            if table_issues:
                invalid_dataset_ids.add(dataset_id)
        if row_issues:
            invalid_dataset_ids.add(dataset_id)

    if any(
        issue.check in {"duplicate_dataset_ids", "duplicate_de_table_paths"}
        for issue in issues
    ):
        invalid_dataset_ids.update(_text(value) for value in df["id"] if _text(value))

    return ValidationSummary(
        total=int(len(df)),
        files_present=files_present,
        benchmark_ready=max(0, int(len(df)) - len(invalid_dataset_ids)),
        issues=issues,
    )


def log_validation_summary(summary: ValidationSummary) -> None:
    logger.info("Datasets in metadata:      %s", summary.total)
    logger.info("Files present:             %s", summary.files_present)
    logger.info("Benchmark-ready datasets:  %s", summary.benchmark_ready)
    if summary.ok:
        return

    logger.error("Validation issues:         %s", len(summary.issues))
    for issue in summary.issues[:MAX_LOGGED_ISSUES]:
        location = f" ({issue.path})" if issue.path else ""
        logger.error(
            "  - %s [%s]: %s%s",
            issue.dataset_id,
            issue.check,
            issue.message,
            location,
        )
    remaining = len(summary.issues) - MAX_LOGGED_ISSUES
    if remaining > 0:
        logger.error("  ... %s more issue(s)", remaining)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate experiment DE table files are benchmark-ready."
    )
    parser.add_argument("--experiments-tsv", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, default=None)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--abs-logfc-threshold", type=float, default=1.0)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args(argv)

    setup_logging(parse_log_level(args.log_level))

    summary = validate_experiments(
        args.experiments_tsv,
        root=args.root,
        fdr_threshold=args.fdr_threshold,
        abs_logfc_threshold=args.abs_logfc_threshold,
    )
    log_validation_summary(summary)
    return 0 if summary.ok else 1


if __name__ == "__main__":
    sys.exit(main())
