"""
Validation utilities for comparing model output to ground truth data.

This module reads generated JSON labels, loads ground truth labels from an Excel file,
normalizes both sides, compares them case-by-case, and writes out text and JSON
reports including accuracy metrics and details for mismatches.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

import pandas as pd

import config as cfg
from utils import normalize_labels
from convert import convert
from lazy_excel import LazyExcelReader

def normalize_gt(label_str: Any) -> set:
    """Normalize a ground truth label string into a set of uppercase labels."""
    # Treat missing values as explicit negative / none.
    if pd.isna(label_str):
        return {"NONE"}

    label_str = str(label_str).strip()

    # Some spreadsheets use the word 'negative' to signify no labels.
    if label_str.lower() == "negative":
        return {"NONE"}

    # Split comma-separated labels, normalize whitespace and uppercase them.
    labels = [label.strip().upper() for label in label_str.split(",")]
    return set(labels)


def normalize_generated(generated_list: Any) -> set:
    """Normalize generated output using shared normalization logic from utils."""
    # normalize_labels already handles lists, strings, and other formats.
    return set(normalize_labels(generated_list))


def _process_case_comparison(gen_case: Dict[str, Any], gt_case: Dict[str, Any] | None, case_id: str) -> Dict[str, Any]:
    """Process a single case comparison and return results for all fields.
    
    Args:
        gen_case: Generated case data
        gt_case: Ground truth case data (None if not found)
        case_id: Case identifier
    
    Returns:
        Dictionary with results for each field
    """
    field_results = {
        "CT_GT": [],
        "CTA_GT": [],
        "CTP_GT": [],
        "Combined_GT": [],
    }
    
    if gt_case is None:
        return {"case_id": case_id, "found": False, "results": field_results}
    
    for field in field_results:
        gen_set = normalize_generated(gen_case.get(field, ["NONE"]))
        gt_set = normalize_gt(gt_case.get(field))
        match = gen_set == gt_set
        
        field_results[field].append({
            "case_id": case_id,
            "generated": sorted(gen_set),
            "ground_truth": sorted(gt_set),
            "match": match,
        })
    
    return {"case_id": case_id, "found": True, "results": field_results}


def check_answers(
    json_file: str,
    ground_truth_file: str,
    report_path: str = "Files/report.txt",
    json_report_path: str = "Files/report.json",
) -> Dict[str, Any]:
    """Load generated data and ground truth, compare them case-by-case, and generate reports."""

    def convert_json_even_without_gt() -> None:
        """Still convert labeled_cases.json to Excel even when validation is skipped."""
        json_path = Path(json_file)
        try:
            excel_path = convert(str(json_path))
            print(f"Converted to Excel: {excel_path}")
        except Exception as e:
            print(f"⚠️ Could not convert JSON to Excel: {e}")

    def skipped_validation_return(reason: str) -> Dict[str, Any]:
        return {
            "overall_accuracy": 0.0,
            "total_correct": 0,
            "total_cases": 0,
            "field_results": {},
            "skipped": True,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Check ground truth path safely.
    # ------------------------------------------------------------------
    if ground_truth_file is None or str(ground_truth_file).strip() == "":
        print("⚠️ No ground truth file configured. Skipping validation.")
        convert_json_even_without_gt()
        return skipped_validation_return("No ground truth file configured")

    gt_path = Path(ground_truth_file)

    if not gt_path.exists():
        print(f"⚠️ Ground truth file not found: {ground_truth_file}")
        print("Skipping validation.")
        convert_json_even_without_gt()
        return skipped_validation_return("Ground truth file not found")

    if not gt_path.is_file():
        print(f"⚠️ Ground truth path is not a file: {ground_truth_file}")
        print("Skipping validation.")
        convert_json_even_without_gt()
        return skipped_validation_return("Ground truth path is not a file")

    if gt_path.suffix.lower() not in {".xlsx", ".xlsm", ".xls", ".csv"}:
        print(f"⚠️ Ground truth file is not an Excel/CSV file: {ground_truth_file}")
        print("Skipping validation.")
        convert_json_even_without_gt()
        return skipped_validation_return("Ground truth file is not an Excel/CSV file")

    with open(json_file, "r", encoding="utf-8") as f:
        generated = json.load(f)

    # Read the ground truth spreadsheet lazily to minimize memory usage.
    try:
        reader = LazyExcelReader(ground_truth_file, chunk_size=50)
        print(f"Reading ground truth file in chunks ({reader.get_stats()})")
    except Exception as e:
        print(f"⚠️ Failed to initialize lazy reader: {e}, falling back to standard read")
        reader = None

    # Build a lookup table from case_id / case name to ground truth labels.
    gt_dict = {}

    if reader:
        for row_dict in reader.read_columns_chunked([
            "Case Name",
            "CT GT",
            "CTA GT",
            "CTP GT",
            "Combined GT",
            "CT",
            "CTA",
            "CTP",
            "Combined",
        ]):
            case_name = str(row_dict.get("Case Name", "")).strip()

            if not case_name:
                continue

            if pd.isna(row_dict.get("CT GT")):
                gt_dict[case_name] = {
                    "CT_GT": row_dict.get("CT"),
                    "CTA_GT": row_dict.get("CTA"),
                    "CTP_GT": row_dict.get("CTP"),
                    "Combined_GT": row_dict.get("Combined"),
                }
            else:
                gt_dict[case_name] = {
                    "CT_GT": row_dict.get("CT GT"),
                    "CTA_GT": row_dict.get("CTA GT"),
                    "CTP_GT": row_dict.get("CTP GT"),
                    "Combined_GT": row_dict.get("Combined GT"),
                }

    else:
        # Fallback to standard pandas read.
        if gt_path.suffix.lower() == ".csv":
            df_gt = pd.read_csv(ground_truth_file)
        else:
            df_gt = pd.read_excel(ground_truth_file)

        for _, row in df_gt.iterrows():
            case_name = str(row.get("Case Name", "")).strip()

            if not case_name:
                continue

            if not pd.isna(row.get("CT GT")):
                gt_dict[case_name] = {
                    "CT_GT": row.get("CT GT"),
                    "CTA_GT": row.get("CTA GT"),
                    "CTP_GT": row.get("CTP GT"),
                    "Combined_GT": row.get("Combined GT"),
                }
            else:
                gt_dict[case_name] = {
                    "CT_GT": row.get("CT"),
                    "CTA_GT": row.get("CTA"),
                    "CTP_GT": row.get("CTP"),
                    "Combined_GT": row.get("Combined"),
                }

    # Initialize result counters and detail storage for each evaluated field.
    results = {
        "CT_GT": {"correct": 0, "total": 0, "details": []},
        "CTA_GT": {"correct": 0, "total": 0, "details": []},
        "CTP_GT": {"correct": 0, "total": 0, "details": []},
        "Combined_GT": {"correct": 0, "total": 0, "details": []},
    }

    # Process cases in parallel using ThreadPoolExecutor.
    max_workers = 8
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for gen_case in generated:
            case_id = str(gen_case.get("case_id"))
            gt_case = gt_dict.get(case_id)

            if gt_case is None:
                print(f"⚠️ Case {case_id} not found in ground truth")
                continue

            future = executor.submit(_process_case_comparison, gen_case, gt_case, case_id)
            futures[future] = case_id

        for future in as_completed(futures):
            try:
                case_result = future.result()

                if case_result["found"]:
                    for field, comparisons in case_result["results"].items():
                        for comparison in comparisons:
                            results[field]["total"] += 1
                            if comparison["match"]:
                                results[field]["correct"] += 1
                            results[field]["details"].append(comparison)

            except Exception as e:
                print(f"⚠️ Error processing case {futures[future]}: {e}")

    print_validation_summary(results)
    write_report(results, report_path=report_path)
    write_json_report(results, json_report_path=json_report_path)

    convert_json_even_without_gt()

    return build_validation_return(results)

def print_validation_summary(results: Dict[str, Any]) -> None:
    """Print a compact summary of validation metrics and a few mismatches."""
    print("\n" + "=" * 70)
    print("ANSWER VALIDATION RESULTS")
    print("=" * 70)

    total_correct = 0
    total_cases = 0

    for field in ["CT_GT", "CTA_GT", "CTP_GT", "Combined_GT"]:
        correct = results[field]["correct"]
        total = results[field]["total"]
        accuracy = (correct / total * 100) if total else 0

        total_correct += correct
        total_cases += total

        print(f"\n{field}: {correct}/{total} ({accuracy:.1f}%)")
        mismatches = [d for d in results[field]["details"] if not d["match"]]

        # Print up to 10 mismatches for quick inspection.
        for mismatch in mismatches[:10]:
            print(f"  - {mismatch['case_id']}")
            print(f"    Generated:    {mismatch['generated']}")
            print(f"    Ground truth: {mismatch['ground_truth']}")

        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")

    overall_accuracy = (total_correct / total_cases * 100) if total_cases else 0
    print(f"\nOVERALL ACCURACY: {total_correct}/{total_cases} ({overall_accuracy:.1f}%)")
    print("=" * 70)


def write_report(results: Dict[str, Any], report_path: str = "Files/report.txt"):
    """Write the human-readable validation report to a text file."""
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with report_file.open("w", encoding="utf-8") as f:
        f.write("\n=== Case Reports ===\n")
        total_correct = 0
        total_cases = 0

        for field in ["CT_GT", "CTA_GT", "CTP_GT", "Combined_GT"]:
            correct = results[field]["correct"]
            total = results[field]["total"]
            accuracy = (correct / total * 100) if total else 0

            total_correct += correct
            total_cases += total

            f.write(f"\n{field}: {correct}/{total} ({accuracy:.1f}%)\n")
            mismatches = [d for d in results[field]["details"] if not d["match"]]

            for mismatch in mismatches[:10]:
                f.write(f"  - {mismatch['case_id']}\n")
                f.write(f"    Generated:    {mismatch['generated']}\n")
                f.write(f"    Ground truth: {mismatch['ground_truth']}\n")

            if len(mismatches) > 10:
                f.write(f"  ... and {len(mismatches) - 10} more\n")

        overall_accuracy = (total_correct / total_cases * 100) if total_cases else 0
        f.write(f"\nOVERALL ACCURACY: {total_correct}/{total_cases} ({overall_accuracy:.1f}%)\n")
        f.write("=" * 70)


def write_json_report(results: Dict[str, Any], json_report_path: str = "Files/report.json") -> None:
    """Write the validation results to a JSON file for programmatic consumption."""
    report_file = Path(json_report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "summary": {},
        "overall_accuracy": 0,
        "field_results": results,
    }

    total_correct = 0
    total_cases = 0

    for field in ["CT_GT", "CTA_GT", "CTP_GT", "Combined_GT"]:
        correct = results[field]["correct"]
        total = results[field]["total"]
        accuracy = (correct / total * 100) if total else 0

        total_correct += correct
        total_cases += total

        output["summary"][field] = {
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy, 1),
        }

    output["overall_accuracy"] = round((total_correct / total_cases * 100) if total_cases else 0, 1)

    with report_file.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def build_validation_return(results: Dict[str, Any]) -> Dict[str, Any]:
    """Build a return structure summarizing the validation results."""
    total_correct = sum(results[field]["correct"] for field in results)
    total_cases = sum(results[field]["total"] for field in results)
    overall_accuracy = (total_correct / total_cases * 100) if total_cases else 0

    return {
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_cases": total_cases,
        "field_results": results,
    }



def run_from_config() -> Dict[str, Any]:
    """Validate the generated JSON and ground truth selected in config.py."""

    generated = getattr(
        cfg,
        "VALIDATION_GENERATED_JSON_FILE",
        getattr(cfg, "OUTPUT_JSON_FILE", ""),
    )
    ground_truth = getattr(
        cfg,
        "VALIDATION_GROUND_TRUTH_FILE",
        getattr(cfg, "GROUND_TRUTH_FILE", ""),
    )
    report_path = getattr(
        cfg,
        "VALIDATION_TEXT_REPORT_FILE",
        cfg.FILES_DIR / "report.txt",
    )
    json_report_path = getattr(
        cfg,
        "VALIDATION_JSON_REPORT_FILE",
        cfg.FILES_DIR / "report.json",
    )

    if not str(generated).strip():
        raise ValueError("config.VALIDATION_GENERATED_JSON_FILE cannot be blank.")
    return check_answers(
        json_file=str(generated),
        ground_truth_file=str(ground_truth),
        report_path=str(report_path),
        json_report_path=str(json_report_path),
    )


def reject_command_line_arguments(arguments: list[str] | None = None) -> None:
    """Keep config.py as the only standalone validation control surface."""

    supplied = list(sys.argv[1:] if arguments is None else arguments)
    if supplied:
        raise SystemExit(
            "Command-line arguments are disabled. Edit the validation settings "
            f"in config.py instead. Unexpected arguments: {supplied}"
        )


def main() -> None:
    reject_command_line_arguments()
    try:
        run_from_config()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Validation configuration error: {exc}") from exc


if __name__ == "__main__":
    main()
