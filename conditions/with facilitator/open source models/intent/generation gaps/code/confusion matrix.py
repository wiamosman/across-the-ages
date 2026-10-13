
import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================================
# CONFIGURATION
# ==========================================================
FACILITATOR_DIRS = [
    "/Users/apple/StudioProjects/last_project_emnlp/with facilitator/generation gaps/results/batch_1_GGI_Results",
    "/Users/apple/StudioProjects/last_project_emnlp/with facilitator/generation gaps/results/batch_2_GGI_Results"
]

BASELINE_DIRS = [
    "/Users/apple/StudioProjects/last_project_emnlp/without facilitator/other models/generation_gaps/results/batch_1_GGI_Results",
    "/Users/apple/StudioProjects/last_project_emnlp/without facilitator/other models/generation_gaps/results/batch_2_GGI_Results"
]

GGI_LABELS = [
    'slang mismatch', 'perceived disrespect', 'stereotyped expectation',
    'ignored input', 'method divergence', 'digital disparity', 'none'
]

# ==========================================================
# LOAD PREDICTIONS
# ==========================================================

def load_predictions(result_dirs):
    model_data = defaultdict(lambda: {'gold': [], 'pred': []})

    for result_dir in result_dirs:
        if not os.path.exists(result_dir):
            continue

        for folder in os.listdir(result_dir):
            folder_path = os.path.join(result_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            model_name = folder

            for fn in os.listdir(folder_path):
                if not fn.endswith('.json'):
                    continue

                file_path = os.path.join(folder_path, fn)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for u in data.get('utterances', []):
                    gold = u.get('gold_label', '').lower().strip()
                    pred = u.get('predicted_label', '').lower().strip()

                    if pred not in GGI_LABELS:
                        pred = 'none'

                    if gold in GGI_LABELS:
                        model_data[model_name]['gold'].append(gold)
                        model_data[model_name]['pred'].append(pred)

    return model_data


# ==========================================================
# FULL CONFUSION MATRIX + ALL METRICS
# ==========================================================

def compute_full_metrics(gold, pred):
    """Compute every metric for every class."""

    # Confusion matrix
    cm = confusion_matrix(gold, pred, labels=GGI_LABELS)

    # Per-class metrics
    report = classification_report(gold, pred, labels=GGI_LABELS, output_dict=True, zero_division=0)

    return cm, report


def print_confusion_matrix(cm, labels, model_name, condition):
    print(f"\n{'='*80}")
    print(f"📊 CONFUSION MATRIX: {model_name} | {condition}")
    print(f"{'='*80}")

    # Header
    header = f"{'Gold/Pred':>20}"
    for label in labels:
        header += f" {label[:8]:>8}"
    header += f" {'Total':>8}"
    print(header)
    print("-" * (20 + 9 * len(labels) + 9))

    # Rows
    for i, gold_label in enumerate(labels):
        row = f"{gold_label[:18]:>20}"
        row_total = 0
        for j in range(len(labels)):
            row += f" {cm[i][j]:>8}"
            row_total += cm[i][j]
        row += f" {row_total:>8}"
        print(row)

    # Column totals
    print("-" * (20 + 9 * len(labels) + 9))
    col_totals = [sum(cm[i][j] for i in range(len(labels))) for j in range(len(labels))]
    total_row = f"{'PREDICTED TOTAL':>20}"
    for total in col_totals:
        total_row += f" {total:>8}"
    total_row += f" {sum(col_totals):>8}"
    print(total_row)


def print_detailed_report(report, model_name, condition):
    print(f"\n{'='*80}")
    print(f"📋 DETAILED CLASSIFICATION REPORT: {model_name} | {condition}")
    print(f"{'='*80}")
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 70)

    for label in GGI_LABELS:
        if label in report:
            r = report[label]
            print(f"{label:<25} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {int(r['support']):>10}")

    print("-" * 70)
    # Macro / weighted averages
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in report:
            r = report[avg_type]
            print(f"{avg_type:<25} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {int(r['support']):>10}")


def save_confusion_matrices_to_csv(all_results, output_file):
    """Save all confusion matrices as structured CSV."""
    rows = []

    for result in all_results:
        model = result['model']
        condition = result['condition']
        cm = result['cm']

        for i, gold_label in enumerate(GGI_LABELS):
            for j, pred_label in enumerate(GGI_LABELS):
                rows.append({
                    'condition': condition,
                    'model': model,
                    'gold_label': gold_label,
                    'predicted_label': pred_label,
                    'count': cm[i][j]
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    return df


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    all_results = []

    for dirs, condition in [(FACILITATOR_DIRS, "With_Facilitator"), (BASELINE_DIRS, "Baseline")]:
        print(f"\n{'#'*80}")
        print(f"# {condition.upper()}")
        print(f"{'#'*80}")

        model_data = load_predictions(dirs)

        for model_name in sorted(model_data.keys()):
            gold = model_data[model_name]['gold']
            pred = model_data[model_name]['pred']

            if len(gold) == 0:
                continue

            cm, report = compute_full_metrics(gold, pred)

            print_confusion_matrix(cm, GGI_LABELS, model_name, condition)
            print_detailed_report(report, model_name, condition)

            all_results.append({
                'model': model_name,
                'condition': condition,
                'cm': cm,
                'report': report,
                'gold': gold,
                'pred': pred
            })

    # Save all confusion matrices
    cm_file = "/Users/apple/StudioProjects/last_project_emnlp/all_confusion_matrices.csv"
    save_confusion_matrices_to_csv(all_results, cm_file)
    print(f"\n✅ All confusion matrices saved to: {cm_file}")

    # Also save per-model detailed reports
    for result in all_results:
        model = result['model']
        condition = result['condition']
        report = result['report']

        report_rows = []
        for label in GGI_LABELS:
            if label in report:
                r = report[label]
                report_rows.append({
                    'class': label,
                    'precision': r['precision'],
                    'recall': r['recall'],
                    'f1_score': r['f1-score'],
                    'support': int(r['support'])
                })

        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report:
                r = report[avg_type]
                report_rows.append({
                    'class': avg_type,
                    'precision': r['precision'],
                    'recall': r['recall'],
                    'f1_score': r['f1-score'],
                    'support': int(r['support'])
                })

        df = pd.DataFrame(report_rows)
        out_file = f"/Users/apple/StudioProjects/last_project_emnlp/report_{condition}_{model}.csv"
        df.to_csv(out_file, index=False)
        print(f"   📁 {model} ({condition}) → {out_file}")
