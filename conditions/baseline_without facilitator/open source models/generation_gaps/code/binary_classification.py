
import os
import json
import pandas as pd
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

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
# BINARY METRICS COMPUTATION
# ==========================================================

def compute_binary_metrics(gold, pred):
    """
    Convert to binary: 
    - 0 = 'none' (negative)
    - 1 = any gap (positive)
    """
    gold_binary = [0 if g == 'none' else 1 for g in gold]
    pred_binary = [0 if p == 'none' else 1 for p in pred]

    metrics = {}

    # 1. None F1 (Binary) = F1 for class 0 ("none")
    # This is the F1 score when "none" is the positive class
    metrics['none_f1_binary'] = f1_score(gold_binary, pred_binary, pos_label=0, zero_division=0)
    metrics['none_precision'] = precision_score(gold_binary, pred_binary, pos_label=0, zero_division=0)
    metrics['none_recall'] = recall_score(gold_binary, pred_binary, pos_label=0, zero_division=0)

    # 2. Minority F1 (Binary) = F1 for class 1 (any gap)
    # This is the F1 score when "any gap" is the positive class
    metrics['minority_f1_binary'] = f1_score(gold_binary, pred_binary, pos_label=1, zero_division=0)
    metrics['minority_precision'] = precision_score(gold_binary, pred_binary, pos_label=1, zero_division=0)
    metrics['minority_recall'] = recall_score(gold_binary, pred_binary, pos_label=1, zero_division=0)

    # 3. Minority-Only Macro F1 (multi-class among gap types only)
    # Filter to only gap samples and compute macro F1 across 6 gap types
    gap_gold = []
    gap_pred = []
    for g, p in zip(gold, pred):
        if g != 'none':
            gap_gold.append(g)
            gap_pred.append(p)

    if len(gap_gold) > 0:
        gap_labels = [l for l in GGI_LABELS if l != 'none']
        metrics['minority_only_macro_f1'] = f1_score(gap_gold, gap_pred, labels=gap_labels, average='macro', zero_division=0)
    else:
        metrics['minority_only_macro_f1'] = 0.0

    # 4. Gap Detection Rate = % of real gaps correctly identified as gaps
    real_gaps = sum(1 for g in gold if g != 'none')
    detected_gaps = sum(1 for g, p in zip(gold, pred) if g != 'none' and p != 'none')
    metrics['gap_detection_rate'] = (detected_gaps / real_gaps * 100) if real_gaps > 0 else 0.0

    # 5. Overall Accuracy
    metrics['accuracy'] = accuracy_score(gold, pred)

    # 6. Support counts
    metrics['total_samples'] = len(gold)
    metrics['real_gaps'] = real_gaps
    metrics['predicted_gaps'] = sum(1 for p in pred if p != 'none')
    metrics['none_gold'] = sum(1 for g in gold if g == 'none')
    metrics['none_pred'] = sum(1 for p in pred if p == 'none')

    # 7. Confusion matrix for binary
    tn, fp, fn, tp = confusion_matrix(gold_binary, pred_binary).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    return metrics


# ==========================================================
# EVALUATE
# ==========================================================

def evaluate_condition(dirs, condition_name):
    print(f"\n{'='*90}")
    print(f"📊 {condition_name.upper()}")
    print(f"{'='*90}")

    model_data = load_predictions(dirs)
    all_results = []

    for model_name in sorted(model_data.keys()):
        gold = model_data[model_name]['gold']
        pred = model_data[model_name]['pred']

        if len(gold) == 0:
            continue

        metrics = compute_binary_metrics(gold, pred)
        metrics['model'] = model_name
        metrics['condition'] = condition_name
        all_results.append(metrics)

        print(f"\n🔹 {model_name}")
        print(f"   Samples: {metrics['total_samples']:,} | Real gaps: {metrics['real_gaps']:,} | Predicted gaps: {metrics['predicted_gaps']:,}")
        print(f"   Confusion: TN={metrics['true_negatives']}, FP={metrics['false_positives']}, FN={metrics['false_negatives']}, TP={metrics['true_positives']}")
        print(f"   → None F1 (Binary):      {metrics['none_f1_binary']:.4f}  [precision={metrics['none_precision']:.4f}, recall={metrics['none_recall']:.4f}]")
        print(f"   → Minority F1 (Binary):  {metrics['minority_f1_binary']:.4f}  [precision={metrics['minority_precision']:.4f}, recall={metrics['minority_recall']:.4f}]")
        print(f"   → Minority-Only Macro F1: {metrics['minority_only_macro_f1']:.4f}")
        print(f"   → Gap Detection Rate:    {metrics['gap_detection_rate']:.2f}%")
        print(f"   → Overall Accuracy:      {metrics['accuracy']:.4f}")

    return all_results


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    facilitator_results = evaluate_condition(FACILITATOR_DIRS, "With Facilitator")
    baseline_results = evaluate_condition(BASELINE_DIRS, "Without Facilitator (Baseline)")

    # Side-by-side comparison
    print(f"\n{'='*90}")
    print("📊 SIDE-BY-SIDE BINARY COMPARISON")
    print(f"{'='*90}")

    all_models = sorted(set(r['model'] for r in facilitator_results + baseline_results))

    for model in all_models:
        fac = next((r for r in facilitator_results if r['model'] == model), None)
        base = next((r for r in baseline_results if r['model'] == model), None)

        if fac and base:
            print(f"\n{'─'*90}")
            print(f"【{model}】")
            print(f"{'─'*90}")
            print(f"{'Metric':<30} {'With Facilitator':>18} {'Baseline':>18} {'Delta':>18} {'Winner':>12}")
            print(f"{'─'*90}")

            metrics_to_compare = [
                ('None F1 (Binary)', 'none_f1_binary'),
                ('Minority F1 (Binary)', 'minority_f1_binary'),
                ('Minority-Only Macro F1', 'minority_only_macro_f1'),
                ('Gap Detection Rate (%)', 'gap_detection_rate'),
                ('Overall Accuracy', 'accuracy'),
            ]

            for label, key in metrics_to_compare:
                f_val = fac[key]
                b_val = base[key]
                delta = f_val - b_val
                winner = "FACILITATOR" if f_val > b_val else "BASELINE" if b_val > f_val else "TIE"
                marker = " <<<" if b_val > f_val else ""
                print(f"{label:<30} {f_val:>18.4f} {b_val:>18.4f} {delta:>+18.4f} {winner:>12}{marker}")

    # Save
    all_results = facilitator_results + baseline_results
    df = pd.DataFrame(all_results)
    cols = ['condition', 'model', 'none_f1_binary', 'minority_f1_binary', 'minority_only_macro_f1',
            'gap_detection_rate', 'accuracy', 'none_precision', 'none_recall', 
            'minority_precision', 'minority_recall',
            'total_samples', 'real_gaps', 'predicted_gaps', 'none_gold', 'none_pred',
            'true_negatives', 'false_positives', 'false_negatives', 'true_positives']
    df = df[[c for c in cols if c in df.columns]]

    output_file = "/Users/apple/StudioProjects/last_project_emnlp/binary_metrics_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to: {output_file}")
