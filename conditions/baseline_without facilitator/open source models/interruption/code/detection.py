import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, 
    confusion_matrix, precision_recall_fscore_support
)

# ==========================================================
# CONFIGURATION — INTERRUPTION TASKS
# ==========================================================

# --- BASELINE (without facilitator) ---
BASELINE_DIRS = [
    "/Users/apple/StudioProjects/last_project_emnlp/without facilitator/other models/interruption/results/batch_1_results",
    "/Users/apple/StudioProjects/last_project_emnlp/without facilitator/other models/interruption/results/batch_2_results"
]

# --- FACILITATED (with facilitator) ---
FACILITATED_DIRS = [
    "/Users/apple/StudioProjects/last_project_emnlp/with facilitator/interruption/results/batch_1_results",
    "/Users/apple/StudioProjects/last_project_emnlp/with facilitator/interruption/results/batch_2_results"
]

OUTPUT_DIR = "/Users/apple/StudioProjects/last_project_emnlp/results_breakdown_interruption"

NULL_CLASS = 'none'


# ==========================================================
# LOAD PREDICTIONS — FIXED FOR INTERRUPTION KEYS
# ==========================================================

def load_predictions(result_dirs, task_type="type", condition_name=""):
    """
    Load predictions from batch result directories.
    
    task_type: "type" uses gold_type/pred_type, "role" uses gold_role/pred_role
    """
    model_data = defaultdict(lambda: {'gold': [], 'pred': []})

    # Select keys based on task
    if task_type == "type":
        gold_key = 'gold_type'
        pred_key = 'pred_type'
    else:
        gold_key = 'gold_role'
        pred_key = 'pred_role'

    for result_dir in result_dirs:
        print(f"\n   📁 Checking: {result_dir}")
        
        if not os.path.exists(result_dir):
            print(f"   ❌ Directory NOT FOUND: {result_dir}")
            continue
        
        items = os.listdir(result_dir)
        print(f"   📂 Contains: {len(items)} items")

        for folder in items:
            folder_path = os.path.join(result_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            model_name = folder
            files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            print(f"   🔍 {model_name}: {len(files)} JSON files")

            for fn in files:
                file_path = os.path.join(folder_path, fn)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Handle both list format and dict-with-utterances format
                if isinstance(data, list):
                    utterances = data
                elif isinstance(data, dict):
                    utterances = data.get('utterances', [])
                else:
                    utterances = []

                count = 0
                for u in utterances:
                    if not isinstance(u, dict):
                        continue

                    # Use interruption-specific keys
                    gold = u.get(gold_key, '').lower().strip() if u.get(gold_key) else 'none'
                    pred = u.get(pred_key, '').lower().strip() if u.get(pred_key) else 'none'

                    if not gold:
                        gold = 'none'
                    if not pred:
                        pred = 'none'

                    model_data[model_name]['gold'].append(gold)
                    model_data[model_name]['pred'].append(pred)
                    count += 1

            print(f"      ✅ Loaded {count} utterances total")

    total_models = len(model_data)
    total_utterances = sum(len(v['gold']) for v in model_data.values())
    print(f"\n   📊 SUMMARY for {condition_name}: {total_models} models, {total_utterances} total utterances")
    
    return model_data


# ==========================================================
# 3-TIER METRICS (AGREED DESIGN)
# ==========================================================

def compute_3tier_metrics(gold, pred, positive_labels):
    metrics = {}

    # --- TIER 1: Any-F1 ---
    gold_binary = [1 if g in positive_labels else 0 for g in gold]
    pred_binary = [1 if p in positive_labels else 0 for p in pred]

    p, r, f1, _ = precision_recall_fscore_support(
        gold_binary, pred_binary, average='binary', zero_division=0
    )
    metrics['any_f1_precision'] = p
    metrics['any_f1_recall'] = r
    metrics['any_f1'] = f1

    # --- TIER 2: Detection Rate ---
    real_interruptions = sum(1 for g in gold if g in positive_labels)
    detected_interruptions = sum(1 for g, p in zip(gold, pred) if g in positive_labels and p in positive_labels)
    metrics['detection_rate'] = (detected_interruptions / real_interruptions * 100) if real_interruptions > 0 else 0.0

    # --- TIER 3: Macro-F1 (Positive only) ---
    pos_gold = []
    pos_pred = []
    for g, p in zip(gold, pred):
        if g in positive_labels:
            pos_gold.append(g)
            pos_pred.append(p)

    if len(pos_gold) > 0:
        present_labels = sorted(list(set(pos_gold)))
        f1_scores = []
        
        for label in present_labels:
            yt_bin = [1 if g == label else 0 for g in pos_gold]
            yp_bin = [1 if p == label else 0 for p in pos_pred]
            
            _, _, f1, _ = precision_recall_fscore_support(
                yt_bin, yp_bin, average='binary', zero_division=0
            )
            f1_scores.append(f1)
        
        metrics['macro_f1_positive'] = np.mean(f1_scores) if f1_scores else 0.0
    else:
        metrics['macro_f1_positive'] = 0.0

    # Reference metrics
    all_labels = sorted(list(set(gold + pred)))
    metrics['accuracy'] = accuracy_score(gold, pred)
    metrics['overall_macro_f1'] = f1_score(gold, pred, labels=all_labels, average='macro', zero_division=0)

    metrics['total_samples'] = len(gold)
    metrics['real_positive'] = real_interruptions
    metrics['predicted_positive'] = sum(1 for p in pred if p in positive_labels)
    metrics['none_gold'] = sum(1 for g in gold if g == NULL_CLASS)
    metrics['none_pred'] = sum(1 for p in pred if p == NULL_CLASS)

    try:
        tn, fp, fn, tp = confusion_matrix(gold_binary, pred_binary).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
    except ValueError:
        metrics['true_negatives'] = 0
        metrics['false_positives'] = 0
        metrics['false_negatives'] = 0
        metrics['true_positives'] = 0

    return metrics


# ==========================================================
# PLOTTING
# ==========================================================

def plot_metric_comparison(results_dict, metric_key, metric_name, save_path, y_max=100):
    models = sorted(set(m for cond in results_dict.values() for m in cond.keys()))
    if not models:
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(models) * 0.5)))
    
    y_pos = np.arange(len(models))
    bar_height = 0.35
    
    baseline_vals = [results_dict['baseline'].get(m, {}).get(metric_key, 0) for m in models]
    facilitated_vals = [results_dict['facilitated'].get(m, {}).get(metric_key, 0) for m in models]

    if metric_key in ['any_f1', 'macro_f1_positive', 'accuracy', 'overall_macro_f1']:
        baseline_vals = [v * 100 for v in baseline_vals]
        facilitated_vals = [v * 100 for v in facilitated_vals]

    ax.barh(y_pos - bar_height/2, baseline_vals, bar_height, 
            label='Baseline', color='#E74C3C', alpha=0.8, edgecolor='black')
    ax.barh(y_pos + bar_height/2, facilitated_vals, bar_height,
            label='With Facilitator', color='#2ECC71', alpha=0.8, edgecolor='black')

    if metric_key == 'detection_rate':
        ax.axvline(x=40, color='green', linestyle='--', alpha=0.3)
        ax.axvline(x=20, color='orange', linestyle='--', alpha=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace('_', ' ').upper() for m in models])
    ax.set_xlabel(f'{metric_name} (%)', fontsize=12)
    ax.set_title(f'{metric_name} by Model — Interruption Detection', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, y_max)
    ax.invert_yaxis()

    for i, (b, f) in enumerate(zip(baseline_vals, facilitated_vals)):
        ax.text(b + 1, i - bar_height/2, f'{b:.1f}', va='center', fontsize=8)
        ax.text(f + 1, i + bar_height/2, f'{f:.1f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: {save_path}")


# ==========================================================
# EVALUATE
# ==========================================================

def evaluate_condition(dirs, task_type, condition_name):
    task_label = "Interruption Type" if task_type == "type" else "Interruption Role"
    print(f"\n{'='*90}")
    print(f"📊 {task_label.upper()} — {condition_name.upper()}")
    print(f"{'='*90}")

    model_data = load_predictions(dirs, task_type, condition_name)
    all_results = {}

    if not model_data:
        print("⚠️ No model data found!")
        return {}

    for model_name in sorted(model_data.keys()):
        gold = model_data[model_name]['gold']
        pred = model_data[model_name]['pred']

        if len(gold) == 0:
            print(f"   ⚠️ {model_name}: No utterances")
            continue

        all_labels = sorted(list(set(gold + pred)))
        positive_labels = [l for l in all_labels if l != NULL_CLASS]
        
        print(f"\n🔹 {model_name} | Labels: {all_labels} | Positive: {positive_labels}")

        metrics = compute_3tier_metrics(gold, pred, positive_labels)
        metrics['model'] = model_name
        metrics['condition'] = condition_name
        metrics['task'] = task_type
        metrics['labels_detected'] = ', '.join(all_labels)
        metrics['positive_labels'] = ', '.join(positive_labels)
        all_results[model_name] = metrics

        print(f"   Samples: {metrics['total_samples']:,} | Real positive: {metrics['real_positive']:,} | Predicted positive: {metrics['predicted_positive']:,}")
        print(f"   Confusion: TN={metrics['true_negatives']}, FP={metrics['false_positives']}, FN={metrics['false_negatives']}, TP={metrics['true_positives']}")
        print(f"   → Any-F1:                {metrics['any_f1']:.4f}")
        print(f"   → Detection Rate:        {metrics['detection_rate']:.2f}%")
        print(f"   → Macro-F1 (Positive):   {metrics['macro_f1_positive']:.4f}")
        print(f"   → Overall Accuracy:      {metrics['accuracy']:.4f}")

    return all_results


# ==========================================================
# MAIN — RUN BOTH TYPE AND ROLE
# ==========================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results_combined = {}

    # --- INTERRUPTION TYPE ---
    print(f"\n{'#'*90}")
    print(f"# INTERRUPTION TYPE EVALUATION")
    print(f"{'#'*90}")
    
    baseline_type = evaluate_condition(BASELINE_DIRS, "type", "Without Facilitator (Baseline)")
    facilitated_type = evaluate_condition(FACILITATED_DIRS, "type", "With Facilitator")
    
    all_results_combined['interruption_type'] = {
        'baseline': baseline_type,
        'facilitated': facilitated_type
    }

    # --- INTERRUPTION ROLE ---
    print(f"\n{'#'*90}")
    print(f"# INTERRUPTION ROLE EVALUATION")
    print(f"{'#'*90}")
    
    baseline_role = evaluate_condition(BASELINE_DIRS, "role", "Without Facilitator (Baseline)")
    facilitated_role = evaluate_condition(FACILITATED_DIRS, "role", "With Facilitator")
    
    all_results_combined['interruption_role'] = {
        'baseline': baseline_role,
        'facilitated': facilitated_role
    }

    # --- SIDE-BY-SIDE COMPARISON ---
    for task_name, task_results in all_results_combined.items():
        print(f"\n{'='*90}")
        print(f"📊 SIDE-BY-SIDE — {task_name.upper().replace('_', ' ')} (3-TIER METRICS)")
        print(f"{'='*90}")

        baseline = task_results['baseline']
        facilitated = task_results['facilitated']
        all_models = sorted(set(facilitated.keys()) | set(baseline.keys()))

        for model in all_models:
            fac = facilitated.get(model)
            base = baseline.get(model)

            if fac and base:
                print(f"\n{'─'*90}")
                print(f"【{model}】")
                print(f"{'─'*90}")
                print(f"{'Metric':<<30} {'With Facilitator':>18} {'Baseline':>18} {'Delta':>18} {'Winner':>12}")
                print(f"{'─'*90}")

                for label, key in [
                    ('Any-F1 (Binary)', 'any_f1'),
                    ('Detection Rate (%)', 'detection_rate'),
                    ('Macro-F1 (Positive)', 'macro_f1_positive'),
                    ('Overall Accuracy', 'accuracy'),
                ]:
                    f_val = fac[key]
                    b_val = base[key]
                    delta = f_val - b_val
                    winner = "FACILITATOR" if f_val > b_val else "BASELINE" if b_val > f_val else "TIE"
                    marker = " <<<" if b_val > f_val else ""
                    print(f"{label:<30} {f_val:>18.4f} {b_val:>18.4f} {delta:>+18.4f} {winner:>12}{marker}")

    # --- SAVE CSV ---
    all_results_list = []
    for task_name, task_results in all_results_combined.items():
        for condition, results in task_results.items():
            for model, metrics in results.items():
                row = metrics.copy()
                row['task_name'] = task_name
                all_results_list.append(row)

    if all_results_list:
        df = pd.DataFrame(all_results_list)
        
        cols = ['task_name', 'condition', 'model', 'labels_detected', 'positive_labels',
                'any_f1', 'any_f1_precision', 'any_f1_recall',
                'detection_rate', 'macro_f1_positive', 'accuracy', 'overall_macro_f1',
                'total_samples', 'real_positive', 'predicted_positive', 'none_gold', 'none_pred',
                'true_negatives', 'false_positives', 'false_negatives', 'true_positives']
        df = df[[c for c in cols if c in df.columns]]

        output_file = os.path.join(OUTPUT_DIR, "interruption_3tier_metrics.csv")
        df.to_csv(output_file, index=False)
        print(f"\n✅ Saved to: {output_file}")

        # --- LATEX TABLES ---
        for task_name, task_results in all_results_combined.items():
            latex_rows = []
            baseline = task_results['baseline']
            facilitated = task_results['facilitated']
            all_models = sorted(set(facilitated.keys()) | set(baseline.keys()))

            for model in all_models:
                fac = facilitated.get(model)
                base = baseline.get(model)
                if fac and base:
                    latex_rows.append({
                        'Model': model.replace('_', ' ').title(),
                        'Any-F1 (Base)': f"{base['any_f1']:.3f}",
                        'Any-F1 (Fac)': f"{fac['any_f1']:.3f}",
                        'Det. Rate (Base)': f"{base['detection_rate']:.1f}\\%",
                        'Det. Rate (Fac)': f"{fac['detection_rate']:.1f}\\%",
                        'Macro-F1 (Base)': f"{base['macro_f1_positive']:.3f}",
                        'Macro-F1 (Fac)': f"{fac['macro_f1_positive']:.3f}"
                    })

            latex_df = pd.DataFrame(latex_rows)
            latex_file = os.path.join(OUTPUT_DIR, f"{task_name}_latex_table.csv")
            latex_df.to_csv(latex_file, index=False)
            
            print(f"\n{'='*90}")
            print(f"📝 LATEX TABLE — {task_name.upper().replace('_', ' ')}")
            print(f"{'='*90}")
            print(latex_df.to_string(index=False))

        # --- PLOTS ---
        for task_name, task_results in all_results_combined.items():
            plot_metric_comparison(task_results, 'detection_rate', 'Detection Rate',
                os.path.join(OUTPUT_DIR, f"{task_name}_detection_rate.png"))
            plot_metric_comparison(task_results, 'any_f1', 'Any-F1',
                os.path.join(OUTPUT_DIR, f"{task_name}_any_f1.png"))
            plot_metric_comparison(task_results, 'macro_f1_positive', 'Macro-F1 (Positive)',
                os.path.join(OUTPUT_DIR, f"{task_name}_macro_f1_positive.png"))

    print(f"\n{'='*90}")
    print("✅ ALL INTERRUPTION EVALUATIONS COMPLETE")
    print(f"{'='*90}")