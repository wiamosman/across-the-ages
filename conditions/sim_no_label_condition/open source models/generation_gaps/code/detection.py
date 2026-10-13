import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, 
    confusion_matrix, precision_recall_fscore_support
)

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

POSITIVE_LABELS = [l for l in GGI_LABELS if l != 'none']
NULL_CLASS = 'none'

OUTPUT_DIR = "/Users/apple/StudioProjects/last_project_emnlp/results_breakdown_open_source"


# ==========================================================
# LOAD PREDICTIONS
# ==========================================================

def load_predictions(result_dirs):
    model_data = defaultdict(lambda: {'gold': [], 'pred': []})

    for result_dir in result_dirs:
        if not os.path.exists(result_dir):
            print(f"⚠️ Directory not found: {result_dir}")
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
# 3-TIER METRICS (AGREED DESIGN)
# ==========================================================

def compute_3tier_metrics(gold, pred):
    """
    Tier 1: Any-F1 — Binary F1 for "any gap" vs "none"
    Tier 2: Detection Rate — Recall over positive classes (% of real gaps caught)
    Tier 3: Macro-F1 (Positive Only) — Unweighted mean F1 across 6 gap types, excl. "none"
    """
    metrics = {}

    # --- TIER 1: Any-F1 (Binary: any gap vs none) ---
    gold_binary = [1 if g != NULL_CLASS else 0 for g in gold]
    pred_binary = [1 if p != NULL_CLASS else 0 for p in pred]

    p, r, f1, _ = precision_recall_fscore_support(
        gold_binary, pred_binary, average='binary', zero_division=0
    )
    metrics['any_f1_precision'] = p
    metrics['any_f1_recall'] = r
    metrics['any_f1'] = f1

    # --- TIER 2: Detection Rate (% of real gaps correctly identified) ---
    real_gaps = sum(1 for g in gold if g != NULL_CLASS)
    detected_gaps = sum(1 for g, p in zip(gold, pred) if g != NULL_CLASS and p != NULL_CLASS)
    metrics['detection_rate'] = (detected_gaps / real_gaps * 100) if real_gaps > 0 else 0.0

    # --- TIER 3: Macro-F1 (Positive classes only, excluding "none") ---
    gap_gold = []
    gap_pred = []
    for g, p in zip(gold, pred):
        if g != NULL_CLASS:
            gap_gold.append(g)
            gap_pred.append(p)

    if len(gap_gold) > 0:
        present_labels = sorted(list(set(gap_gold)))
        f1_scores = []
        
        for label in present_labels:
            yt_bin = [1 if g == label else 0 for g in gap_gold]
            yp_bin = [1 if p == label else 0 for p in gap_pred]
            
            _, _, f1, _ = precision_recall_fscore_support(
                yt_bin, yp_bin, average='binary', zero_division=0
            )
            f1_scores.append(f1)
        
        metrics['macro_f1_positive'] = np.mean(f1_scores) if f1_scores else 0.0
    else:
        metrics['macro_f1_positive'] = 0.0

    # --- Reference metrics ---
    metrics['accuracy'] = accuracy_score(gold, pred)
    metrics['overall_macro_f1'] = f1_score(gold, pred, labels=GGI_LABELS, average='macro', zero_division=0)

    # --- Support counts ---
    metrics['total_samples'] = len(gold)
    metrics['real_gaps'] = real_gaps
    metrics['predicted_gaps'] = sum(1 for p in pred if p != NULL_CLASS)
    metrics['none_gold'] = sum(1 for g in gold if g == NULL_CLASS)
    metrics['none_pred'] = sum(1 for p in pred if p == NULL_CLASS)

    # --- Binary confusion matrix ---
    tn, fp, fn, tp = confusion_matrix(gold_binary, pred_binary).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    return metrics


# ==========================================================
# PLOTTING
# ==========================================================

def plot_detection_rate_comparison(results_dict, save_path):
    """Horizontal bar chart: Detection Rate by model and condition"""
    models = sorted(set(m for cond in results_dict.values() for m in cond.keys()))
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(models) * 0.5)))
    
    y_pos = np.arange(len(models))
    bar_height = 0.35
    
    baseline_rates = [results_dict['baseline'].get(m, {}).get('detection_rate', 0) for m in models]
    facilitated_rates = [results_dict['facilitated'].get(m, {}).get('detection_rate', 0) for m in models]
    
    bars1 = ax.barh(y_pos - bar_height/2, baseline_rates, bar_height, 
                    label='Baseline', color='#E74C3C', alpha=0.8, edgecolor='black')
    bars2 = ax.barh(y_pos + bar_height/2, facilitated_rates, bar_height,
                    label='With Facilitator', color='#2ECC71', alpha=0.8, edgecolor='black')
    
    # Color threshold lines
    ax.axvline(x=40, color='green', linestyle='--', alpha=0.3, label='Good (≥40%)')
    ax.axvline(x=20, color='orange', linestyle='--', alpha=0.3, label='Fair (≥20%)')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace('_', ' ').upper() for m in models])
    ax.set_xlabel('Detection Rate (%)', fontsize=12)
    ax.set_title('Generation Gap Detection Rate by Model', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            width = bar.get_width()
            if width > 0:
                ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                       f'{width:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: {save_path}")


def plot_any_f1_comparison(results_dict, save_path):
    """Grouped bar chart: Any-F1 by model"""
    models = sorted(set(m for cond in results_dict.values() for m in cond.keys()))
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(models) * 0.5)))
    
    y_pos = np.arange(len(models))
    bar_height = 0.35
    
    baseline_f1 = [results_dict['baseline'].get(m, {}).get('any_f1', 0) for m in models]
    facilitated_f1 = [results_dict['facilitated'].get(m, {}).get('any_f1', 0) for m in models]
    
    ax.barh(y_pos - bar_height/2, baseline_f1, bar_height, 
            label='Baseline', color='#E74C3C', alpha=0.8, edgecolor='black')
    ax.barh(y_pos + bar_height/2, facilitated_f1, bar_height,
            label='With Facilitator', color='#2ECC71', alpha=0.8, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace('_', ' ').upper() for m in models])
    ax.set_xlabel('Any-F1 Score', fontsize=12)
    ax.set_title('Any-F1 (Binary Gap Detection) by Model', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: {save_path}")


# ==========================================================
# EVALUATE
# ==========================================================

def evaluate_condition(dirs, condition_name):
    print(f"\n{'='*90}")
    print(f"📊 {condition_name.upper()}")
    print(f"{'='*90}")

    model_data = load_predictions(dirs)
    all_results = {}

    for model_name in sorted(model_data.keys()):
        gold = model_data[model_name]['gold']
        pred = model_data[model_name]['pred']

        if len(gold) == 0:
            continue

        metrics = compute_3tier_metrics(gold, pred)
        metrics['model'] = model_name
        metrics['condition'] = condition_name
        all_results[model_name] = metrics

        print(f"\n🔹 {model_name}")
        print(f"   Samples: {metrics['total_samples']:,} | Real gaps: {metrics['real_gaps']:,} | Predicted gaps: {metrics['predicted_gaps']:,}")
        print(f"   Confusion: TN={metrics['true_negatives']}, FP={metrics['false_positives']}, FN={metrics['false_negatives']}, TP={metrics['true_positives']}")
        print(f"   → Any-F1:                {metrics['any_f1']:.4f}  [P={metrics['any_f1_precision']:.4f}, R={metrics['any_f1_recall']:.4f}]")
        print(f"   → Detection Rate:        {metrics['detection_rate']:.2f}%")
        print(f"   → Macro-F1 (Positive):   {metrics['macro_f1_positive']:.4f}")
        print(f"   → Overall Accuracy:      {metrics['accuracy']:.4f}")

    return all_results


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Evaluate both conditions
    facilitated_results = evaluate_condition(FACILITATOR_DIRS, "With Facilitator")
    baseline_results = evaluate_condition(BASELINE_DIRS, "Without Facilitator (Baseline)")

    # Store for plotting
    results_by_condition = {
        'baseline': baseline_results,
        'facilitated': facilitated_results
    }

    # --- SIDE-BY-SIDE COMPARISON ---
    print(f"\n{'='*90}")
    print("📊 SIDE-BY-SIDE COMPARISON (3-TIER METRICS)")
    print(f"{'='*90}")

    all_models = sorted(set(facilitated_results.keys()) | set(baseline_results.keys()))

    for model in all_models:
        fac = facilitated_results.get(model)
        base = baseline_results.get(model)

        if fac and base:
            print(f"\n{'─'*90}")
            print(f"【{model}】")
            print(f"{'─'*90}")
            print(f"{'Metric':<<30} {'With Facilitator':>18} {'Baseline':>18} {'Delta':>18} {'Winner':>12}")
            print(f"{'─'*90}")

            metrics_to_compare = [
                ('Any-F1 (Binary)', 'any_f1'),
                ('Detection Rate (%)', 'detection_rate'),
                ('Macro-F1 (Positive)', 'macro_f1_positive'),
                ('Overall Accuracy', 'accuracy'),
            ]

            for label, key in metrics_to_compare:
                f_val = fac[key]
                b_val = base[key]
                delta = f_val - b_val
                winner = "FACILITATOR" if f_val > b_val else "BASELINE" if b_val > f_val else "TIE"
                marker = " <<<" if b_val > f_val else ""
                print(f"{label:<30} {f_val:>18.4f} {b_val:>18.4f} {delta:>+18.4f} {winner:>12}{marker}")

    # --- SAVE CSV ---
    all_results_list = list(facilitated_results.values()) + list(baseline_results.values())
    df = pd.DataFrame(all_results_list)
    
    cols = ['condition', 'model', 'any_f1', 'any_f1_precision', 'any_f1_recall',
            'detection_rate', 'macro_f1_positive', 'accuracy', 'overall_macro_f1',
            'total_samples', 'real_gaps', 'predicted_gaps', 'none_gold', 'none_pred',
            'true_negatives', 'false_positives', 'false_negatives', 'true_positives']
    df = df[[c for c in cols if c in df.columns]]

    output_file = os.path.join(OUTPUT_DIR, "open_source_3tier_metrics.csv")
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to: {output_file}")

    # --- LATEX TABLE ---
    latex_rows = []
    for model in all_models:
        fac = facilitated_results.get(model)
        base = baseline_results.get(model)
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
    latex_file = os.path.join(OUTPUT_DIR, "open_source_latex_table.csv")
    latex_df.to_csv(latex_file, index=False)
    
    print(f"\n{'='*90}")
    print("📝 LATEX TABLE FORMAT")
    print(f"{'='*90}")
    print(latex_df.to_string(index=False))
    print(f"\n💾 LaTeX table saved to: {latex_file}")

    # --- PLOTS ---
    plot_detection_rate_comparison(results_by_condition, 
        os.path.join(OUTPUT_DIR, "open_source_detection_rate.png"))
    plot_any_f1_comparison(results_by_condition,
        os.path.join(OUTPUT_DIR, "open_source_any_f1.png"))

    print(f"\n{'='*90}")
    print("✅ ALL EVALUATIONS COMPLETE")
    print(f"{'='*90}")