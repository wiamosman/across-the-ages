import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, f1_score, recall_score
)

# ==========================================================
# CONFIGURATION — GEMINI PATHS
# ==========================================================

# --- BASELINE (without facilitator) ---
BASELINE_ORIGINAL_DIR = "/Users/apple/StudioProjects/last_project_emnlp/labeled_data_original"
BASELINE_ORIGINAL_PREFIX = "transcript_"
BASELINE_PRED_DIR = "/Users/apple/StudioProjects/last_project_emnlp/without facilitator/gemini/gemini_labels"
BASELINE_PRED_PREFIX = "gemini_"
BASELINE_OUTPUT_DIR = "/Users/apple/StudioProjects/last_project_emnlp/without facilitator/gemini/results_breakdown"

# --- FACILITATED (with facilitator) ---
FACILITATED_ORIGINAL_DIR = "/Users/apple/StudioProjects/last_project_emnlp/facilitator/batch_2_facilitator_gpt_labeled"
FACILITATED_ORIGINAL_PREFIX = "gpt_"
FACILITATED_PRED_DIR = "/Users/apple/StudioProjects/last_project_emnlp/with facilitator/open source model/gemini/gemini_predictions"
FACILITATED_PRED_PREFIX = "gemini_"
FACILITATED_OUTPUT_DIR = "/Users/apple/StudioProjects/last_project_emnlp/with facilitator/open source model/gemini/results_breakdown"

NUM_FILES = 104

# ==========================================================
# TASK DEFINITIONS
# ==========================================================

TASKS = {
    "generation_gap": {
        "key": "generation_gap_indicator",
        "labels": [
            'slang mismatch', 'perceived disrespect', 'stereotyped expectation', 
            'ignored input', 'method divergence', 'digital disparity', 'none'
        ],
        "positive_labels": [
            'slang mismatch', 'perceived disrespect', 'stereotyped expectation', 
            'ignored input', 'method divergence', 'digital disparity'
        ],
        "null_class": "none",
        "cmap": "BuGn",
        "normalize_fn": lambda t: str(t).lower().strip().replace("_", " ").replace("-", " ") if t else "none"
    },
    "interruption_type": {
        "key": "interruption_type",
        "labels": None,
        "positive_labels": None,
        "null_class": "none",
        "cmap": "Oranges",
        "normalize_fn": lambda t: str(t).lower().strip() if t else "none"
    },
    "interruption_role": {
        "key": "interruption_role",
        "labels": None,
        "positive_labels": None,
        "null_class": "none",
        "cmap": "Greens",
        "normalize_fn": lambda t: str(t).lower().strip() if t else "none"
    }
}


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def safe_json_load(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    return raw.get('utterances', [])


def extract_data(task_key, normalize_fn, original_dir, original_prefix, pred_dir, pred_prefix):
    y_true, y_pred = [], []
    found_count = 0

    for i in range(1, NUM_FILES + 1):
        orig_path = os.path.join(original_dir, f"{original_prefix}{i}.json")
        pred_path = os.path.join(pred_dir, f"{pred_prefix}{i}.json")

        if os.path.exists(orig_path) and os.path.exists(pred_path):
            found_count += 1
            try:
                data_o = safe_json_load(orig_path)
                data_p = safe_json_load(pred_path)
                loop_range = min(len(data_o), len(data_p))

                for idx in range(loop_range):
                    uo, up = data_o[idx], data_p[idx]
                    if isinstance(uo, dict) and isinstance(up, dict):
                        gold = normalize_fn(uo.get(task_key))
                        pred = normalize_fn(up.get(task_key))
                        y_true.append(gold)
                        y_pred.append(pred)
            except Exception as e:
                print(f"❌ Error in File {i}: {type(e).__name__}: {e}")

    print(f"   Found {found_count} matching file pairs | Extracted {len(y_true)} utterances")
    return y_true, y_pred


def calculate_any_f1(y_true, y_pred, positive_labels):
    y_true_binary = [1 if t in positive_labels else 0 for t in y_true]
    y_pred_binary = [1 if p in positive_labels else 0 for p in y_pred]
    
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='binary', zero_division=0
    )
    return {'precision': p, 'recall': r, 'f1': f1}


def calculate_detection_rate(y_true, y_pred, positive_labels):
    y_true_binary = [1 if t in positive_labels else 0 for t in y_true]
    y_pred_binary = [1 if p in positive_labels else 0 for p in y_pred]
    return recall_score(y_true_binary, y_pred_binary, zero_division=0)


def calculate_macro_f1_positive(y_true, y_pred, positive_labels):
    filtered_true = []
    filtered_pred = []
    
    for t, p in zip(y_true, y_pred):
        if t in positive_labels:
            filtered_true.append(t)
            filtered_pred.append(p)
    
    if not filtered_true:
        return 0.0
    
    present_labels = sorted(list(set(filtered_true)))
    f1_scores = []
    
    for label in present_labels:
        yt_bin = [1 if t == label else 0 for t in filtered_true]
        yp_bin = [1 if p == label else 0 for p in filtered_pred]
        
        _, _, f1, _ = precision_recall_fscore_support(
            yt_bin, yp_bin, average='binary', zero_division=0
        )
        f1_scores.append(f1)
    
    return np.mean(f1_scores) if f1_scores else 0.0


def calculate_per_class_f1(y_true, y_pred, labels):
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame({
        'Label': labels,
        'Precision': p,
        'Recall': r,
        'F1': f1,
        'Support': support
    })


def plot_confusion_matrix(y_true, y_pred, labels, title, cmap, save_path):
    plt.figure(figsize=(max(10, len(labels)), max(8, len(labels) * 0.6)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels, cbar=False)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                plt.text(j + 0.5, i + 0.75, f'({cm_norm[i,j]:.1%})', 
                        ha='center', va='center', fontsize=8, 
                        color='white' if cm_norm[i,j] > 0.5 else 'black')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Actual (Ground Truth)', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved confusion matrix: {save_path}")


def plot_detection_rate_comparison(results_dict, save_path):
    tasks = list(results_dict.keys())
    if not tasks:
        return
    
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 6))
    if len(tasks) == 1:
        axes = [axes]
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        task_name = task.replace('_', ' ').title()
        
        model_names = []
        detection_rates = []
        
        for model, data in results_dict[task].items():
            model_names.append(model.replace('_', ' ').upper())
            detection_rates.append(data['detection_rate'] * 100)
        
        bar_colors = []
        for rate in detection_rates:
            if rate >= 40:
                bar_colors.append('#2ECC71')
            elif rate >= 20:
                bar_colors.append('#F39C12')
            else:
                bar_colors.append('#E74C3C')
        
        bars = ax.barh(model_names, detection_rates, color=bar_colors, 
                      edgecolor='black', linewidth=0.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Detection Rate (%)', fontsize=12)
        ax.set_title(f'{task_name}\nDetection Rate', fontsize=13, fontweight='bold')
        
        for bar, rate in zip(bars, detection_rates):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                   f'{rate:.1f}%', va='center', fontsize=10, fontweight='bold')
        
        ax.axvline(x=40, color='green', linestyle='--', alpha=0.5)
        ax.axvline(x=20, color='orange', linestyle='--', alpha=0.5)
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved detection rate comparison: {save_path}")


# ==========================================================
# MAIN EVALUATION FUNCTION
# ==========================================================

def evaluate_breakdown_task(task_name, config, original_dir, original_prefix, 
                            pred_dir, pred_prefix, output_dir, condition_name):
    print(f"\n{'='*70}")
    print(f"🔍 {condition_name.upper()}: {task_name.upper().replace('_', ' ')}")
    print(f"{'='*70}")

    task_dir = ensure_dir(os.path.join(output_dir, task_name))
    key = config["key"]
    norm_fn = config["normalize_fn"]
    null_class = config["null_class"]
    cmap = config["cmap"]

    y_true, y_pred = extract_data(key, norm_fn, original_dir, original_prefix, 
                                  pred_dir, pred_prefix)
    if not y_true:
        print(f"⚠️ No data found. Skipping.")
        return None

    if config.get("labels"):
        labels = [norm_fn(l) for l in config["labels"]]
        positive_labels = [norm_fn(l) for l in config["positive_labels"]]
    else:
        labels = sorted(list(set(y_true + y_pred)))
        positive_labels = [l for l in labels if l != null_class]
    
    print(f"   Labels detected: {labels}")
    print(f"   Positive labels: {positive_labels}")

    any_f1_results = calculate_any_f1(y_true, y_pred, positive_labels)
    detection_rate = calculate_detection_rate(y_true, y_pred, positive_labels)
    macro_f1_positive = calculate_macro_f1_positive(y_true, y_pred, positive_labels)

    overall_acc = accuracy_score(y_true, y_pred)
    overall_macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    per_class_df = calculate_per_class_f1(y_true, y_pred, labels)
    per_class_path = os.path.join(task_dir, f"{task_name}_per_class_{condition_name}.csv")
    per_class_df.to_csv(per_class_path, index=False)

    summary = {
        'Condition': condition_name,
        'Task': task_name,
        'Any-F1 (Precision)': round(any_f1_results['precision'], 4),
        'Any-F1 (Recall)': round(any_f1_results['recall'], 4),
        'Any-F1 (F1)': round(any_f1_results['f1'], 4),
        'Detection Rate': round(detection_rate, 4),
        'Macro-F1 (Positive Only)': round(macro_f1_positive, 4),
        'Overall Accuracy': round(overall_acc, 4),
        'Overall Macro-F1 (All)': round(overall_macro_f1, 4),
        'Total Utterances': len(y_true),
        'Positive Utterances': sum(1 for t in y_true if t in positive_labels),
        'None Utterances': sum(1 for t in y_true if t == null_class)
    }

    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(task_dir, f"{task_name}_summary_{condition_name}.csv")
    summary_df.to_csv(summary_path, index=False)

    plot_confusion_matrix(
        y_true, y_pred, labels,
        title=f"{task_name.replace('_', ' ').title()} — {condition_name.title()}\nFull Classification",
        cmap=cmap,
        save_path=os.path.join(task_dir, f"{task_name}_matrix_full_{condition_name}.png")
    )

    pos_labels = [l for l in labels if l != null_class]
    if pos_labels:
        y_true_pos = [t for t in y_true if t in pos_labels]
        y_pred_pos = [p for t, p in zip(y_true, y_pred) if t in pos_labels]
        if y_true_pos:
            plot_confusion_matrix(
                y_true_pos, y_pred_pos, pos_labels,
                title=f"{task_name.replace('_', ' ').title()} — {condition_name.title()}\nPositive Classes Only",
                cmap="Blues",
                save_path=os.path.join(task_dir, f"{task_name}_matrix_positive_{condition_name}.png")
            )

    print(f"\n📊 RESULTS FOR {task_name.upper().replace('_', ' ')} ({condition_name.upper()})")
    print(f"   Any-F1 (Binary):     {summary['Any-F1 (F1)']:.4f}")
    print(f"   Detection Rate:      {summary['Detection Rate']:.4f} ({summary['Detection Rate']*100:.1f}%)")
    print(f"   Macro-F1 (Positive): {summary['Macro-F1 (Positive Only)']:.4f}")
    print(f"   Overall Accuracy:    {summary['Overall Accuracy']:.4f}")
    print(f"   Overall Macro-F1:    {summary['Overall Macro-F1 (All)']:.4f}")
    print(f"   Total/Positive/None: {summary['Total Utterances']}/{summary['Positive Utterances']}/{summary['None Utterances']}")
    print(f"   💾 Saved to: {task_dir}")

    return summary


# ==========================================================
# RUN EVALUATION FOR BOTH CONDITIONS
# ==========================================================

def main():
    all_results = {'baseline': {}, 'facilitated': {}}

    # --- BASELINE ---
    print(f"\n{'#'*70}")
    print(f"# GEMINI BASELINE EVALUATION (Without Facilitator)")
    print(f"{'#'*70}")
    
    ensure_dir(BASELINE_OUTPUT_DIR)
    
    for task_name, config in TASKS.items():
        result = evaluate_breakdown_task(
            task_name, config,
            BASELINE_ORIGINAL_DIR, BASELINE_ORIGINAL_PREFIX,
            BASELINE_PRED_DIR, BASELINE_PRED_PREFIX,
            BASELINE_OUTPUT_DIR, "baseline"
        )
        if result:
            all_results['baseline'][task_name] = result

    # --- FACILITATED ---
    print(f"\n{'#'*70}")
    print(f"# GEMINI FACILITATED EVALUATION (With Facilitator)")
    print(f"{'#'*70}")
    
    ensure_dir(FACILITATED_OUTPUT_DIR)
    
    for task_name, config in TASKS.items():
        result = evaluate_breakdown_task(
            task_name, config,
            FACILITATED_ORIGINAL_DIR, FACILITATED_ORIGINAL_PREFIX,
            FACILITATED_PRED_DIR, FACILITATED_PRED_PREFIX,
            FACILITATED_OUTPUT_DIR, "facilitated"
        )
        if result:
            all_results['facilitated'][task_name] = result

    # --- COMBINED SUMMARY TABLE ---
    print(f"\n{'='*70}")
    print(f"📋 GEMINI COMBINED SUMMARY TABLE")
    print(f"{'='*70}")

    rows = []
    for task_name in TASKS.keys():
        for condition in ['baseline', 'facilitated']:
            if task_name in all_results[condition]:
                r = all_results[condition][task_name]
                rows.append({
                    'Task': task_name.replace('_', ' ').title(),
                    'Condition': condition.title(),
                    'Any-F1': r['Any-F1 (F1)'],
                    'Detection Rate': f"{r['Detection Rate']*100:.1f}%",
                    'Macro-F1 (Positive)': r['Macro-F1 (Positive Only)'],
                    'Overall Acc': r['Overall Accuracy']
                })

    combined_df = pd.DataFrame(rows)
    combined_path = os.path.join(BASELINE_OUTPUT_DIR, "gemini_combined_summary.csv")
    combined_df.to_csv(combined_path, index=False)
    print(combined_df.to_string(index=False))
    print(f"\n💾 Combined summary saved to: {combined_path}")

    # --- LATEX TABLE ---
    latex_rows = []
    for task_name in TASKS.keys():
        base = all_results['baseline'].get(task_name, {})
        fac = all_results['facilitated'].get(task_name, {})
        
        if base and fac:
            latex_rows.append({
                'Task': task_name.replace('_', ' ').title(),
                'Any-F1 (Base)': f"{base['Any-F1 (F1)']:.3f}",
                'Any-F1 (Fac)': f"{fac['Any-F1 (F1)']:.3f}",
                'Det. Rate (Base)': f"{base['Detection Rate']*100:.1f}\\%",
                'Det. Rate (Fac)': f"{fac['Detection Rate']*100:.1f}\\%",
                'Macro-F1 (Base)': f"{base['Macro-F1 (Positive Only)']:.3f}",
                'Macro-F1 (Fac)': f"{fac['Macro-F1 (Positive Only)']:.3f}"
            })

    latex_df = pd.DataFrame(latex_rows)
    latex_path = os.path.join(BASELINE_OUTPUT_DIR, "gemini_latex_table.csv")
    latex_df.to_csv(latex_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"📝 GEMINI LATEX TABLE FORMAT")
    print(f"{'='*70}")
    print(latex_df.to_string(index=False))
    print(f"\n💾 LaTeX table saved to: {latex_path}")

    # --- DETECTION RATE COMPARISON PLOT ---
    plot_data = {}
    for task in TASKS.keys():
        plot_data[task] = {}
        for condition in ['baseline', 'facilitated']:
            label = f"Gemini {condition.title()}"
            if task in all_results[condition]:
                plot_data[task][label] = {
                    'detection_rate': all_results[condition][task]['Detection Rate']
                }

    plot_path = os.path.join(BASELINE_OUTPUT_DIR, "gemini_detection_rate_comparison.png")
    plot_detection_rate_comparison(plot_data, plot_path)

    print(f"\n{'='*70}")
    print(f"✅ ALL GEMINI EVALUATIONS COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()