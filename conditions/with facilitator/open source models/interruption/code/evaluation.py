import os
import json
import glob
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# ==========================================================
# 1. LABEL DEFINITIONS
# ==========================================================
ROLE_LABELS = ['interrupted speaker', 'intervening speaker', 'none']
TYPE_LABELS = ['disruptive intervention', 'cooperative assistance', 'cooperative agreement', 'none']

# ==========================================================
# 2. CLEAN MODEL NAME
# ==========================================================
def clean_model_name(folder_name):
    name = folder_name
    name = name.replace("_", " ").replace(".", " ").replace("-", " ")
    while "  " in name:
        name = name.replace("  ", " ")
    name = name.replace("Qwen2 5 1 5B", "Qwen2.5-1.5B")
    name = name.replace("Qwen 2 5 1 5B", "Qwen2.5-1.5B")
    name = name.replace("Qwen2 5 7B", "Qwen2.5-7B")
    name = name.replace("Qwen 2 5 7B", "Qwen2.5-7B")
    name = name.replace("Llama3 1 8B", "Llama-3.1-8B")
    name = name.replace("Llama 3 1 8B", "Llama-3.1-8B")
    name = name.replace("Llama3 2 3B", "Llama-3.2-3B")
    name = name.replace("Llama 3 2 3B", "Llama-3.2-3B")
    name = name.replace("Mistral 7B", "Mistral-7B")
    name = name.replace("Mistral7B", "Mistral-7B")
    name = name.replace("InternLM2 5 7B", "InternLM-2.5-7B")
    name = name.replace("InternLM 2 5 7B", "InternLM-2.5-7B")
    prefixes = ["meta llama ", "mistralai ", "Qwen ", "internlm ", "meta ", "qwen "]
    for prefix in prefixes:
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
    while "  " in name:
        name = name.replace("  ", " ")
    return name.strip()

# ==========================================================
# 3. LOAD RESULTS
# ==========================================================
def load_model_from_batches(batch_dirs):
    all_results = []
    for batch_dir in batch_dirs:
        if not os.path.exists(batch_dir):
            continue
        result_files = sorted(glob.glob(os.path.join(batch_dir, "results_file_*.json")))
        for rf in result_files:
            with open(rf, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_results.extend(data)
    return all_results

def discover_models(batch1_path, batch2_path):
    models = {}
    for batch_path in [batch1_path, batch2_path]:
        if not os.path.exists(batch_path):
            continue
        for folder in os.listdir(batch_path):
            folder_path = os.path.join(batch_path, folder)
            if os.path.isdir(folder_path):
                clean_name = clean_model_name(folder)
                if clean_name not in models:
                    models[clean_name] = []
                if folder_path not in models[clean_name]:
                    models[clean_name].append(folder_path)
    return models

# ==========================================================
# 4. METRICS CALCULATION
# ==========================================================
def calculate_full_metrics(y_true, y_pred, labels, task_name=""):
    valid_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) 
                     if t in labels and p in labels]
    if not valid_indices:
        return None

    y_true_f = [y_true[i] for i in valid_indices]
    y_pred_f = [y_pred[i] for i in valid_indices]

    accuracy = accuracy_score(y_true_f, y_pred_f)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_f, y_pred_f, labels=labels, average='macro', zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true_f, y_pred_f, labels=labels, average='weighted', zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        y_true_f, y_pred_f, labels=labels, average=None, zero_division=0
    )

    # Non-none metrics
    non_none_labels = [l for l in labels if l != 'none']
    non_none_indices = [i for i, t in enumerate(y_true_f) if t in non_none_labels]
    nn_metrics = None
    if non_none_indices:
        y_true_nn = [y_true_f[i] for i in non_none_indices]
        y_pred_nn = [y_pred_f[i] for i in non_none_indices]
        nn_p, nn_r, nn_f1, _ = precision_recall_fscore_support(
            y_true_nn, y_pred_nn, labels=non_none_labels, average='macro', zero_division=0
        )
        nn_metrics = {'precision': nn_p, 'recall': nn_r, 'f1': nn_f1, 'samples': len(non_none_indices)}

    return {
        'task': task_name,
        'accuracy': accuracy,
        'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1,
        'weighted_p': weighted_p, 'weighted_r': weighted_r, 'weighted_f1': weighted_f1,
        'per_class': {label: {'precision': float(per_class_p[i]), 'recall': float(per_class_r[i]), 
                              'f1': float(per_class_f1[i]), 'support': int(support[i])}
                     for i, label in enumerate(labels)},
        'non_none': nn_metrics,
        'total': len(y_true_f)
    }

def calculate_combined(role_true, role_pred, type_true, type_pred):
    correct = sum(1 for rt, rp, tt, tp in zip(role_true, role_pred, type_true, type_pred)
                  if rt == rp and tt == tp)
    total = len(role_true)
    return {'acc': correct / total if total > 0 else 0, 'correct': correct, 'total': total}

# ==========================================================
# 5. PRINT DETAILED PER-CLASS TABLES
# ==========================================================
def print_per_class_table(model, condition, metrics, labels, task_name):
    print(f"\n{'─'*95}")
    print(f"📋 {condition.upper()} — {model} — {task_name}")
    print(f"{'─'*95}")
    print(f"{'Label':<28} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print(f"{'─'*95}")

    for label in labels:
        stats = metrics['per_class'][label]
        marker = "🔥" if label != 'none' else "  "
        print(f"{marker} {label:<25} {stats['precision']:>12.4f} {stats['recall']:>12.4f} {stats['f1']:>12.4f} {stats['support']:>10}")

    print(f"{'─'*95}")
    print(f"  {'OVERALL':<25} {'—':>12} {'—':>12} {'—':>12} {metrics['total']:>10}")
    print(f"  {'Accuracy':<25} {metrics['accuracy']:>12.4f} {'—':>12} {'—':>12} {'—':>10}")
    print(f"  {'Macro Avg':<25} {metrics['macro_p']:>12.4f} {metrics['macro_r']:>12.4f} {metrics['macro_f1']:>12.4f} {'—':>10}")
    print(f"  {'Weighted Avg':<25} {metrics['weighted_p']:>12.4f} {metrics['weighted_r']:>12.4f} {metrics['weighted_f1']:>12.4f} {'—':>10}")

    if metrics.get('non_none'):
        nn = metrics['non_none']
        print(f"\n  🎯 NON-NONE ONLY (excl. 'none'):")
        print(f"     Macro Precision: {nn['precision']:.4f} | Macro Recall: {nn['recall']:.4f} | Macro F1: {nn['f1']:.4f} | Samples: {nn['samples']}")

    print(f"{'─'*95}")

def print_full_model_report(report):
    model = report['model']
    cond = report['condition']
    n = report['total_samples']

    print(f"\n{'='*95}")
    print(f"📊 {cond.upper()}: {model} | Total Samples: {n}")
    print(f"{'='*95}")

    if report['role_metrics']:
        print_per_class_table(model, cond, report['role_metrics'], ROLE_LABELS, "INTERRUPTION ROLE")

    if report['type_metrics']:
        print_per_class_table(model, cond, report['type_metrics'], TYPE_LABELS, "INTERRUPTION TYPE")

    cm = report['combined_metrics']
    print(f"\n🟢 COMBINED (Role AND Type both correct): {cm['acc']:.4f} ({cm['correct']}/{cm['total']})")
    print(f"{'='*95}\n")

# ==========================================================
# 6. COMPARISON TABLES
# ==========================================================
def print_detailed_comparison(fac_reports, base_reports):
    fac_lookup = {r['model']: r for r in fac_reports}
    base_lookup = {r['model']: r for r in base_reports}
    all_models = sorted(set(fac_lookup.keys()) & set(base_lookup.keys()))

    # ROLE per-class F1 comparison
    print(f"\n{'='*110}")
    print(f"🔴 INTERRUPTION ROLE — Per-Class F1 Comparison (Facilitator vs Baseline)")
    print(f"{'='*110}")
    print(f"{'Model':<20} {'Condition':<12} {'interrupted':>14} {'intervening':>14} {'none':>14} {'Macro F1':>12} {'Accuracy':>10}")
    print(f"{'─'*110}")
    for model in all_models:
        for cond, lookup in [('Facilitator', fac_lookup), ('Baseline', base_lookup)]:
            r = lookup[model]['role_metrics']
            pc = r['per_class']
            print(f"{model:<20} {cond:<12} {pc['interrupted speaker']['f1']:>14.4f} {pc['intervening speaker']['f1']:>14.4f} {pc['none']['f1']:>14.4f} {r['macro_f1']:>12.4f} {r['accuracy']:>10.4f}")
    print(f"{'='*110}")

    # TYPE per-class F1 comparison
    print(f"\n{'='*130}")
    print(f"🔵 INTERRUPTION TYPE — Per-Class F1 Comparison (Facilitator vs Baseline)")
    print(f"{'='*130}")
    print(f"{'Model':<20} {'Condition':<12} {'disruptive':>14} {'assistance':>14} {'agreement':>14} {'none':>14} {'Macro F1':>12} {'Accuracy':>10}")
    print(f"{'─'*130}")
    for model in all_models:
        for cond, lookup in [('Facilitator', fac_lookup), ('Baseline', base_lookup)]:
            r = lookup[model]['type_metrics']
            pc = r['per_class']
            print(f"{model:<20} {cond:<12} {pc['disruptive intervention']['f1']:>14.4f} {pc['cooperative assistance']['f1']:>14.4f} {pc['cooperative agreement']['f1']:>14.4f} {pc['none']['f1']:>14.4f} {r['macro_f1']:>12.4f} {r['accuracy']:>10.4f}")
    print(f"{'='*130}")

    # ROLE per-class PRECISION comparison
    print(f"\n{'='*110}")
    print(f"🔴 INTERRUPTION ROLE — Per-Class PRECISION Comparison")
    print(f"{'='*110}")
    print(f"{'Model':<20} {'Condition':<12} {'interrupted':>14} {'intervening':>14} {'none':>14} {'Macro P':>12}")
    print(f"{'─'*110}")
    for model in all_models:
        for cond, lookup in [('Facilitator', fac_lookup), ('Baseline', base_lookup)]:
            r = lookup[model]['role_metrics']
            pc = r['per_class']
            print(f"{model:<20} {cond:<12} {pc['interrupted speaker']['precision']:>14.4f} {pc['intervening speaker']['precision']:>14.4f} {pc['none']['precision']:>14.4f} {r['macro_p']:>12.4f}")
    print(f"{'='*110}")

    # ROLE per-class RECALL comparison
    print(f"\n{'='*110}")
    print(f"🔴 INTERRUPTION ROLE — Per-Class RECALL Comparison")
    print(f"{'='*110}")
    print(f"{'Model':<20} {'Condition':<12} {'interrupted':>14} {'intervening':>14} {'none':>14} {'Macro R':>12}")
    print(f"{'─'*110}")
    for model in all_models:
        for cond, lookup in [('Facilitator', fac_lookup), ('Baseline', base_lookup)]:
            r = lookup[model]['role_metrics']
            pc = r['per_class']
            print(f"{model:<20} {cond:<12} {pc['interrupted speaker']['recall']:>14.4f} {pc['intervening speaker']['recall']:>14.4f} {pc['none']['recall']:>14.4f} {r['macro_r']:>12.4f}")
    print(f"{'='*110}")

    # TYPE per-class PRECISION comparison
    print(f"\n{'='*130}")
    print(f"🔵 INTERRUPTION TYPE — Per-Class PRECISION Comparison")
    print(f"{'='*130}")
    print(f"{'Model':<20} {'Condition':<12} {'disruptive':>14} {'assistance':>14} {'agreement':>14} {'none':>14} {'Macro P':>12}")
    print(f"{'─'*130}")
    for model in all_models:
        for cond, lookup in [('Facilitator', fac_lookup), ('Baseline', base_lookup)]:
            r = lookup[model]['type_metrics']
            pc = r['per_class']
            print(f"{model:<20} {cond:<12} {pc['disruptive intervention']['precision']:>14.4f} {pc['cooperative assistance']['precision']:>14.4f} {pc['cooperative agreement']['precision']:>14.4f} {pc['none']['precision']:>14.4f} {r['macro_p']:>12.4f}")
    print(f"{'='*130}")

    # TYPE per-class RECALL comparison
    print(f"\n{'='*130}")
    print(f"🔵 INTERRUPTION TYPE — Per-Class RECALL Comparison")
    print(f"{'='*130}")
    print(f"{'Model':<20} {'Condition':<12} {'disruptive':>14} {'assistance':>14} {'agreement':>14} {'none':>14} {'Macro R':>12}")
    print(f"{'─'*130}")
    for model in all_models:
        for cond, lookup in [('Facilitator', fac_lookup), ('Baseline', base_lookup)]:
            r = lookup[model]['type_metrics']
            pc = r['per_class']
            print(f"{model:<20} {cond:<12} {pc['disruptive intervention']['recall']:>14.4f} {pc['cooperative assistance']['recall']:>14.4f} {pc['cooperative agreement']['recall']:>14.4f} {pc['none']['recall']:>14.4f} {r['macro_r']:>12.4f}")
    print(f"{'='*130}")

# ==========================================================
# 7. MAIN
# ==========================================================
def evaluate_condition(batch1, batch2, condition, out_dir):
    print(f"\n{'#'*60}\n# {condition.upper()}\n{'#'*60}")
    models = discover_models(batch1, batch2)
    if not models:
        print("❌ No models found!")
        return []
    print(f"✅ Found {len(models)} model(s): {', '.join(sorted(models.keys()))}")
    for name, paths in sorted(models.items()):
        print(f"   • {name}: {len(paths)} batch folder(s)")

    reports = []
    for model_name, batch_paths in sorted(models.items()):
        print(f"\n📂 Loading {model_name} from {len(batch_paths)} batch folder(s)...")
        for bp in batch_paths:
            print(f"      └─ {bp}")
        results = load_model_from_batches(batch_paths)
        print(f"   ✅ {len(results)} total samples")
        if not results:
            continue

        role_true = [r['gold_role'] for r in results]
        role_pred = [r['pred_role'] for r in results]
        type_true = [r['gold_type'] for r in results]
        type_pred = [r['pred_type'] for r in results]

        role_metrics = calculate_full_metrics(role_true, role_pred, ROLE_LABELS, "Role")
        type_metrics = calculate_full_metrics(type_true, type_pred, TYPE_LABELS, "Type")
        combined = calculate_combined(role_true, role_pred, type_true, type_pred)

        report = {
            'model': model_name, 'condition': condition,
            'total_samples': len(results),
            'role_metrics': role_metrics,
            'type_metrics': type_metrics,
            'combined_metrics': combined
        }
        reports.append(report)
        print_full_model_report(report)

        os.makedirs(out_dir, exist_ok=True)
        safe = model_name.replace(" ", "_").replace("-", "_")
        with open(os.path.join(out_dir, f"{safe}_{condition}_full.json"), 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return reports

def run_all(fac1, fac2, base1, base2, out_dir):
    fac = evaluate_condition(fac1, fac2, "facilitator", out_dir)
    base = evaluate_condition(base1, base2, "baseline", out_dir)
    if fac and base:
        print_detailed_comparison(fac, base)

    summary = {'facilitator': fac, 'baseline': base}
    with open(os.path.join(out_dir, "full_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ All saved to: {out_dir}/")

# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    FAC1 = "/Users/apple/StudioProjects/last_project_emnlp/with facilitator/interruption/results/batch_1_results"
    FAC2 = "/Users/apple/StudioProjects/last_project_emnlp/with facilitator/interruption/results/batch_2_results"
    BASE1 = "/Users/apple/StudioProjects/last_project_emnlp/without facilitator/other models/interruption/results/batch_1_results"
    BASE2 = "/Users/apple/StudioProjects/last_project_emnlp/without facilitator/other models/interruption/results/batch_2_results"
    OUT = "/Users/apple/StudioProjects/last_project_emnlp/interruption_full_evaluation"

    run_all(FAC1, FAC2, BASE1, BASE2, OUT)
