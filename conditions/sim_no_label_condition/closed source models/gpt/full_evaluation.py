import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ==========================================================
# 1. CONFIGURATION & PATHS
# ==========================================================

OUTPUT_DIR = "/Users/apple/StudioProjects/last_project_emnlp/with facilitator/open source model/gpt/results" #"/Users/apple/StudioProjects/last_project_emnlp/without facilitator/gpt/results"#"/Users/apple/StudioProjects/last_project_emnlp/without facilitator/gpt/results"#
ORIGINAL_DIR ="/Users/apple/StudioProjects/last_project_emnlp/facilitator/batch_2_facilitator_gpt_labeled" #"#"/Users/apple/StudioProjects/last_project_emnlp/labeled_data_original"# #
PREDICTION_DIR ="/Users/apple/StudioProjects/last_project_emnlp/with facilitator/open source model/gpt/gpt_prediction" #"/Users/apple/StudioProjects/last_project_emnlp/without facilitator/gpt/gpt_labels"#"/Users/apple/StudioProjects/last_project_emnlp/without facilitator/gpt/gpt_labels"#
PRED_PREFIX ="expert_labeled_file"#"labeled_audio" #
NUM_FILES = 104

# ==========================================================
# TASK CONFIGURATIONS
# ==========================================================

TASKS = {
    "dialogue_act": {
        "key": "dialogue_act",
        "labels": [
            'greeting', 'question', 'answer', 'statement-opinion', 'statement-non-opinion',
            'apology', 'command', 'agreement', 'disagreement', 'acknowledge', 'backchannel', 'others'
        ],
        "null_class": "others",
        "cmap": "Purples",
        "normalize_fn": lambda t: str(t).lower().strip().replace("_", "-").replace(" ", "-") if t else "others"
    },
    "emotion": {
        "key": "emotion",
        "labels": [
            'angry', 'happy', 'sad', 'neutral', 'frustrated', 
            'excited', 'surprise', 'fear', 'joy', 'disgust'
        ],
        "null_class": "none",
        "cmap": "Reds",
        "normalize_fn": lambda t: _normalize_emotion(t)
    },
    "intent": {
        "key": "intent",
        "labels": [
            "complain", "praise", "apologize", "thank", "criticize", "agree", "taunt", "flaunt", "joke",
            "oppose", "comfort", "care", "inform", "advise", "arrange", "introduce", "leave", "prevent",
            "greet", "ask for help", "doubt", "acknowledge", "refuse", "warn", "emphasize", 
            "asking for opinions", "confirm", "explain", "invite", "plan"
        ],
        "null_class": "none",
        "cmap": "Blues",
        "normalize_fn": lambda t: _normalize_intent(t)
    },
    "generation_gap_indicator": {
        "key": "generation_gap_indicator",
        "labels": [
            'slang mismatch', 'perceived disrespect', 'stereotyped expectation', 
            'ignored input', 'method divergence', 'digital disparity', 'none'
        ],
        "null_class": "none",
        "cmap": "BuGn",
        "normalize_fn": lambda t: str(t).lower().strip().replace("_", " ").replace("-", " ") if t else "none",
        "extra_modes": ["no_double_none"]
    },
    "interruption_role": {
        "key": "interruption_role",
        "labels": None,
        "null_class": "none",
        "cmap": "Greens",
        "normalize_fn": lambda t: str(t).lower().strip() if t else "none"
    },
    "interruption_type": {
        "key": "interruption_type",
        "labels": None,
        "null_class": "none",
        "cmap": "Greens",
        "normalize_fn": lambda t: str(t).lower().strip() if t else "none"
    }
}


def _normalize_emotion(text):
    if not text: return "none"
    text = str(text).lower().strip().replace("_", " ").replace("-", " ")
    synonyms = {"anger": "angry", "sadness": "sad", "happiness": "happy"}
    return synonyms.get(text, text)


def _normalize_intent(text):
    if not text: return "none"
    text = str(text).lower().strip().replace("_", " ").replace("-", " ")
    if text == "apologise": return "apologize"
    return text


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def calculate_metrics_set(yt, yp):
    acc = accuracy_score(yt, yp)
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)
    wp, _, wf1, _ = precision_recall_fscore_support(yt, yp, average='weighted', zero_division=0)
    return [acc, p, r, f1, wp, wf1]


def safe_json_load(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    return raw.get('utterances', [])


def extract_data(task_key, normalize_fn):
    y_true, y_pred = [], []
    found_count = 0

    for i in range(1, NUM_FILES + 1):
        orig_path = os.path.join(ORIGINAL_DIR, f"gpt_{i}.json")
        pred_path = os.path.join(PREDICTION_DIR, f"{PRED_PREFIX}{i}.json")

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


# ==========================================================
# MAIN PROCESSING PIPELINE
# ==========================================================

def process_task(task_name, config):
    print(f"\n{'='*60}")
    print(f"🔍 PROCESSING: {task_name.upper().replace('_', ' ')}")
    print(f"{'='*60}")

    task_dir = ensure_dir(os.path.join(OUTPUT_DIR, task_name.replace("_", "")))
    key = config["key"]
    norm_fn = config["normalize_fn"]
    null_class = config["null_class"]
    cmap = config["cmap"]

    y_true, y_pred = extract_data(key, norm_fn)
    if not y_true:
        print(f"⚠️ No data found for {task_name}. Skipping.")
        return

    if config.get("labels"):
        labels = [norm_fn(l) for l in config["labels"]]
    else:
        labels = sorted(list(set(y_true + y_pred)))

    overall = calculate_metrics_set(y_true, y_pred)

    y_true_filt = [t for t, p in zip(y_true, y_pred) if t != null_class]
    y_pred_filt = [p for t, p in zip(y_true, y_pred) if t != null_class]
    detection = calculate_metrics_set(y_true_filt, y_pred_filt) if y_true_filt else [0]*6

    report_rows = {
        "Metric": ["ACC", "P", "R", "F1", "WP", "WF1"],
        "Overall (All)": [round(x, 4) for x in overall],
        "Detection (Non-None)": [round(x, 4) for x in detection]
    }

    extra_reports = {}
    if config.get("extra_modes"):
        for mode in config["extra_modes"]:
            if mode == "no_double_none":
                y_t_ndn = []
                y_p_ndn = []
                for t, p in zip(y_true, y_pred):
                    if not (t == null_class and p == null_class):
                        y_t_ndn.append(t)
                        y_p_ndn.append(p)
                if y_t_ndn:
                    ndn_metrics = calculate_metrics_set(y_t_ndn, y_p_ndn)
                    report_rows["No-Double-None"] = [round(x, 4) for x in ndn_metrics]
                    extra_reports["no_double_none"] = (y_t_ndn, y_p_ndn)

    report_df = pd.DataFrame(report_rows)
    csv_path = os.path.join(task_dir, f"{task_name}_dual_performance.csv")
    report_df.to_csv(csv_path, index=False)

    if config.get("labels"):
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        class_df = pd.DataFrame({
            "Label": config["labels"],
            "P": p_c, "R": r_c, "F1": f1_c
        })
        class_df.to_csv(os.path.join(task_dir, f"{task_name}_per_class.csv"), index=False)

    _plot_confusion_matrix(
        y_true, y_pred, labels,
        title=f"{task_name.replace('_', ' ').title()} — Full",
        cmap=cmap,
        save_path=os.path.join(task_dir, f"{task_name}_matrix_full.png")
    )

    if y_true_filt:
        non_none_labels = [l for l in labels if l != null_class]
        if non_none_labels:
            _plot_confusion_matrix(
                y_true_filt, y_pred_filt, non_none_labels,
                title=f"{task_name.replace('_', ' ').title()} — Non-None Only",
                cmap="Blues",
                save_path=os.path.join(task_dir, f"{task_name}_matrix_non_none.png")
            )

    for mode, (yt_m, yp_m) in extra_reports.items():
        _plot_confusion_matrix(
            yt_m, yp_m, labels,
            title=f"{task_name.replace('_', ' ').title()} — No Double None",
            cmap="Blues",
            save_path=os.path.join(task_dir, f"{task_name}_matrix_{mode}.png")
        )

    print(f"\n📊 {task_name.upper().replace('_', ' ')} REPORT")
    print(report_df.to_string(index=False))
    print(f"   Total: {len(y_true)} | Non-None: {len(y_true_filt)}")
    print(f"   💾 Saved to: {task_dir}")


def _plot_confusion_matrix(y_true, y_pred, labels, title, cmap, save_path):
    plt.figure(figsize=(max(10, len(labels)), max(8, len(labels) * 0.6)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual (Ground Truth)')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    ensure_dir(OUTPUT_DIR)
    print(f"🚀 Starting Unified Evaluation Pipeline")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   Source: {ORIGINAL_DIR}")
    print(f"   Predictions: {PREDICTION_DIR} ({PRED_PREFIX}*.json)")

    for task_name, config in TASKS.items():
        process_task(task_name, config)

    print(f"\n{'='*60}")
    print(f"✅ ALL TASKS COMPLETE")
    print(f"   Results: {OUTPUT_DIR}")
    print(f"{'='*60}")