import os
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

# ==========================================================
# PATHS TO BOTH RESULT FOLDERS
# ==========================================================
RESULT_DIRS = [
   # "/Users/apple/StudioProjects/last_project_emnlp/without facilitator/other models/intent/results/batch_1_intent_results",
    #"/Users/apple/StudioProjects/last_project_emnlp/without facilitator/other models/intent/results/batch_2_intent_results"
"/Users/apple/StudioProjects/last_project_emnlp/with facilitator/intent/results/batch_1_intent_results",
"/Users/apple/StudioProjects/last_project_emnlp/with facilitator/intent/results/batch_2_intent_results"
]

# ==========================================================
# INTENT LABELS
# ==========================================================
INTENT_LABELS = [
    "complain", "praise", "apologise", "thank", "criticize",
    "agree", "taunt", "flaunt", "joke", "oppose", "comfort",
    "care", "inform", "advise", "arrange", "introduce",
    "leave", "prevent", "greet", "ask for help",
    "acknowledge", "asking for opinions", "confirm",
    "explain", "warn", "doubt", "refuse", "plan",
    "invite", "emphasize"
]

INTENT_SET = set(INTENT_LABELS)

# ==========================================================
# COLLECT ALL MODEL NAMES ACROSS BOTH BATCHES
# ==========================================================
all_models = set()

for result_dir in RESULT_DIRS:
    if os.path.exists(result_dir):
        for folder in os.listdir(result_dir):
            if os.path.isdir(os.path.join(result_dir, folder)):
                all_models.add(folder)

# ==========================================================
# EVALUATE EACH MODEL (MERGE DATA FROM BOTH BATCHES)
# ==========================================================
summary_results = []

for model_name in sorted(all_models):

    print(f"\nEvaluating: {model_name}")

    y_true = []
    y_pred = []

    # Gather data from both batch folders for this model
    for result_dir in RESULT_DIRS:

        model_folder_path = os.path.join(result_dir, model_name)

        if not os.path.exists(model_folder_path):
            continue

        print(f"  → Reading from: {model_folder_path}")

        for file_name in os.listdir(model_folder_path):

            if not file_name.endswith(".json"):
                continue

            file_path = os.path.join(model_folder_path, file_name)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            utterances = data.get("utterances", [])

            for utt in utterances:

                golden = utt.get("golden", "").strip().lower()
                predicted = utt.get("predicted", "").strip().lower()

                # Only evaluate valid gold labels
                if golden in INTENT_SET:
                    y_true.append(golden)
                    y_pred.append(predicted)

    if len(y_true) == 0:
        print(f"⚠️ No valid data in {model_name}")
        continue

    # ======================================================
    # METRICS
    # ======================================================
    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=INTENT_LABELS,
        average="macro",
        zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=INTENT_LABELS,
        average="weighted",
        zero_division=0
    )

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=INTENT_LABELS,
        average="micro",
        zero_division=0
    )

    # Save results
    summary_results.append({
        "Model": model_name,
        "Samples": len(y_true),

        "Accuracy": round(accuracy, 4),

        "Macro Precision": round(macro_precision, 4),
        "Macro Recall": round(macro_recall, 4),
        "Macro F1": round(macro_f1, 4),

        "Weighted Precision": round(weighted_precision, 4),
        "Weighted Recall": round(weighted_recall, 4),
        "Weighted F1": round(weighted_f1, 4),

        "Micro Precision": round(micro_precision, 4),
        "Micro Recall": round(micro_recall, 4),
        "Micro F1": round(micro_f1, 4)
    })

# ==========================================================
# SAVE SUMMARY CSV
# ==========================================================
summary_df = pd.DataFrame(summary_results)

summary_df = summary_df.sort_values(
    by="Macro F1",
    ascending=False
)

# Save in the parent results directory
save_path = os.path.join(
    os.path.dirname(RESULT_DIRS[0]),
    "intent_evaluation_summary_combined.csv"
)

summary_df.to_csv(save_path, index=False)

# ==========================================================
# PRINT RESULTS
# ==========================================================
print("\n" + "=" * 80)
print("FINAL COMBINED RESULTS")
print("=" * 80)

print(summary_df.to_string(index=False))

print("=" * 80)
print(f"\nSaved to:\n{save_path}")