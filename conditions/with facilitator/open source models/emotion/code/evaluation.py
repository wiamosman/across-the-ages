import os
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

# ==========================================================
# CONFIGURATION
# ==========================================================

RESULTS_DIRS = [
    #"/Users/apple/StudioProjects/last_project_emnlp/without facilitator/other models/emotion/batch_1/emotion_results/",
    #"/Users/apple/StudioProjects/last_project_emnlp/without facilitator/other models/emotion/batch_2/emotion_results"
"/Users/apple/StudioProjects/last_project_emnlp/with facilitator/emotion/batch_1/emotion_resutls",
"/Users/apple/StudioProjects/last_project_emnlp/with facilitator/emotion/batch_2/emotion_results"
]

# EXACT SAME LABELS AS ORIGINAL SCRIPT
EMOTION_LABELS_RAW = ['neutral', 'surprise', 'fear', 'happy', 'sad', 'frustrated', 'excited', 'angry', 'disgust', 'joy']

EMOTION_LABELS_SET = set(
    label.strip().lower()
    for label in EMOTION_LABELS_RAW
)

EMOTION_LABELS_LIST = list(
    EMOTION_LABELS_SET
)

# ==========================================================
# EVALUATION
# ==========================================================

def evaluate():

    summary = []

    # Find all model folders
    folders = set()

    for result_dir in RESULTS_DIRS:

        folders.update([
            d for d in os.listdir(result_dir)
            if os.path.isdir(
                os.path.join(result_dir, d)
            )
            and d not in [
                'analysis',
                'backup_originals'
            ]
        ])

    folders = sorted(folders)

    for folder in folders:

        y_true = []
        y_pred = []

        total_files = 0

        # Read from BOTH batches
        for result_dir in RESULTS_DIRS:

            folder_path = os.path.join(
                result_dir,
                folder
            )

            if not os.path.exists(folder_path):
                continue

            for fn in os.listdir(folder_path):

                if not fn.endswith('.json'):
                    continue

                total_files += 1

                with open(
                    os.path.join(folder_path, fn),
                    'r',
                    encoding='utf-8'
                ) as f:

                    data = json.load(f)

                for u in data.get(
                    'utterances',
                    []
                ):

                    g = u.get(
                        'golden',
                        ''
                    ).lower().strip()

                    p = u.get(
                        'predicted',
                        ''
                    ).lower().strip()

                    # EXACT SAME FILTERING
                    if g in EMOTION_LABELS_SET:
                        y_true.append(g)
                        y_pred.append(p)

        # ==================================================
        # EXACT SAME METRICS
        # ==================================================

        if y_true:

            acc = accuracy_score(
                y_true,
                y_pred
            )

            _, _, macro_f1, _ = (
                precision_recall_fscore_support(
                    y_true,
                    y_pred,
                    labels=EMOTION_LABELS_LIST,
                    average="macro",
                    zero_division=0
                )
            )

            summary.append({
                "Model": folder,
                "Files": total_files,
                "Samples": len(y_true),
                "Accuracy": round(acc, 4),
                "Macro F1": round(macro_f1, 4)
            })

            print(
                f"\n🏷️ {folder}"
                f" | Files: {total_files}"
                f" | Samples: {len(y_true)}"
                f" | Acc: {acc:.4f}"
                f" | Macro F1: {macro_f1:.4f}"
            )

    # ======================================================
    # FINAL RESULTS
    # ======================================================

    df = pd.DataFrame(summary)

    df = df.sort_values(
        "Macro F1",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("FINAL COMBINED RESULTS")
    print("=" * 70)

    print(df.to_markdown(index=False))

    output_path = os.path.join(
        RESULTS_DIRS[0],
        "combined_results_summary.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\n📁 Saved to:\n"
        f"{output_path}"
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    evaluate()