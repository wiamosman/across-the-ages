import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_intent_classification(results_base_path):
    all_metrics = []
    
    # Normalize path
    results_base_path = os.path.normpath(results_base_path)
    
    if not os.path.exists(results_base_path):
        print(f"❌ Path not found: {results_base_path}")
        return

    # Iterate through each model folder (e.g., Llama-3.1-8B, Qwen2.5-7B, etc.)
    model_folders = [d for d in os.listdir(results_base_path) if os.path.isdir(os.path.join(results_base_path, d))]
    
    if not model_folders:
        print(f"⚠️ No subdirectories found in {results_base_path}")
        return

    for model_name in model_folders:
        model_path = os.path.join(results_base_path, model_name)
        y_true = []
        y_pred = []
        unknown_count = 0
        total_count = 0

        # Load all JSON prediction files in the model's folder
        for filename in os.listdir(model_path):
            if filename.endswith(".json"):
                with open(os.path.join(model_path, filename), 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        # The intent code structure uses data['utterances']
                        utterances = data.get('utterances', [])
                        
                        for u in utterances:
                            total_count += 1
                            gold = str(u.get('golden', '')).strip().lower()
                            pred = str(u.get('predicted', '')).strip().lower()

                            if pred == "unknown":
                                unknown_count += 1
                            
                            # We only evaluate if there is a valid ground truth
                            if gold and gold != "unknown":
                                y_true.append(gold)
                                y_pred.append(pred)
                    except Exception as e:
                        print(f"⚠️ Error reading {filename} in {model_name}: {e}")

        if not y_true:
            print(f"⏩ Skipping {model_name}: No valid samples found.")
            continue

        # Calculate Metrics
        acc = accuracy_score(y_true, y_pred)
        # Macro: average of F1 per class (crucial for intent because labels are imbalanced)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        # Weighted: F1 weighted by the number of true instances per class
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        unknown_rate = (unknown_count / total_count) if total_count > 0 else 0

        all_metrics.append({
            "Model": model_name,
            "Accuracy": acc,
            "Precision (Macro)": p_macro,
            "Recall (Macro)": r_macro,
            "F1 (Macro)": f1_macro,
            "F1 (Weighted)": f1_weighted,
            "Unknown Rate": unknown_rate,
            "Total Utterances": total_count
        })
        print(f"✅ Evaluated {model_name}")

    # Create Summary Table
    df = pd.DataFrame(all_metrics)
    if not df.empty:
        df = df.sort_values(by="F1 (Macro)", ascending=False)
        
        # Formatting for terminal
        pd.options.display.float_format = '{:.4f}'.format
        print("\n" + "="*90)
        print("🎯 INTENT CLASSIFICATION BENCHMARK RESULTS")
        print("="*90)
        print(df.to_string(index=False))
        print("="*90)
        
        # Save to CSV
        output_file = "intent_final_evaluation.csv"
        df.to_csv(output_file, index=False)
        print(f"📊 Results saved to: {os.path.abspath(output_file)}")
    else:
        print("❌ Evaluation failed: No data collected.")

# --- EXECUTION ---
path_to_results = "/Users/apple/StudioProjects/last_project_emnlp/without facilitator/other models/intent/results/intent_results"
evaluate_intent_classification(path_to_results)