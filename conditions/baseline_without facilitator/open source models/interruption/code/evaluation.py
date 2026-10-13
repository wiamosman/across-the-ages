import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def run_evaluation(paths_to_search):
    all_metrics = []
    
    for root_path in paths_to_search:
        # Normalize path for Mac/Unix
        root_path = os.path.normpath(root_path)
        
        if not os.path.exists(root_path):
            print(f"⚠️ Path not found, skipping: {root_path}")
            continue
            
        subdirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
        
        # If the folder has no subdirectories, evaluate the folder itself
        # Otherwise, evaluate every subfolder inside it
        targets = [os.path.join(root_path, d) for d in subdirs] if subdirs else [root_path]

        for target_dir in targets:
            model_name = os.path.basename(target_dir)
            g_roles, p_roles = [], []
            g_types, p_types = [], []
            
            # Read every JSON result file
            files_found = 0
            for file in os.listdir(target_dir):
                if file.endswith(".json"):
                    files_found += 1
                    with open(os.path.join(target_dir, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for item in data:
                            # Use .get() to avoid KeyErrors if a model failed to output a field
                            g_roles.append(str(item.get('gold_role', 'none')).lower())
                            p_roles.append(str(item.get('pred_role', 'none')).lower())
                            g_types.append(str(item.get('gold_type', 'none')).lower())
                            p_types.append(str(item.get('pred_type', 'none')).lower())
            
            if not g_roles:
                continue
                
            print(f"✅ Processed {model_name} ({files_found} files)")

            def get_scores(gold, pred, task):
                acc = accuracy_score(gold, pred)
                prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(gold, pred, average='macro', zero_division=0)
                prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(gold, pred, average='weighted', zero_division=0)
                
                return {
                    "Model": model_name,
                    "Task": task,
                    "Accuracy": acc,
                    "Precision (Macro)": prec_m,
                    "Recall (Macro)": rec_m,
                    "F1 (Macro)": f1_m,
                    "F1 (Weighted)": f1_w
                }

            all_metrics.append(get_scores(g_roles, p_roles, "Role"))
            all_metrics.append(get_scores(g_types, p_types, "Type"))

    return pd.DataFrame(all_metrics)

# --- CONFIGURATION ---
# The script was failing on InternLM-2_5-7B because of the underscore/dot naming. 
# Ensure these paths match your folder names exactly.
base_path = "/Users/apple/StudioProjects/last_project_emnlp/without facilitator/other models/interruption/results"

paths = [
    os.path.join(base_path, "Final_ChatTemplate_Results_Phase1"),
    os.path.join(base_path, "InternLM-2_5-7B") # Check if folder is actually InternLM-2.5-7B
]

# Run 
df_results = run_evaluation(paths)

if not df_results.empty:
    # Sorting and Formatting
    df_results = df_results.sort_values(by=["Task", "F1 (Macro)"], ascending=False)
    pd.options.display.float_format = '{:.4f}'.format
    
    print("\n" + "="*80)
    print("🔥 INTERRUPTION DETECTION EVALUATION RESULTS 🔥")
    print("="*80)
    # Using print(df) instead of display() for terminal output
    print(df_results.to_string(index=False))
    print("="*80)

    # Export
    output_csv = "interruption_metrics_summary.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\n📊 Metrics saved to: {os.path.abspath(output_csv)}")
else:
    print("\n❌ No data found. Check your folder paths and JSON file structure.")