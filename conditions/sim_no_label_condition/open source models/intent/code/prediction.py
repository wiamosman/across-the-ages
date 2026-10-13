# ==========================================================
# INTENT CLASSIFICATION BENCHMARK - Phase 1
# 5 Models on Transformers 5.0.0 (NO rope_scaling fix needed!)
# ==========================================================

import torch
import json
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm
import re
import gc
import shutil
import traceback
from google.colab import files, userdata
from huggingface_hub import login

# ==========================================================
### 1. SETUP & AUTHENTICATION ###
# ==========================================================
try:
    hf_token = userdata.get('HF_TOKEN')
    if hf_token:
        login(token=hf_token)
        print("✅ Authenticated with Hugging Face.")
except Exception as e:
    print(f"⚠️ HF Login skipped: {e}")

print(f"Transformers version: {__import__('transformers').__version__}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================================
### 2. INTENT DEFINITIONS & PROMPTS ###
# ==========================================================
INTENT_LABELS_RAW = [
    "complain", "praise", "apologise", "thank", "criticize", "agree", "taunt", "flaunt", "joke",
    "oppose", "comfort", "care", "inform", "advise", "arrange", "introduce", "leave", "prevent",
    "greet", "ask for help", "acknowledge", "asking for opinions", "confirm", "explain", "warn", "doubt", "refuse", "plan", "invite", "emphasize"
]

INTENT_LABELS_SET = set(label.strip().lower() for label in INTENT_LABELS_RAW)
INTENT_LABELS_LIST = list(INTENT_LABELS_SET)
INTENT_LABELS_CSV = ', '.join(INTENT_LABELS_LIST)


FEW_SHOT_SAMPLES = [
    {"text": "Hello, welcome back to the show.", "intent": "greet"},
    {"text": "That's not right; the oven temperature is too low.", "intent": "criticize"},
    {"text": "I think we should put the cake in the fridge for an hour first. What do you think?", "intent": "asking for opinions"},
    {"text": "I'm so sorry, I totally forgot the butter.", "intent": "apologise"},
    {"text": "That's a fantastic idea, I agree!", "intent": "praise"},
    {"text": "Aha, I see what you mean.", "intent": "acknowledge"},
]

def format_few_shot_examples(samples):
    return "\n\n".join([f"Utterance: {s['text']}\nIntent: {s['intent']}" for s in samples])

FEW_SHOT_EXAMPLES_STRING = format_few_shot_examples(FEW_SHOT_SAMPLES)

CLASSIFICATION_INSTRUCTION = f"""You are an expert dialogue intent classifier. Your task is to classify the intent expressed in the dialogue utterance provided.

The intent must be EXACTLY ONE of the authorized intent categories listed here (Total of {len(INTENT_LABELS_LIST)}):
[{INTENT_LABELS_CSV}]

Here are a few examples to guide your classification:
{FEW_SHOT_EXAMPLES_STRING}

--- End of Examples ---
Your final response must ONLY contain the predicted intent label (e.g., 'greet', 'thank', 'apologise'). Do not write any explanation or surrounding text, and do not use quotes around the label."""

# ==========================================================
### 3. MODEL CONFIGURATIONS (5 Models on Transformers 5.0) ###
# ==========================================================
# NOTE: NO rope_scaling fix needed! Transformers 5.0 handles it automatically.
MODEL_CONFIGS = {
    "Llama-3.1-8B": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "trust_remote_code": False,
        "supports_system": True,
        "max_new_tokens": 8,
        "repetition_penalty": 1.05,
        "extra_stop_token_ids": [128009],
    },
    "Llama-3.2-3B": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "trust_remote_code": False,
        "supports_system": True,
        "max_new_tokens": 8,
        "repetition_penalty": 1.1,
        "extra_stop_token_ids": [128009],
    },
    "Qwen2.5-7B": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "trust_remote_code": False,
        "supports_system": True,
        "max_new_tokens": 8,
        "repetition_penalty": 1.05,
        "extra_stop_token_ids": [151645, 151643],
    },
    "Qwen2.5-1.5B": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "trust_remote_code": False,
        "supports_system": True,
        "max_new_tokens": 8,
        "repetition_penalty": 1.1,
        "extra_stop_token_ids": [151645, 151643],
    },
    "Mistral-7B": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "trust_remote_code": False,
        "supports_system": False,  # CRITICAL: No system role support
        "max_new_tokens": 8,
        "repetition_penalty": 1.05,
        "extra_stop_token_ids": [2],
    },
   "InternLM-2.5-7B": {
    "hf_id": "internlm/internlm2_5-7b-chat",
    "trust_remote_code": True,
    "supports_system": True,
    "max_new_tokens": 30,
    "repetition_penalty": 1.05,
    "extra_stop_token_ids": [92542, 92543],
}
}




# ==========================================================
### 4. UTILITIES & DATA LOADING ###
# ==========================================================
def get_file_pairs():
    """Matches audioXX.json with transcript_X.json (Files 36-104)."""
    pairs = []
    for i in range(36, 105):
        audio_file = f"/content/audio{i:02d}.json"
        truth_file = f"/content/transcript_{i}.json"
        if os.path.exists(audio_file) and os.path.exists(truth_file):
            pairs.append((audio_file, truth_file, i))
    return pairs


def clean_label(label_str: str) -> str:
    """Extracts the first valid intent from model output."""
    if not label_str or not isinstance(label_str, str):
        return "unknown"

    cleaned = re.sub(r'[^\w\s]', ' ', label_str.lower()).strip()
    words = cleaned.split()
    if not words:
        return "unknown"

    # Check first word
    first_word = words[0]
    if first_word in INTENT_LABELS_SET:
        return first_word

    # Check all words for multi-word labels (n-gram matching)
    for n in range(min(4, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            candidate = " ".join(words[i:i+n])
            if candidate in INTENT_LABELS_SET:
                return candidate




def build_messages(text, supports_system):
    """Builds messages list according to model-specific requirements."""
    if supports_system:
        return [
            {"role": "system", "content": CLASSIFICATION_INSTRUCTION},
            {"role": "user", "content": f"Utterance: {text}\nIntent:"}
        ]
    else:
        # Mistral: merge system into user message
        full_prompt = f"{CLASSIFICATION_INSTRUCTION}\n\nUtterance: {text}\nIntent:"
        return [{"role": "user", "content": full_prompt}]


# ==========================================================
### 5. INFERENCE ENGINE ###
# ==========================================================
def run_model(model_name, config, file_pairs, base_output_dir):
    print(f"\n{'='*60}")
    print(f"🚀 STARTING: {model_name}")
    print(f"   HF ID: {config['hf_id']}")
    print(f"{'='*60}")

    sanitized_name = model_name.replace(" ", "_").replace(".", "_")
    model_path = os.path.join(base_output_dir, sanitized_name)
    os.makedirs(model_path, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config["hf_id"], 
            trust_remote_code=config["trust_remote_code"],
            padding_side="left"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        tokenizer.model_max_length = 2048

        # Apply custom chat template for Mistral v0.2
        if model_name == "Mistral-7B":
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
                "{% elif message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]"
                "{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}"
                "{% endif %}"
                "{% endfor %}"
            )
            print("   🔧 Applied custom chat template for Mistral v0.2")

        # Load model - NO rope_scaling fix needed for transformers 5.0!
        model = AutoModelForCausalLM.from_pretrained(
            config["hf_id"], 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=config["trust_remote_code"], 
            low_cpu_mem_usage=True
        )
        model.eval()
        print(f"   ✅ Model loaded on {next(model.parameters()).device}")

        # Verify stop tokens
        if "extra_stop_token_ids" in config:
            print(f"   🔍 Stop tokens: {config['extra_stop_token_ids']}")
            for tid in config["extra_stop_token_ids"]:
                try:
                    tok = tokenizer.convert_ids_to_tokens([tid])
                    print(f"      ID {tid} -> {tok}")
                except:
                    print(f"      ID {tid} -> [unknown]")

        for audio_path, truth_path, idx in file_pairs:
            with open(audio_path, 'r', encoding='utf-8') as f:
                audio_content = json.load(f)
            with open(truth_path, 'r', encoding='utf-8') as f:
                truth_content = json.load(f)

            audio_utts = audio_content.get('utterances', audio_content)
            truth_utts = truth_content.get('utterances', truth_content)

            slim_results = {
                "model_used": model_name, 
                "hf_id": config["hf_id"],
                "file_index": idx, 
                "utterances": []
            }

            for i in tqdm(range(len(audio_utts)), desc=f"{model_name} - File {idx}"):
                text = audio_utts[i].get('text', '')
                golden = str(truth_utts[i].get('intent', '')).lower().strip() if i < len(truth_utts) else "unknown"

                messages = build_messages(text, config["supports_system"])

                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                except Exception as e:
                    print(f"\n⚠️ Chat template failed: {e}")
                    prompt = f"{CLASSIFICATION_INSTRUCTION}\n\nUtterance: {text}\nIntent: "

                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True,
                    max_length=2048
                ).to(model.device)

                # Build eos_token_id list
                eos_ids = [tokenizer.eos_token_id]
                if "extra_stop_token_ids" in config:
                    eos_ids.extend(config["extra_stop_token_ids"])

                gen_kwargs = {
                    "max_new_tokens": config["max_new_tokens"],
                    "do_sample": False,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": eos_ids,
                    "repetition_penalty": config["repetition_penalty"],
                }

                with torch.inference_mode():
                    output = model.generate(**inputs, **gen_kwargs)

                generated_ids = output[0][inputs.input_ids.shape[-1]:]
                raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
                predicted = clean_label(raw_answer)

                slim_results["utterances"].append({
                    "text": text,
                    "golden": golden,
                    "predicted": predicted,
                    "raw": raw_answer.strip()
                })

            out_path = os.path.join(model_path, f"prediction_transcript_{idx}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(slim_results, f, indent=4, ensure_ascii=False)

            print(f"   ✓ Saved: {out_path}")

        print(f"✅ Completed: {model_name}")

    except Exception as e:
        print(f"\n❌ ERROR with {model_name}: {e}")
        traceback.print_exc()
    finally:
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"   🧹 Cleaned up GPU memory")


# ==========================================================
### 6. EVALUATION & EXPORT ###
# ==========================================================
def evaluate_and_save(base_output_dir):
    summary = []
    folders = [d for d in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, d))]

    if not folders:
        print("\n⚠️ No result folders found.")
        return

    for folder in folders:
        y_true, y_pred = [], []
        unknown_count = 0
        total_count = 0
        folder_path = os.path.join(base_output_dir, folder)

        for fn in os.listdir(folder_path):
            if fn.endswith(".json"):
                with open(os.path.join(folder_path, fn), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for u in data['utterances']:
                        total_count += 1
                        pred = u['predicted']
                        golden = u['golden']

                        if pred == "unknown":
                            unknown_count += 1

                        if pred != "unknown" and golden in INTENT_LABELS_SET:
                            y_true.append(golden)
                            y_pred.append(pred)

        if y_true:
            acc = accuracy_score(y_true, y_pred)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, 
                labels=INTENT_LABELS_LIST, 
                average="macro", 
                zero_division=0
            )
            unknown_rate = unknown_count / total_count if total_count > 0 else 0
            summary.append({
                "Model": folder, 
                "Samples": len(y_true),
                "Accuracy": f"{acc:.4f}", 
                "Macro-F1": f"{f1:.4f}",
                "Unknown_Rate": f"{unknown_rate:.4f}",
                "Total_Samples": total_count
            })
        else:
            summary.append({
                "Model": folder, 
                "Samples": 0,
                "Accuracy": "N/A", 
                "Macro-F1": "N/A",
                "Unknown_Rate": "N/A",
                "Total_Samples": total_count
            })

    df = pd.DataFrame(summary)
    df = df.sort_values("Macro-F1", ascending=False, key=lambda x: pd.to_numeric(x, errors='coerce'))
    df.to_csv("intent_results_summary.csv", index=False)

    print("\n" + "="*60)
    print("### INTENT CLASSIFICATION BENCHMARK SUMMARY ###")
    print("="*60)
    print(df.to_markdown(index=False) if not df.empty else "No results to display.")
    print("="*60)

    return df


# ==========================================================
### 7. EXECUTION ###
# ==========================================================
RESULT_DIR = "intent_prediction_results"
if os.path.exists(RESULT_DIR): 
    shutil.rmtree(RESULT_DIR)
os.makedirs(RESULT_DIR)

file_pairs = get_file_pairs()

if file_pairs:
    print(f"📂 Found {len(file_pairs)} valid audio/transcript pairs.")
    print(f"🎯 Phase 1: Running 5 models on transformers {__import__('transformers').__version__}")
    print(f"   (Llama-3.1, Llama-3.2, Qwen2.5-7B, Qwen2.5-1.5B, Mistral-7B)")
    print(f"\n⚠️ InternLM-2.5-7B will run separately with transformers 4.41.2")
    print(f"   Run this AFTER this script completes:")
    print(f"   !pip install transformers==4.41.2 --quiet")
    print(f"   Then run the InternLM-only cell below\n")

    for name, config in MODEL_CONFIGS.items():
        run_model(name, config, file_pairs, RESULT_DIR)

    evaluate_and_save(RESULT_DIR)

    shutil.make_archive("intent_results_5models", 'zip', RESULT_DIR)
    print("\n📦 Results packaged: intent_results_5models.zip")

    try:
        files.download("intent_results_5models.zip")
        files.download("intent_results_summary.csv")
        print("📥 Downloads initiated.")
    except Exception as e:
        print(f"⚠️ Download failed: {e}")

else:
    print("❌ No matching audio/transcript pairs found in /content.")
