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

# ==========================================================
### 2. CONFIGURATION & DIALOGUE ACT LABELS ###
# ==========================================================
DA_LABELS_RAW = [
    'greeting', 'question', 'answer', 'statement-opinion', 'statement-non-opinion',
    'apology', 'command', 'agreement', 'disagreement', 'acknowledge', 'backchannel', 'others'
]
DA_LABELS_SET = set(label.strip().lower() for label in DA_LABELS_RAW)
DA_LABELS_LIST = list(DA_LABELS_SET)
DA_LABELS_CSV = ', '.join(DA_LABELS_LIST)

FEW_SHOT_SAMPLES = [
    {"history": "None (Start of dialogue)", "target": "Hello! How are you doing today?", "label": "greeting"},
    {"history": "Turn: Hello! How are you doing today?", "target": "I'm doing great, thanks for asking!", "label": "statement-non-opinion"},
    {"history": "Turn: Hello! How are you doing today?\nTurn: I'm doing great, thanks for asking!", "target": "What are your plans for the weekend?", "label": "question"},
    {"history": "Turn: What are your plans for the weekend?", "target": "I'm going hiking with some friends.", "label": "answer"},
    {"history": "Turn: I'm going hiking with some friends.", "target": "That sounds like an amazing adventure!", "label": "statement-opinion"},
    {"history": "Turn: That sounds like an amazing adventure!", "target": "Yes, I totally agree with you!", "label": "agreement"},
    {"history": "Turn: Yes, I totally agree with you!", "target": "Actually, I think rain is forecasted.", "label": "disagreement"},
    {"history": "Turn: Actually, I think rain is forecasted.", "target": "Oh no, I forgot to check the weather!", "label": "apology"},
    {"history": "Turn: Oh no, I forgot to check the weather!", "target": "Please bring an umbrella just in case.", "label": "command"},
    {"history": "Turn: Please bring an umbrella just in case.", "target": "Okay, I will do that.", "label": "acknowledge"},
    {"history": "Turn: Okay, I will do that.", "target": "Uh-huh, that makes sense.", "label": "backchannel"},
    {"history": "Turn: Uh-huh, that makes sense.", "target": "[Sighs] I don't know what to say.", "label": "others"},
]

def format_few_shot_examples(samples):
    formatted = []
    for s in samples:
        formatted.append(f"History:\n{s['history']}\nTarget Utterance: {s['target']}\nDialogue Act: {s['label']}")
    return "\n\n".join(formatted)

FEW_SHOT_EXAMPLES_STRING = format_few_shot_examples(FEW_SHOT_SAMPLES)

CLASSIFICATION_INSTRUCTION = f"""You are an expert dialogue act classifier.
Your task is to classify the 'Target' utterance based on the 'History'.

AUTHORIZED LABELS: [{DA_LABELS_CSV}]

DEFINITIONS:
- greeting: Opening a conversation. (e.g., "Hello!", "Hey there!")
- question: Any inquiry or request for information. (e.g., "What are we doing?", "Do you have it?")
- answer: A direct response to a previous question. (e.g., "We are making cookies.", "Yes, I have them.")
- statement-opinion: Subjective judgments, beliefs, or feelings. (e.g., "I think this is fun.", "That sounds great!")
- statement-non-opinion: Objective facts, status updates, or descriptions of current actions. (e.g., "I am adding the flour now.", "The oven is preheated.")
- apology: Expressing regret or correcting your own error. (e.g., "Sorry, I meant three eggs.")
- command: A direct instruction or order. (e.g., "Pour the milk.", "Pass me the bowl.")
- agreement: Confirming or accepting a suggestion/statement. (e.g., "Exactly.", "I agree.", "Yes, let's do that.")
- disagreement: Rejecting a suggestion or correcting a statement. (e.g., "I don't think so.", "No, that's wrong.")
- acknowledge: Showing you understood without necessarily agreeing. (e.g., "Okay.", "I see.", "Got it.")
- backchannel: Signals showing you are listening. (e.g., "Uh-huh.", "Mhm.")
- others: Filler or non-verbal sounds. (e.g., "[Laughs]", "Hmm...")

RULES:
1. Respond with ONLY the label name. No explanation.
2. If the Target describes an action being taken (e.g., "I'm setting up the camera"), classify as 'statement-non-opinion'.
3. A response to a question is 'answer'.

Here are a few examples to guide your classification:
{FEW_SHOT_EXAMPLES_STRING}

--- End of Examples ---
Your final response must ONLY contain the predicted dialogue act label. Do not write any explanation or surrounding text, and do not use quotes around the label."""


# ==========================================================
### 3. MODEL CONFIGURATIONS ###
# ==========================================================
MODEL_CONFIGS = {
    "Llama-3.1-8B": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "trust_remote_code": False,
        "supports_system": True,
        "max_new_tokens": 10,
        "repetition_penalty": 1.05,
        "extra_stop_token_ids": [128009],
        "fallback_template": "llama3",  # <|begin_of_text|><|start_header_id|>system... format
    },
    "Llama-3.2-3B": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "trust_remote_code": False,
        "supports_system": True,
        "max_new_tokens": 10,
        "repetition_penalty": 1.1,
        "extra_stop_token_ids": [128009],
        "fallback_template": "llama3",
    },
    "Qwen2.5-7B": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "trust_remote_code": False,
        "supports_system": True,
        "max_new_tokens": 10,
        "repetition_penalty": 1.05,
        "extra_stop_token_ids": [151645, 151643],
        "fallback_template": "qwen",
    },
    "Qwen2.5-1.5B": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "trust_remote_code": False,
        "supports_system": True,
        "max_new_tokens": 10,
        "repetition_penalty": 1.1,
        "extra_stop_token_ids": [151645, 151643],
        "fallback_template": "qwen",
    },
    "Mistral-7B": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "trust_remote_code": False,
        "supports_system": False,  # CRITICAL: No system role support
        "max_new_tokens": 10,
        "repetition_penalty": 1.05,
        "extra_stop_token_ids": [2],
        "fallback_template": "mistral",  # [INST] ... [/INST] format
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



def build_messages(text, history_text, supports_system):
    """Builds messages list according to model-specific requirements."""
    user_prompt = f"History:\n{history_text}\nTarget Utterance: {text}\nDialogue Act:"

    if supports_system:
        return [
            {"role": "system", "content": CLASSIFICATION_INSTRUCTION},
            {"role": "user", "content": user_prompt}
        ]
    else:
        # Mistral: merge system into user message
        full_prompt = f"{CLASSIFICATION_INSTRUCTION}\n\n{user_prompt}"
        return [{"role": "user", "content": full_prompt}]


def build_fallback_prompt(text, history_text, template_type):
    """Model-specific fallback when apply_chat_template fails."""
    user_prompt = f"History:\n{history_text}\nTarget Utterance: {text}\nDialogue Act:"

    if template_type == "llama3":
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{CLASSIFICATION_INSTRUCTION}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    elif template_type == "qwen":
        return f"<|im_start|>system\n{CLASSIFICATION_INSTRUCTION}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

    elif template_type == "mistral":
        full_prompt = f"{CLASSIFICATION_INSTRUCTION}\n\n{user_prompt}"
        return f"[INST] {full_prompt} [/INST]"

    elif template_type == "internlm":
        return f"<|im_start|>system\n{CLASSIFICATION_INSTRUCTION}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

    else:
        return f"{CLASSIFICATION_INSTRUCTION}\n\n{user_prompt}"


# ==========================================================
### 4. UTILITIES & DATA LOADING ###
# ==========================================================
def get_file_pairs():
    """Matches audioXX.json with transcript_X.json (Files 1-35)."""
    pairs = []
    for i in range(36, 105):
        audio_file = f"/content/audio{i:02d}.json"
        truth_file = f"/content/transcript_{i}.json"
        if os.path.exists(audio_file) and os.path.exists(truth_file):
            pairs.append((audio_file, truth_file, i))
    return pairs


def clean_label(label_str: str) -> str:
    """Extracts the first valid DA label from the FIRST word/token of output only."""
    if not label_str or not isinstance(label_str, str):
        return "others"

    # Remove punctuation and normalize
    cleaned = re.sub(r'[^\w\s-]', ' ', label_str.lower()).strip()
    words = cleaned.split()

    if not words:
        return "others"

    # Check first word (handles multi-word labels like "statement opinion")
    first_word = words[0]
    if first_word in DA_LABELS_SET:
        return first_word

    # Check first two words combined (for labels like "statement-opinion")
    if len(words) >= 2:
        two_words = f"{words[0]}-{words[1]}"
        if two_words in DA_LABELS_SET:
            return two_words

    # Check remaining words individually
    for word in words[1:]:
        if word in DA_LABELS_SET:
            return word

    # Check remaining pairs
    for i in range(1, len(words) - 1):
        pair = f"{words[i]}-{words[i+1]}"
        if pair in DA_LABELS_SET:
            return pair

    return "others"


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
        # FIX: Set model_max_length DURING tokenizer initialization
        tokenizer = AutoTokenizer.from_pretrained(
            config["hf_id"], 
            trust_remote_code=config["trust_remote_code"],
            padding_side="left",
            model_max_length=2048  # ← SET HERE, not after
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # REMOVE: tokenizer.model_max_length = 2048  # This was causing InternLM crash!

        model = AutoModelForCausalLM.from_pretrained(
            config["hf_id"], 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=config["trust_remote_code"], 
            low_cpu_mem_usage=True
        )
        model.eval()
        print(f"   Model loaded on {next(model.parameters()).device}")

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

                # Fetch ground truth
                golden = "others"
                if i < len(truth_utts):
                    golden = str(truth_utts[i].get('dialogue_act', '')).lower().strip().replace('_', '-')

                # Build history context (last 2 turns)
                history_list = audio_utts[max(0, i-2):i]
                history_text = "\n".join([f"Turn: {h.get('text', '')}" for h in history_list])
                if not history_text:
                    history_text = "None (Start of dialogue)"

                messages = build_messages(text, history_text, config["supports_system"])

                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                except Exception as e:
                    print(f"\n⚠️ Chat template failed for {model_name}: {e}")
                    prompt = build_fallback_prompt(text, history_text, config.get("fallback_template", "default"))

                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True,
                    max_length=2048
                ).to(model.device)

                gen_kwargs = {
                    "max_new_tokens": config["max_new_tokens"],
                    "do_sample": False,  # CONSISTENT: greedy decoding for all models
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": [tokenizer.eos_token_id] + config["extra_stop_token_ids"],
                    "repetition_penalty": config["repetition_penalty"],
                    "use_cache": True,  # FIX: Always True, never disable
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
        folder_path = os.path.join(base_output_dir, folder)

        for fn in os.listdir(folder_path):
            if fn.endswith(".json"):
                with open(os.path.join(folder_path, fn), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for u in data['utterances']:
                        gold_std = u['golden'].replace('_', '-')
                        if gold_std in DA_LABELS_SET:
                            y_true.append(gold_std)
                            y_pred.append(u['predicted'])

        if y_true:
            acc = accuracy_score(y_true, y_pred)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, 
                labels=DA_LABELS_LIST, 
                average="macro", 
                zero_division=0
            )
            summary.append({
                "Model": folder, 
                "Samples": len(y_true),
                "Accuracy": f"{acc:.4f}", 
                "Macro-F1": f"{f1:.4f}"
            })
        else:
            summary.append({
                "Model": folder, 
                "Samples": 0,
                "Accuracy": "N/A", 
                "Macro-F1": "N/A"
            })

    df = pd.DataFrame(summary)
    df = df.sort_values("Macro-F1", ascending=False, key=lambda x: pd.to_numeric(x, errors='coerce'))
    df.to_csv("da_results_summary.csv", index=False)

    print("\n" + "="*60)
    print("### BENCHMARK SUMMARY ###")
    print("="*60)
    print(df.to_markdown(index=False) if not df.empty else "No results to display.")
    print("="*60)

    return df


# ==========================================================
### 7. EXECUTION ###
# ==========================================================
RESULT_DIR = "da_prediction_results"
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

    shutil.make_archive("da_results_5models", 'zip', RESULT_DIR)
    print("\n📦 Results packaged: da_results_5models.zip")

    try:
        files.download("da_results_5models.zip")
        files.download("da_results_summary.csv")
        print("📥 Downloads initiated.")
    except Exception as e:
        print(f"⚠️ Download failed: {e}")

else:
    print("❌ No matching audio/transcript pairs found in /content.")
