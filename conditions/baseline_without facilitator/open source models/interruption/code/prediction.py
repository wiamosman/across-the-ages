import torch
import json
import os
import gc
import shutil
import re
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
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
### 2. INTERRUPTION TASK DEFINITIONS ###
# ==========================================================
ROLE_LABELS = ['interrupted speaker', 'intervening speaker', 'none']
TYPE_LABELS = ['disruptive intervention', 'cooperative assistance', 'cooperative agreement', 'none']

SYSTEM_PROMPT = """You are an expert conversational analyst specializing in Turn-taking and Overlap.

TASK:
Analyze if the 'Target' utterance is part of an interruption. 

RULES:

1. 'interrupted speaker': Use ONLY if the Target's text ends abruptly because Speaker B started talking.
2. 'intervening speaker': Use ONLY if the Target starts talking while the previous speaker was in the middle of a sentence.

ROLES:
- interrupted speaker: Target was cut off by the other person.
- intervening speaker: Target cut off the other person.
- none: Natural flow.

TYPES (Only for intervening speaker):
- disruptive intervention: Target changes the topic or steals the floor.
- cooperative assistance: Target finishes the other person's sentence or helps with a word.
- cooperative agreement: Target overlaps briefly to say "Right" or "Exactly".
- none: If role is not 'intervening speaker'.

OUTPUT FORMAT:
Role: <label>
Type: <label>"""

FEW_SHOT = [
    {
        "role": "user", 
        "content": "CONVERSATION HISTORY:\nSpeaker B: How long do you soak them for?\n\nTARGET FOR ANALYSIS:\nSpeaker A (Target): About 30 minutes."
    },
    {
        "role": "assistant", 
        "content": "Role: none\nType: none"
    },
    {
        "role": "user", 
        "content": "CONVERSATION HISTORY:\nSpeaker B: You need to add the sugar and then--\n\nTARGET FOR ANALYSIS:\nSpeaker A (Target): Wait, how much sugar?"
    },
    {
        "role": "assistant", 
        "content": "Role: intervening speaker\nType: disruptive intervention" 
    },
    {
        "role": "user", 
        "content": "CONVERSATION HISTORY:\nSpeaker B: First you wash the mushrooms and..\n\nTARGET FOR ANALYSIS:\nSpeaker A (Target): Then we fry them?"
    },
    {
        "role": "assistant", 
        "content": "Role: intervening speaker\nType: cooperative assistance"
    }
]

# ==========================================================
### 3. MODEL CONFIGURATIONS (from Emotion Code) ###
# ==========================================================
MODEL_CONFIGS = {
    "Llama-3.1-8B": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "trust_remote_code": False,
        "supports_system": True,
        "max_new_tokens": 30,
        "repetition_penalty": 1.05,
        "extra_stop_token_ids": [128009],
    },
    "Llama-3.2-3B": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "trust_remote_code": False,
        "supports_system": True,
        "max_new_tokens": 30,
        "repetition_penalty": 1.1,
        "extra_stop_token_ids": [128009],
    },
    "Qwen2.5-7B": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "trust_remote_code": False,
        "supports_system": True,
        "max_new_tokens": 30,
        "repetition_penalty": 1.05,
        "extra_stop_token_ids": [151645, 151643],
    },
    "Qwen2.5-1.5B": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "trust_remote_code": False,
        "supports_system": True,
        "max_new_tokens": 30,
        "repetition_penalty": 1.1,
        "extra_stop_token_ids": [151645, 151643],
    },
    "Mistral-7B": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "trust_remote_code": False,
        "supports_system": False,  # CRITICAL: No system role support
        "max_new_tokens": 30,
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
### 4. UTILITIES ###
# ==========================================================
def parse_output(raw_str):
    raw_str = raw_str.lower().strip()
    p_role, p_type = "none", "none"
    for r in ROLE_LABELS:
        if r in raw_str: p_role = r; break
    for t in TYPE_LABELS:
        if t in raw_str: p_type = t; break
    return p_role, p_type

def build_messages(context_final, target_line, supports_system):
    """Builds messages list according to model-specific requirements."""
    user_content = f"CONVERSATION HISTORY:\n{context_final}\n\nTARGET FOR ANALYSIS:\n{target_line}"
    
    if supports_system:
        return [{"role": "system", "content": SYSTEM_PROMPT}] + FEW_SHOT + [
            {"role": "user", "content": user_content}
        ]
    else:
        # Mistral: merge system prompt into user message
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_content}"
        return FEW_SHOT + [{"role": "user", "content": full_prompt}]

# ==========================================================
### 5. INFERENCE ENGINE ###
# ==========================================================
def run_model(model_name, config, base_dir, input_path='/content'):
    print(f"\n{'='*60}")
    print(f"🚀 STARTING: {model_name}")
    print(f"   HF ID: {config['hf_id']}")
    print(f"{'='*60}")

    sanitized_name = model_name.replace(" ", "_").replace(".", "_").replace("/", "_")
    model_folder = os.path.join(base_dir, sanitized_name)
    os.makedirs(model_folder, exist_ok=True)

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

        model = AutoModelForCausalLM.from_pretrained(
            config["hf_id"], 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=config["trust_remote_code"], 
            low_cpu_mem_usage=True
        )
        model.eval()
        print(f"   Model loaded on {next(model.parameters()).device}")

        for i in range(1, 36):
            audio_f = f"{input_path}/audio{i:02d}.json"
            truth_f = f"{input_path}/transcript_{i}.json"
            
            if not os.path.exists(audio_f) or not os.path.exists(truth_f):
                continue

            with open(audio_f, 'r', encoding='utf-8') as fa, open(truth_f, 'r', encoding='utf-8') as ft:
                audio_data = json.load(fa).get('utterances', [])
                truth_data = json.load(ft).get('utterances', [])
            
            file_results = []
            for idx in tqdm(range(len(audio_data)), desc=f"{model_name} - File {i}"):
                target_u = audio_data[idx]
                context_units = audio_data[max(0, idx-2):idx]
                
                ctx_str = ""
                for c_u in context_units:
                    s_label = "Speaker A" if c_u['speaker'] == target_u['speaker'] else "Speaker B"
                    ctx_str += f"{s_label} ({c_u['speaker']}): {c_u['text']}\n"
                
                context_final = ctx_str.strip() if ctx_str else "No prior context."
                target_line = f"Speaker A ({target_u['speaker']}) (Target): {target_u['text']}"
                
                messages = build_messages(context_final, target_line, config["supports_system"])

                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                except Exception as e:
                    print(f"\n⚠️ Chat template failed: {e}")
                    prompt = f"{SYSTEM_PROMPT}\n\nCONVERSATION HISTORY:\n{context_final}\n\nTARGET FOR ANALYSIS:\n{target_line}"

                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True,
                    max_length=2048
                ).to(model.device)

                gen_kwargs = {
                    "max_new_tokens": config["max_new_tokens"],
                    "do_sample": False,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": [tokenizer.eos_token_id] + config["extra_stop_token_ids"],
                    "repetition_penalty": config["repetition_penalty"],
                }

                with torch.inference_mode():
                    out = model.generate(**inputs, **gen_kwargs)

                generated_ids = out[0][inputs.input_ids.shape[-1]:]
                raw_gen = tokenizer.decode(generated_ids, skip_special_tokens=True)
                role, itype = parse_output(raw_gen)
                
                file_results.append({
                    "index": idx,
                    "context": context_final,
                    "target": target_line,
                    "gold_role": str(truth_data[idx].get('interruption_role', 'none')).lower(),
                    "gold_type": str(truth_data[idx].get('interruption_type', 'none')).lower(),
                    "pred_role": role,
                    "pred_type": itype,
                    "raw_output": raw_gen.strip()
                })
            
            out_path = os.path.join(model_folder, f"results_file_{i}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(file_results, f, indent=4, ensure_ascii=False)

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
### 6. EXECUTION ###
# ==========================================================
RAW_OUTPUT_DIR = "interruption_results_final"
if os.path.exists(RAW_OUTPUT_DIR): 
    shutil.rmtree(RAW_OUTPUT_DIR)
os.makedirs(RAW_OUTPUT_DIR)

print(f"🎯 Phase 1: Running 5 models on transformers {__import__('transformers').__version__}")
print(f"   (Llama-3.1, Llama-3.2, Qwen2.5-7B, Qwen2.5-1.5B, Mistral-7B)")
print(f"\n⚠️ InternLM-2.5-7B will run separately with transformers 4.41.2")
print(f"   Run this AFTER this script completes:")
print(f"   !pip install transformers==4.41.2 --quiet")
print(f"   Then run the InternLM-only cell below\n")

for name, config in MODEL_CONFIGS.items():
    try:
        run_model(name, config, RAW_OUTPUT_DIR)
    except Exception as e:
        print(f"❌ Failed running {name}: {e}")

shutil.make_archive("Final_ChatTemplate_Results_Phase1", 'zip', RAW_OUTPUT_DIR)
print("\n📦 Results packaged: Final_ChatTemplate_Results_Phase1.zip")

try:
    files.download("Final_ChatTemplate_Results_Phase1.zip")
    print("📥 Download initiated.")
except Exception as e:
    print(f"⚠️ Download failed: {e}")