import torch
import json
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import gc
import shutil
import traceback
from google.colab import files, userdata
from huggingface_hub import login

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
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

# ==========================================
# 2. GGI PROMPT & VALID LABELS
# ==========================================
VALID_LABELS = ['slang mismatch', 'perceived disrespect', 'stereotyped expectation', 'ignored input', 'method divergence', 'digital disparity', 'none']

SYSTEM_PROMPT = f"""You are an expert in intergenerational linguistics. Analyze the conversational turn between a grandparent and grandchild and detect Generational Gap Indicators.

CRITICAL INSTRUCTIONS:
1. Compare the TARGET turn against the PREVIOUS CONTEXT.
2. Focus on "Social Friction": A gap only exists if there is tension, confusion, or a breakdown in mutual understanding.
3. If the speakers are in sync—even if they use different words—label it 'none'.

DECISION HIERARCHY (apply in order):
1. Is there NO tension, confusion, or emotional friction? → 'none'
2. Is the conflict about LANGUAGE/VOCABULARY causing a stall? → 'slang mismatch'
3. Is the conflict about DIGITAL TECHNOLOGY skills? → 'digital disparity'
4. Is one person DISMISSING or shutting down the other's contribution? → 'ignored input'
5. Is one person ATTACKING the other's dignity/status/personhood? → 'perceived disrespect'
6. Is one person using a GENERATIONAL STEREOTYPE or label? → 'stereotyped expectation'
7. Is the conflict about the PROCEDURE/METHOD for a task? → 'method divergence'
If multiple apply, pick the MOST SPECIFIC one (lower number = more specific).

LABEL DEFINITIONS:
- slang mismatch: The "Language Barrier." Triggered when age-specific vocabulary (e.g., "bussin," "groovy") causes a literal stall in the conversation. Look for clarification requests ("What does that mean?"), confusion, or the conversation halting because of unknown words.
- perceived disrespect: The "Status Clash." Triggered by a violation of social norms or "face-threats." Look for behavior that undermines the other's dignity, leads to emotional withdrawal, or causes a defensive counter-attack. This attacks the PERSON, not their generation.
- stereotyped expectation: The "Generalization Trap." Triggered when a speaker interacts with a LABEL ("lazy youth," "stubborn elderly") rather than the individual. Must result in defensiveness or a misaligned response. This attacks the GENERATION, not the person.
- ignored input: The "Dismissive Gap." Triggered when one speaker explicitly disregards advice, contribution, or presence, or gives harsh imperative orders that cause visible dissatisfaction in the other. The other person feels unheard or restricted.
- method divergence: The "Process War." A conflict over the 'right' way to do a non-digital task (e.g., cooking, cleaning). Both agree on the goal but argue over the procedure with expressed tension or friction. Simply describing a different method is NOT enough.
- digital disparity: The "Tech Anxiety." Friction specifically regarding digital skills. Look for impatience from the tech-fluent person or anxiety, frustration, and a sense of "being left behind" from the hesitant person.
- none: The "Cooperative Flow." Used for neutral, playful, or informational turns. Even if there is a difference in age, if there is no social friction or emotional resistance, it is 'none'.

Your response must ONLY contain the label name from this list: [{', '.join(VALID_LABELS)}]."""

FEW_SHOT_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    
    # none: Cooperative, no friction
    {"role": "user", "content": "Context: None (Beginning of conversation)\nTarget: Speaker A (grandson): 'Hey Grandma, are you ready to start the video?'"},
    {"role": "assistant", "content": "none"},
    
    {"role": "user", "content": "Context: Speaker A (grandson): 'I finished my homework.'\nTarget: Speaker B (grandmother): 'Good job! Let's have some cookies.'"},
    {"role": "assistant", "content": "none"},
    
    # slang mismatch: Vocabulary confusion, conversation stalls
    {"role": "user", "content": "Context: Speaker A (grandson): 'That stream was totally bussin'.'\nTarget: Speaker B (grandmother): 'What is a bussin? Is that a type of vehicle?'"},
    {"role": "assistant", "content": "slang mismatch"},
    
    {"role": "user", "content": "Context: Speaker A (grandson): 'The fit was absolutely fire, no cap.'\nTarget: Speaker B (grandmother): 'Fire? Cap? Are you talking about hats and arson? I don't understand.'"},
    {"role": "assistant", "content": "slang mismatch"},
    
    # stereotyped expectation: Attacks the GENERATION as a group
    {"role": "user", "content": "Context: Speaker A (grandson): 'I'm a bit tired today.'\nTarget: Speaker B (grandmother): 'Typical. Your generation has no stamina because you never go outside.'"},
    {"role": "assistant", "content": "stereotyped expectation"},
    
    {"role": "user", "content": "Context: Speaker A (grandmother): 'I need help carrying these bags.'\nTarget: Speaker B (grandson): 'Ugh, old people always complain about everything. You boomers are so weak.'"},
    {"role": "assistant", "content": "stereotyped expectation"},
    
    # perceived disrespect: Attacks the PERSON's dignity/status
    {"role": "user", "content": "Context: Speaker A (grandmother): 'Let me show you how to knead the dough properly.'\nTarget: Speaker B (grandson): 'I don't care how you did it in the stone age, just let me do it my way.'"},
    {"role": "assistant", "content": "perceived disrespect"},
    
    {"role": "user", "content": "Context: Speaker A (grandson): 'I think we should take the bus.'\nTarget: Speaker B (grandmother): 'You are so stupid. Stop talking and let the adults decide.'"},
    {"role": "assistant", "content": "perceived disrespect"},
    
    # ignored input: Dismissing contribution, shutting down
    {"role": "user", "content": "Context: Speaker A (grandson): 'Maybe we could add some garlic to the sauce?'\nTarget: Speaker B (grandmother): 'Be quiet. I don't need your opinion. Just watch and learn.'"},
    {"role": "assistant", "content": "ignored input"},
    
    {"role": "user", "content": "Context: Speaker A (grandmother): 'I saved all the family photos in this album for you.'\nTarget: Speaker B (grandson): 'Whatever. I don't care about your old pictures. Put them away.'"},
    {"role": "assistant", "content": "ignored input"},
    
    # method divergence: Conflict over HOW to do a task
    {"role": "user", "content": "Context: Speaker A (grandson): 'I just use the microwave to melt the butter.'\nTarget: Speaker B (grandmother): 'That is a lazy, terrible way to bake. It ruins the consistency! You must use the stovetop.'"},
    {"role": "assistant", "content": "method divergence"},
    
    {"role": "user", "content": "Context: Speaker A (grandmother): 'We should wash the dishes by hand to make sure they're clean.'\nTarget: Speaker B (grandson): 'No, the dishwasher does it better. Your way wastes water and takes forever!'"},
    {"role": "assistant", "content": "method divergence"},
    
    # digital disparity: Tech-related friction
    {"role": "user", "content": "Context: Speaker A (grandson): 'Just click the link in the bio.'\nTarget: Speaker B (grandmother): 'I've been clicking the screen for ten minutes and nothing is happening! This machine is making me cry.'"},
    {"role": "assistant", "content": "digital disparity"},
    
    {"role": "user", "content": "Context: Speaker A (grandmother): 'How do I send this photo to your mother?'\nTarget: Speaker B (grandson): 'Ugh, I already showed you five times! Why can't you remember anything? Just give me the phone.'"},
    {"role": "assistant", "content": "digital disparity"}
]

# ==========================================
# 3. MODEL CONFIGURATIONS
# ==========================================
MODEL_CONFIGS = {
    "Llama-3.1-8B": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "trust_remote_code": False,
        "supports_system": True,
        "max_new_tokens": 20,
        "repetition_penalty": 1.05,
        "extra_stop_token_ids": [128009],
    },
    "Llama-3.2-3B": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "trust_remote_code": False,
        "supports_system": True,
        "max_new_tokens": 20,
        "repetition_penalty": 1.1,
        "extra_stop_token_ids": [128009],
    },
    "Qwen2.5-7B": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "trust_remote_code": False,
        "supports_system": True,
        "max_new_tokens": 20,
        "repetition_penalty": 1.05,
        "extra_stop_token_ids": [151645, 151643],
    },
    "Qwen2.5-1.5B": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "trust_remote_code": False,
        "supports_system": True,
        "max_new_tokens": 20,
        "repetition_penalty": 1.1,
        "extra_stop_token_ids": [151645, 151643],
    },
    "Mistral-7B": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "trust_remote_code": False,
        "supports_system": False,
        "max_new_tokens": 20,
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

# ==========================================
# 4. UTILITIES
# ==========================================
def get_file_indices():
    indices = []
    for i in range(36, 105):
        if os.path.exists(f"/content/transcript_{i}.json"):
            indices.append(i)
    return indices

def clean_label(raw_ans: str) -> str:
    raw_ans = raw_ans.lower().strip()
    for l in VALID_LABELS:
        if l in raw_ans:
            return l
    return "none"

def build_ggi_messages(context_str, target_str, supports_system):
    """Build messages with model-specific system-role handling."""
    user_content = f"Context: {context_str}\nTarget: {target_str}"
    
    if supports_system:
        return FEW_SHOT_MESSAGES + [{"role": "user", "content": user_content}]
    
    # Mistral: no system role — prepend system text to first user message
    messages = []
    for msg in FEW_SHOT_MESSAGES:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "user" and not messages:
            messages.append({
                "role": "user",
                "content": f"{SYSTEM_PROMPT}\n\n{msg['content']}"
            })
        else:
            messages.append(msg)
    messages.append({"role": "user", "content": user_content})
    return messages

# ==========================================
# 5. INFERENCE ENGINE
# ==========================================
def run_classification_benchmark(model_name, config, file_indices, base_output_dir):
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

        model = AutoModelForCausalLM.from_pretrained(
            config["hf_id"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=config["trust_remote_code"],
            low_cpu_mem_usage=True
        )
        model.eval()
        print(f"   Model loaded on {next(model.parameters()).device}")

        for idx in file_indices:
            truth_path = f"/content/transcript_{idx}.json"
            with open(truth_path, 'r') as f:
                truth_data = json.load(f).get('utterances', [])

            slim_results = {
                "model_used": model_name,
                "hf_id": config["hf_id"],
                "file_index": idx,
                "utterances": []
            }

            for i in tqdm(range(len(truth_data)), desc=f"{model_name} - File {idx}"):
                u_target = truth_data[i]
                
                if i == 0:
                    context_str = "None (Beginning of conversation)"
                elif i == 1:
                    u_p1 = truth_data[0]
                    context_str = f"Speaker A ({u_p1.get('speaker')}): {u_p1.get('text')}"
                else:
                    u_p1 = truth_data[i-2]
                    u_p2 = truth_data[i-1]
                    context_str = f"Speaker A ({u_p1.get('speaker')}): {u_p1.get('text')} | Speaker B ({u_p2.get('speaker')}): {u_p2.get('text')}"
                
                target_str = f"Target ({u_target.get('speaker')}): {u_target.get('text')}"
                
                messages = build_ggi_messages(context_str, target_str, config["supports_system"])

                try:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    print(f"\n⚠️ Chat template failed: {e}")
                    prompt = f"{SYSTEM_PROMPT}\n\nContext: {context_str}\nTarget: {target_str}\nLabel:"

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
                    output = model.generate(**inputs, **gen_kwargs)

                generated_ids = output[0][inputs.input_ids.shape[-1]:]
                raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
                predicted = clean_label(raw_answer)

                gk = next((k for k in u_target.keys() if 'gap' in k.lower()), None)
                gold_label = str(u_target.get(gk, 'none')).lower().strip() if gk else 'none'

                slim_results["utterances"].append({
                    "turn_index": i,
                    "context": context_str,
                    "target": target_str,
                    "gold_label": gold_label,
                    "predicted_label": predicted,
                    "raw": raw_answer.strip()
                })

            out_path = os.path.join(model_path, f"results_file_{idx}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(slim_results, f, indent=4, ensure_ascii=False)

            print(f"   ✓ Saved: {out_path}")

        print(f"✅ Completed: {model_name}")

    except Exception as e:
        print(f"\n❌ ERROR with {model_name}: {e}")
        traceback.print_exc()
    finally:
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"   🧹 Cleaned up GPU memory")

# ==========================================
# 6. EXECUTION
# ==========================================
RESULT_DIR = "prediction_ggi"
if os.path.exists(RESULT_DIR):
    shutil.rmtree(RESULT_DIR)
os.makedirs(RESULT_DIR)

file_indices = get_file_indices()

if file_indices:
    print(f"📂 Found {len(file_indices)} transcript files.")
    print(f"🎯 Phase 1: Running 5 models on transformers {__import__('transformers').__version__}")
    print(f"   (Llama-3.1, Llama-3.2, Qwen2.5-7B, Qwen2.5-1.5B, Mistral-7B)")
    print(f"\n⚠️ InternLM-2.5-7B will run separately with transformers 4.41.2")
    print(f"   Run this AFTER this script completes:")
    print(f"   !pip install transformers==4.41.2 --quiet")
    print(f"   Then RESTART runtime and run the InternLM-only cell below\n")

    for name, config in MODEL_CONFIGS.items():
        run_classification_benchmark(name, config, file_indices, RESULT_DIR)

    shutil.make_archive("GGI_Final_Results_Phase1", 'zip', RESULT_DIR)
    print("\n📦 Results packaged: GGI_Final_Results_Phase1.zip")

    try:
        files.download("GGI_Final_Results_Phase1.zip")
        print("📥 Download initiated.")
    except Exception as e:
        print(f"⚠️ Download failed: {e}")
else:
    print("❌ No transcript files found.")