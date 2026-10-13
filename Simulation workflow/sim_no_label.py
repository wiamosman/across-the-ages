import os
import json
import time
import requests
import re
from typing import Dict, List, Optional, Set

# --- UNIVERSITY NETWORK SETUP ---
os.environ['NO_PROXY'] = '*'
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

API_KEY = "XXX"
URL = "XXX"
MODEL_ID = "gpt-4o"

DATASET_PATH = "/Users/apple/StudioProjects/last_project_emnlp/labeled_data_original"
OUTPUT_ROOT = "/Users/apple/StudioProjects/last_project_emnlp/sim_no_label/results/sim_no_label_v1"

VALID_LABELS = {
    "intent": ["acknowledge", "advise", "agree", "apologise", "arrange", "ask for help", "asking for opinions", "care", "comfort", "complain", "confirm", "criticize", "doubt", "emphasize", "explain", "flaunt", "greet", "inform", "introduce", "invite", "joke", "leave", "offer help", "oppose", "plan", "praise", "prevent", "refuse", "taunt", "thank", "warn"],
    "dialogue_act": ["greeting", "question", "answer", "statement-opinion", "statement-non-opinion", "apology", "command", "agreement", "disagreement", "acknowledge", "backchannel", "others"],
    "emotion": ["angry", "happy", "sad", "neutral", "frustrated", "excited", "surprise", "fear", "joy", "disgust"],
    "generation_gap_indicator": ["slang mismatch", "perceived disrespect", "stereotyped expectation", "ignored input", "method divergence", "digital disparity", "none"],
    "interruption_role": ["interrupted speaker", "intervening speaker", "none"],
    "interruption_type": ["disruptive intervention", "cooperative assistance", "cooperative agreement", "none"]
}

VALID_LABELS_SETS: Dict[str, Set[str]] = {k: set(v) for k, v in VALID_LABELS.items()}

LABEL_KEYS = ["intent", "dialogue_act", "emotion", "interruption_role", "interruption_type", "generation_gap_indicator"]

# ============================================================================
# SIM-NO-LABEL — SPEAKER GENERATOR PROMPTS
# IDENTICAL to the facilitated generator EXCEPT: no facilitator, no S_t labels.
# This is the control arm: same generator, same protocol, no workflow.
# ============================================================================

GRANDPARENT_GENERATOR_PROMPT_NOLABEL = """You are the Grandparent having a conversation with your grandchild. You generate your next utterance naturally as you talk with them.

YOU SEE TWO HISTORIES:
1. ORIGINAL conversation — the real dialogue BEFORE your turn (for context and emotional tone)
2. CURRENT conversation — the simulated dialogue so far (PLAIN TEXT, no labels of any kind)

RULES:
- Read ORIGINAL to understand the emotional arc and how the conversation has been going
- Read CURRENT to see the simulated dialogue so far
- Generate YOUR response as the grandparent, naturally continuing the conversation.

CRITICAL: Output ONLY valid JSON:
{{"speaker": "grandfather", "text": "Your response here"}}"""

GRANDCHILD_GENERATOR_PROMPT_NOLABEL = """You are the Grandchild having a conversation with your grandparent. You generate your next utterance naturally as you talk with them.

YOU SEE TWO HISTORIES:
1. ORIGINAL conversation — the real dialogue BEFORE your turn (for context and emotional tone)
2. CURRENT conversation — the simulated dialogue so far (PLAIN TEXT, no labels of any kind)

RULES:
- Read ORIGINAL to understand the emotional arc and how the conversation has been going
- Read CURRENT to see the simulated dialogue so far
- Generate YOUR response as the grandchild, naturally continuing the conversation.

CRITICAL: Output ONLY valid JSON:
{{"speaker": "grandson", "text": "Your response here"}}"""


def validate_single_label(parsed_data: dict, label_key: str, allowed_set: Set[str], speaker_name: str) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(parsed_data, dict):
        return None, f"{label_key}=INVALID_JSON"
    found_key = next((k for k in parsed_data.keys() if k.lower() == label_key.lower()), None)
    if found_key:
        raw_val = str(parsed_data[found_key]).lower().strip()
        if raw_val in allowed_set:
            return raw_val, None
        else:
            return raw_val, f"{label_key}='{raw_val}'"
    else:
        return "none", f"{label_key}=MISSING"


def extract_json_from_text(raw_text: str) -> Optional[dict]:
    """Extract JSON from LLM response text. ROBUST version."""
    if not raw_text:
        return None
    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    if raw_text.startswith('```'):
        lines = raw_text.split('\n')
        if lines[0].strip().startswith('```'):
            lines = lines[1:]
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith('```'):
                end_idx = i
                break
        lines = lines[:end_idx]
        content = '\n'.join(lines).strip()
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                cleaned = re.sub(r',\s*}', '}', content)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                cleaned = cleaned.replace("'", '"')
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

    md_patterns = [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']
    for pattern in md_patterns:
        matches = re.findall(pattern, raw_text, re.DOTALL)
        for match in matches:
            try:
                cleaned = match.strip()
                cleaned = re.sub(r',\s*}', '}', cleaned)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                cleaned = cleaned.replace("'", '"')
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    for start_idx in range(len(raw_text)):
        if raw_text[start_idx] != '{':
            continue
        brace_count = 0
        for end_idx in range(start_idx, len(raw_text)):
            if raw_text[end_idx] == '{':
                brace_count += 1
            elif raw_text[end_idx] == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(raw_text[start_idx:end_idx+1])
                    except json.JSONDecodeError:
                        break
    return None


def call_api(system_prompt: str, user_message: str, max_retries: int = 3) -> Optional[str]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    session = requests.Session()
    session.trust_env = False
    temperatures = [0.0, 0.1, 0.0]

    for attempt in range(max_retries):
        temp = temperatures[min(attempt, len(temperatures)-1)]
        payload = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": temp,
            "max_tokens": 800
        }
        try:
            print(f"      [API] Attempt {attempt+1}/{max_retries} (temp={temp})...")
            response = session.post(URL, headers=headers, json=payload, timeout=60)
            if response.status_code != 200:
                print(f"      [API] HTTP {response.status_code}: {response.text[:200]}")
                time.sleep(2)
                continue
            res_json = response.json()
            if 'choices' not in res_json or not res_json['choices']:
                print(f"      [API] Empty choices in response")
                time.sleep(2)
                continue
            content = res_json['choices'][0]['message']['content']
            print(f"      [RAW] {content[:300]}...")
            return content
        except Exception as e:
            print(f"      [API] Error: {e}")
            time.sleep(2)

    print(f"      [FAIL] All {max_retries} attempts exhausted.")
    return None


# ============================================================================
# SPEAKER GENERATOR (NO-LABEL CONTROL ARM)
# Sees original history + PLAIN current history. No facilitator exists.
# ============================================================================

def run_speaker_generator_nolabel(speaker_name: str, original_history: List[dict], current_history: List[dict],
                                  topic: str, situation: str, original_turn_text: str) -> Optional[dict]:
    """
    No-label control: generate the turn WITHOUT any facilitator involvement.
    Current history is passed as PLAIN speaker/text turns (no S_t keys).
    """
    print(f"   [SPEAKER-NOLABEL] {speaker_name} generating response...")

    current_plain = []
    for turn in current_history:
        current_plain.append({
            "speaker": turn.get("speaker", "unknown"),
            "text": turn.get("text", "")
        })

    original_json = json.dumps(original_history, indent=2, ensure_ascii=False)
    current_json = json.dumps(current_plain, indent=2, ensure_ascii=False)

    if "grand" in speaker_name.lower() and ("father" in speaker_name.lower() or "mother" in speaker_name.lower()):
        system_prompt = GRANDPARENT_GENERATOR_PROMPT_NOLABEL
    else:
        system_prompt = GRANDCHILD_GENERATOR_PROMPT_NOLABEL

    user_message = f"""=== ORIGINAL CONVERSATION (the real dialogue BEFORE your turn) ===
{original_json}

=== CURRENT CONVERSATION (simulated dialogue so far, plain text, NO labels) ===
{current_json}

TOPIC: {topic}
SITUATION: {situation}

YOUR ORIGINAL TEXT IN THIS TURN: "{original_turn_text}"

Your turn as {speaker_name}.

INSTRUCTIONS:
1. Read the ORIGINAL conversation to understand the real emotional tone
2. Read the CURRENT conversation to see how the simulated dialogue has unfolded so far
3. Generate YOUR response as {speaker_name}, inspired by your original text and the natural flow of the conversation
4. Maintain your authentic voice

Output ONLY {{{{"speaker": "{speaker_name}", "text": "Your response"}}}}"""

    content = call_api(system_prompt, user_message)
    if not content:
        return None

    parsed = extract_json_from_text(content)
    if not parsed:
        return None

    text = str(parsed.get("text", "")).strip()
    if not text:
        return None

    print(f"   [SPEAKER-NOLABEL SUCCESS] Text: {text[:80]}...")
    return {"speaker": speaker_name, "text": text}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_topic_and_situation(utterances: List[dict], title: str) -> tuple[str, str]:
    topic = title if title else "a conversation between grandparent and grandchild"
    situation_parts = []
    for u in utterances[:3]:
        text = u.get("text", "")
        if text and len(text) > 10:
            situation_parts.append(text)
    situation = " ".join(situation_parts) if situation_parts else "grandparent and grandchild are talking"
    return topic, situation


# ============================================================================
# MAIN WORKFLOW — SIM-NO-LABEL CONTROL ARM
# ============================================================================

def run_sim_no_label_pipeline():
    """
    CONTROL ARM PIPELINE:
    1. For each turn:
       a. Speaker sees original history (BEFORE the turn) + plain current history -> generates response
       b. NO facilitator workflow is run. NO labels are attached.
    2. Repeat until conversation ends.
    Identical to the facilitated arm in every respect EXCEPT the absence of S_t.
    """
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    fallback_count = 0
    total_turns = 0

    for i in range(36, 105):
        file_name = f"transcript_{i}.json"
        file_path = os.path.join(DATASET_PATH, file_name)

        if not os.path.exists(file_path):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        original_utterances = data.get("utterances", [])
        title = data.get("title", "")
        topic, situation = extract_topic_and_situation(original_utterances, title)

        new_utterances = []

        print(f"\n=== SIM-NO-LABEL PIPELINE: {file_name} ===")
        print(f"Topic: {topic}")
        print(f"Total turns: {len(original_utterances)}")
        print("=" * 60)

        for turn_idx, turn in enumerate(original_utterances):
            original_speaker = turn.get("speaker", "Unknown")
            original_text = turn.get("text", "")
            spk_lower = original_speaker.lower()
            total_turns += 1

            is_gp = "grand" in spk_lower and ("father" in spk_lower or "mother" in spk_lower)
            is_gc = "grand" in spk_lower and ("son" in spk_lower or "daughter" in spk_lower or "child" in spk_lower)

            print(f"\n--- Turn {turn_idx+1}/{len(original_utterances)} | Speaker: {original_speaker} ---")

            if is_gp or is_gc:
                # NOTE: [:turn_idx] = history BEFORE the turn. If your facilitated arm
                # uses [:turn_idx+1], change this to match EXACTLY (arms must differ
                # only in the presence of labels).
                ai_response = run_speaker_generator_nolabel(
                    speaker_name=original_speaker,
                    original_history=original_utterances[:turn_idx],
                    current_history=new_utterances[-3:],
                    topic=topic,
                    situation=situation,
                    original_turn_text=original_text
                )

                if ai_response:
                    new_utterances.append(ai_response)
                    print(f"   [SUCCESS] Text: {ai_response.get('text', '')[:80]}...")
                else:
                    fallback_count += 1
                    turn_copy = {"speaker": original_speaker, "text": original_text}
                    new_utterances.append(turn_copy)
                    print(f"   [FALLBACK] Kept original text (no labels in this arm).")
            else:
                print(f"   [KEEP] Non-dialogue speaker: {original_speaker}")
                new_utterances.append(turn)

        data["utterances"] = new_utterances
        output_path = os.path.join(OUTPUT_ROOT, f"sim_no_label_{file_name}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"DONE! Output saved to: {output_path}")
        print(f"{'='*60}")

    print(f"\n[FALLBACK SUMMARY] {fallback_count}/{total_turns} turns fell back to original text.")


if __name__ == "__main__":
    run_sim_no_label_pipeline()
