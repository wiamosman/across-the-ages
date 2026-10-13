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
BASE_OUTPUT_ROOT = "/Users/apple/StudioProjects/last_project_emnlp/abliation_study"

# ============================================================================
# 5 ABLATION VARIANTS — each removes exactly one agent
# ============================================================================
ABLATION_VARIANTS = [
    {
        "name": "no_intent",
        "config": {
            "use_intent": False,
            "use_dialogue_act": True,
            "use_emotion": True,
            "use_interruption": True,
            "use_gap": True
        }
    },
    {
        "name": "no_emotion",
        "config": {
            "use_intent": True,
            "use_dialogue_act": True,
            "use_emotion": False,
            "use_interruption": True,
            "use_gap": True
        }
    },
    {
        "name": "no_dialogue_act",
        "config": {
            "use_intent": True,
            "use_dialogue_act": False,
            "use_emotion": True,
            "use_interruption": True,
            "use_gap": True
        }
    },
    {
        "name": "no_interruption",
        "config": {
            "use_intent": True,
            "use_dialogue_act": True,
            "use_emotion": True,
            "use_interruption": False,
            "use_gap": True
        }
    },
    {
        "name": "no_generation_gap",
        "config": {
            "use_intent": True,
            "use_dialogue_act": True,
            "use_emotion": True,
            "use_interruption": True,
            "use_gap": False
        }
    }
]

VALID_LABELS = {
    "intent": ["acknowledge", "advise", "agree", "apologise", "arrange", "ask for help",
               "asking for opinions", "care", "comfort", "complain", "confirm", "criticize",
               "doubt", "emphasize", "explain", "flaunt", "greet", "inform", "introduce",
               "invite", "joke", "leave", "offer help", "oppose", "plan", "praise",
               "prevent", "refuse", "taunt", "thank", "warn"],
    "dialogue_act": ["greeting", "question", "answer", "statement-opinion",
                     "statement-non-opinion", "apology", "command", "agreement",
                     "disagreement", "acknowledge", "backchannel", "others"],
    "emotion": ["angry", "happy", "sad", "neutral", "frustrated", "excited",
                "surprise", "fear", "joy", "disgust"],
    "generation_gap_indicator": ["slang mismatch", "perceived disrespect",
                                 "stereotyped expectation", "ignored input",
                                 "method divergence", "digital disparity", "none"],
    "interruption_role": ["interrupted speaker", "intervening speaker", "none"],
    "interruption_type": ["disruptive intervention", "cooperative assistance",
                          "cooperative agreement", "none"]
}

VALID_LABELS_SETS: Dict[str, Set[str]] = {k: set(v) for k, v in VALID_LABELS.items()}


def get_active_label_keys(config: dict) -> List[str]:
    """Return only the label keys that are enabled in the ablation config."""
    keys = []
    if config.get("use_intent", True):
        keys.append("intent")
    if config.get("use_dialogue_act", True):
        keys.append("dialogue_act")
    if config.get("use_emotion", True):
        keys.append("emotion")
    if config.get("use_interruption", True):
        keys.extend(["interruption_role", "interruption_type"])
    if config.get("use_gap", True):
        keys.append("generation_gap_indicator")
    return keys


# ============================================================================
# FACILITATOR PROMPTS
# ============================================================================

INTENT_AGENT_PROMPT = """You are the Intent Analysis Agent (A_int). 
Identify the speaker's underlying communicative goal.

Valid intents: {intent_list}

Output ONLY valid JSON:
{{"intent": "<chosen_intent>"}}"""

DIALOGUE_ACT_AGENT_PROMPT = """You are the Dialogue Act Analysis Agent (A_act).
Analyze the functional role of this utterance in the conversation.

Valid dialogue acts: {dialogue_act_list}

Output ONLY valid JSON:
{{"dialogue_act": "<chosen_act>"}}"""

EMOTION_AGENT_PROMPT = """You are the Emotion Analysis Agent (A_emo).
Infer the speaker's affective state.

Valid emotions: {emotion_list}

Output ONLY valid JSON:
{{"emotion": "<chosen_emotion>"}}"""

INTERRUPTION_AGENT_PROMPT = """You are the Interruption Analysis Agent (A_intp).
You analyze turn-taking behavior and conversational flow to detect interruptions.

You are given:
- The conversation history (last 3 turns, with their labels)
- The current utterance
- The pre-computed lower-level labels for the current utterance: Intent, Dialogue Act, and Emotion

Your task is to determine if an interruption occurred. To do this, examine the conversation history and current utterance, and consider the following:

Turn-taking and flow:
- Did the current speaker start before the previous speaker finished?
- Is there speaker overlap, a backchannel, or a floor-grab?
- Is the speaker finishing the other's sentence, agreeing enthusiastically, or cutting in to oppose/change topic?
- Does the flow show one speaker being cut off or dominated?

Use the pre-computed labels as signals to help understand the interruption:
- Consider the intent: does it suggest the speaker was cut off or cut in?
- Consider the dialogue act: does it indicate a backchannel, overlap, or floor-grab?
- Consider the emotion: does it suggest urgency, disagreement, or eagerness to interject?

Ground your final decision primarily in the conversational flow and turn-taking patterns, using the labels as supporting signals.

Valid interruption roles: {interruption_role_list}
Valid interruption types: {interruption_type_list}

Definitions:
- interrupted speaker: this speaker was cut off by the other
- intervening speaker: this speaker cut in while the other was talking
- none: normal turn transition

- disruptive intervention: cutting in to change topic, oppose, or steal the floor
- cooperative assistance: finishing the other's sentence or helping them express
- cooperative agreement: brief overlap to agree or support ("Exactly!", "Right")
- none: no interruption, or not an intervening speaker

Output ONLY valid JSON:
{{"interruption_role": "<role>", "interruption_type": "<type>"}}"""

GENERATION_GAP_AGENT_PROMPT = """You are the Generation Gap Analysis Agent (A_gap).
You detect intergenerational tensions by analyzing the conversation content and flow.

You are given:
- The conversation history (last 3 turns, with their labels)
- The current utterance
- The pre-computed lower-level labels for the current utterance: Intent, Dialogue Act, and Emotion

Your task is to detect intergenerational tension. To do this, examine the conversation history and current utterance, and consider the following:

Conversation content and flow:
- Is there confusion or frustration about slang, expressions, or language used by the other generation?
- Is there tension about technology, digital skills, or modern vs. traditional methods?
- Does one speaker dismiss or shut down the other's contribution?
- Does one speaker feel disrespected, attacked, or undermined?
- Does one speaker generalize about the other generation ("you young people", "old people", etc.)?
- Is there a clash over procedures, traditions, or how to do something?
- Does the conversation show recurring miscommunication patterns between grandparent and grandchild?

Use the pre-computed labels as signals to help understand the generation gap:
- Consider the intent: does it suggest misunderstanding, dismissal, or stereotyping across generations?
- Consider the dialogue act: does it show conflict, command, or refusal?
- Consider the emotion: does it signal frustration, anger, or disrespect tied to generational differences?

Ground your final decision primarily in the conversational content and flow, using the labels as supporting signals.

Valid indicators: {generation_gap_list}

Definitions:
- slang mismatch: confusion or frustration about language, slang, or expressions used by the other generation
- digital disparity: tension related to technology, digital skills, or modern vs. traditional methods
- ignored input: one speaker dismisses or shuts down the other's contribution
- perceived disrespect: one speaker feels attacked, insulted, or their dignity undermined
- stereotyped expectation: one speaker generalizes about the other generation ("you young people", "old people", etc.)
- method divergence: clash over how to do something (procedures, traditions, modern approaches)
- none: no intergenerational tension detected

Output ONLY valid JSON:
{{"generation_gap_indicator": "<indicator>"}}"""

# ============================================================================
# SPEAKER GENERATOR PROMPTS
# ============================================================================

GRANDPARENT_GENERATOR_PROMPT = """You are the Grandparent having a conversation with your grandchild. You generate your next utterance naturally as you talk with them.

YOU SEE TWO HISTORIES:
1. ORIGINAL conversation — the real dialogue up to your turn (for context and emotional tone)
2. CURRENT conversation — simulated dialogue so far, WITH facilitator labels (S_t) on each turn

RULES:
- Read ORIGINAL to understand what you really said and the emotional arc
- Read CURRENT (with S_t labels showing available intent, emotion, gaps, interruptions) to see the current flow
- Generate YOUR response as grandparent.

CRITICAL: Output ONLY valid JSON:
{{"speaker": "grandfather", "text": "Your response here"}}"""

GRANDCHILD_GENERATOR_PROMPT = """You are the Grandchild having a conversation with your grandparent. You generate your next utterance naturally as you talk with them.

YOU SEE TWO HISTORIES:
1. ORIGINAL conversation — the real dialogue up to your turn (for context and emotional tone)
2. CURRENT conversation — simulated dialogue so far, WITH facilitator labels (S_t) on each turn

RULES:
- Read ORIGINAL to understand what you really said and the emotional arc
- Read CURRENT (with S_t labels showing available intent, emotion, gaps, interruptions) to see the current flow
- Use S_t labels to inform your response.
- Generate YOUR response as grandchild.

CRITICAL: Output ONLY valid JSON:
{{"speaker": "grandson", "text": "Your response here"}}"""


def validate_single_label(parsed_data: dict, label_key: str, allowed_set: Set[str],
                          speaker_name: str) -> tuple[Optional[str], Optional[str]]:
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
# FACILITATOR — 5 ANALYTIC AGENTS
# ============================================================================

def run_intent_agent(utterance: dict, history: List[dict]) -> tuple[Optional[str], list[str]]:
    system_prompt = INTENT_AGENT_PROMPT.format(intent_list='", "'.join(VALID_LABELS['intent']))
    history_json = json.dumps(history[-3:], indent=2, ensure_ascii=False)
    utterance_json = json.dumps(utterance, indent=2, ensure_ascii=False)
    user_message = f"""CONVERSATION HISTORY (last {len(history[-3:])} turns):\n{history_json}\n\nANALYZE THIS UTTERANCE:\n{utterance_json}\n\nOutput ONLY the JSON with the intent label."""

    content = call_api(system_prompt, user_message)
    if not content:
        return None, ["intent=API_FAIL"]
    parsed = extract_json_from_text(content)
    if not parsed:
        return None, ["intent=PARSE_FAIL"]
    label, error = validate_single_label(parsed, "intent", VALID_LABELS_SETS["intent"],
                                         utterance.get("speaker", "unknown"))
    errors = [error] if error else []
    if not errors:
        print(f"      [A_int SUCCESS] intent={label}")
    else:
        print(f"      [A_int INVALID] {', '.join(errors)}")
    return label, errors


def run_dialogue_act_agent(utterance: dict, history: List[dict]) -> tuple[Optional[str], list[str]]:
    system_prompt = DIALOGUE_ACT_AGENT_PROMPT.format(dialogue_act_list='", "'.join(VALID_LABELS['dialogue_act']))
    history_json = json.dumps(history[-3:], indent=2, ensure_ascii=False)
    utterance_json = json.dumps(utterance, indent=2, ensure_ascii=False)
    user_message = f"""CONVERSATION HISTORY (last {len(history[-3:])} turns):\n{history_json}\n\nANALYZE THIS UTTERANCE:\n{utterance_json}\n\nOutput ONLY the JSON with the dialogue_act label."""

    content = call_api(system_prompt, user_message)
    if not content:
        return None, ["dialogue_act=API_FAIL"]
    parsed = extract_json_from_text(content)
    if not parsed:
        return None, ["dialogue_act=PARSE_FAIL"]
    label, error = validate_single_label(parsed, "dialogue_act", VALID_LABELS_SETS["dialogue_act"],
                                         utterance.get("speaker", "unknown"))
    errors = [error] if error else []
    if not errors:
        print(f"      [A_act SUCCESS] dialogue_act={label}")
    else:
        print(f"      [A_act INVALID] {', '.join(errors)}")
    return label, errors


def run_emotion_agent(utterance: dict, history: List[dict]) -> tuple[Optional[str], list[str]]:
    system_prompt = EMOTION_AGENT_PROMPT.format(emotion_list='", "'.join(VALID_LABELS['emotion']))
    history_json = json.dumps(history[-3:], indent=2, ensure_ascii=False)
    utterance_json = json.dumps(utterance, indent=2, ensure_ascii=False)
    user_message = f"""CONVERSATION HISTORY (last {len(history[-3:])} turns):\n{history_json}\n\nANALYZE THIS UTTERANCE:\n{utterance_json}\n\nOutput ONLY the JSON with the emotion label."""

    content = call_api(system_prompt, user_message)
    if not content:
        return None, ["emotion=API_FAIL"]
    parsed = extract_json_from_text(content)
    if not parsed:
        return None, ["emotion=PARSE_FAIL"]
    label, error = validate_single_label(parsed, "emotion", VALID_LABELS_SETS["emotion"],
                                         utterance.get("speaker", "unknown"))
    errors = [error] if error else []
    if not errors:
        print(f"      [A_emo SUCCESS] emotion={label}")
    else:
        print(f"      [A_emo INVALID] {', '.join(errors)}")
    return label, errors


def run_interruption_agent(utterance: dict, history: List[dict],
                           intent: str, dialogue_act: str, emotion: str,
                           ablation_config: dict) -> tuple[Optional[dict], list[str]]:
    system_prompt = INTERRUPTION_AGENT_PROMPT.format(
        interruption_role_list='", "'.join(VALID_LABELS['interruption_role']),
        interruption_type_list='", "'.join(VALID_LABELS['interruption_type'])
    )

    active_keys = get_active_label_keys(ablation_config)

    history_with_labels = []
    for turn in history[-3:]:
        turn_copy = {
            "speaker": turn.get("speaker", "unknown"),
            "text": turn.get("text", ""),
            "labels": {}
        }
        for k in active_keys:
            if k in turn:
                turn_copy["labels"][k] = turn[k]
        history_with_labels.append(turn_copy)

    history_json = json.dumps(history_with_labels, indent=2, ensure_ascii=False)
    utterance_json = json.dumps(utterance, indent=2, ensure_ascii=False)

    intent_disp = intent if ablation_config.get("use_intent", True) else "N/A (ablated)"
    dialogue_act_disp = dialogue_act if ablation_config.get("use_dialogue_act", True) else "N/A (ablated)"
    emotion_disp = emotion if ablation_config.get("use_emotion", True) else "N/A (ablated)"

    user_message = f"""CONVERSATION HISTORY (last {len(history[-3:])} turns):
{history_json}

CURRENT UTTERANCE TO ANALYZE:
{utterance_json}

PRE-COMPUTED LABELS FOR THIS UTTERANCE:
- Intent: {intent_disp}
- Dialogue Act: {dialogue_act_disp}
- Emotion: {emotion_disp}

Output ONLY valid JSON with interruption_role and interruption_type."""

    content = call_api(system_prompt, user_message)
    if not content:
        return None, ["interruption=API_FAIL"]
    parsed = extract_json_from_text(content)
    if not parsed:
        return None, ["interruption=PARSE_FAIL"]

    errors = []
    role, role_err = validate_single_label(parsed, "interruption_role",
                                           VALID_LABELS_SETS["interruption_role"],
                                           utterance.get("speaker", "unknown"))
    type_label, type_err = validate_single_label(parsed, "interruption_type",
                                                 VALID_LABELS_SETS["interruption_type"],
                                                 utterance.get("speaker", "unknown"))
    if role_err:
        errors.append(role_err)
    if type_err:
        errors.append(type_err)

    result = {
        "interruption_role": role if role else "none",
        "interruption_type": type_label if type_label else "none"
    }
    if not errors:
        print(f"      [A_intp SUCCESS] role={role}, type={type_label}")
    else:
        print(f"      [A_intp INVALID] {', '.join(errors)}")
    return result, errors


def run_generation_gap_agent(utterance: dict, history: List[dict],
                             intent: str, dialogue_act: str, emotion: str,
                             ablation_config: dict) -> tuple[Optional[str], list[str]]:
    system_prompt = GENERATION_GAP_AGENT_PROMPT.format(
        generation_gap_list='", "'.join(VALID_LABELS['generation_gap_indicator'])
    )

    active_keys = get_active_label_keys(ablation_config)

    history_with_labels = []
    for turn in history[-3:]:
        turn_copy = {
            "speaker": turn.get("speaker", "unknown"),
            "text": turn.get("text", ""),
            "labels": {}
        }
        for k in active_keys:
            if k in turn:
                turn_copy["labels"][k] = turn[k]
        history_with_labels.append(turn_copy)

    history_json = json.dumps(history_with_labels, indent=2, ensure_ascii=False)
    utterance_json = json.dumps(utterance, indent=2, ensure_ascii=False)

    intent_disp = intent if ablation_config.get("use_intent", True) else "N/A (ablated)"
    dialogue_act_disp = dialogue_act if ablation_config.get("use_dialogue_act", True) else "N/A (ablated)"
    emotion_disp = emotion if ablation_config.get("use_emotion", True) else "N/A (ablated)"

    user_message = f"""CONVERSATION HISTORY (last {len(history[-3:])} turns):
{history_json}

CURRENT UTTERANCE TO ANALYZE:
{utterance_json}

PRE-COMPUTED LABELS FOR THIS UTTERANCE:
- Intent: {intent_disp}
- Dialogue Act: {dialogue_act_disp}
- Emotion: {emotion_disp}

Output ONLY valid JSON with generation_gap_indicator."""

    content = call_api(system_prompt, user_message)
    if not content:
        return None, ["generation_gap=API_FAIL"]
    parsed = extract_json_from_text(content)
    if not parsed:
        return None, ["generation_gap=PARSE_FAIL"]
    label, error = validate_single_label(parsed, "generation_gap_indicator",
                                         VALID_LABELS_SETS["generation_gap_indicator"],
                                         utterance.get("speaker", "unknown"))
    errors = [error] if error else []
    if not errors:
        print(f"      [A_gap SUCCESS] generation_gap={label}")
    else:
        print(f"      [A_gap INVALID] {', '.join(errors)}")
    return label, errors


# ============================================================================
# FACILITATOR — Aggregates agents into S_t (CASCADING with ablation)
# ============================================================================

def run_facilitator(utterance: dict, current_history: List[dict],
                    ablation_config: dict) -> Optional[dict]:
    """
    Facilitator workflow. Agents marked False in ablation_config are SKIPPED.
    """
    print(f"   [FACILITATOR] Analyzing {utterance.get('speaker', '?')}'s utterance...")
    speaker = utterance.get("speaker", "unknown")
    all_errors = []

    intent = None
    dialogue_act = None
    emotion = None

    if ablation_config.get("use_intent", True):
        intent, intent_errs = run_intent_agent(utterance, current_history)
        all_errors.extend(intent_errs)
    else:
        print("      [A_int ABLATED] Skipping intent analysis.")

    if ablation_config.get("use_dialogue_act", True):
        dialogue_act, act_errs = run_dialogue_act_agent(utterance, current_history)
        all_errors.extend(act_errs)
    else:
        print("      [A_act ABLATED] Skipping dialogue act analysis.")

    if ablation_config.get("use_emotion", True):
        emotion, emotion_errs = run_emotion_agent(utterance, current_history)
        all_errors.extend(emotion_errs)
    else:
        print("      [A_emo ABLATED] Skipping emotion analysis.")

    intent_for_cascade = intent if intent else "none"
    dialogue_act_for_cascade = dialogue_act if dialogue_act else "none"
    emotion_for_cascade = emotion if emotion else "none"

    interruption = None
    if ablation_config.get("use_interruption", True):
        interruption, intp_errs = run_interruption_agent(
            utterance, current_history,
            intent=intent_for_cascade,
            dialogue_act=dialogue_act_for_cascade,
            emotion=emotion_for_cascade,
            ablation_config=ablation_config
        )
        all_errors.extend(intp_errs)
    else:
        print("      [A_intp ABLATED] Skipping interruption analysis.")

    gap = None
    if ablation_config.get("use_gap", True):
        gap, gap_errs = run_generation_gap_agent(
            utterance, current_history,
            intent=intent_for_cascade,
            dialogue_act=dialogue_act_for_cascade,
            emotion=emotion_for_cascade,
            ablation_config=ablation_config
        )
        all_errors.extend(gap_errs)
    else:
        print("      [A_gap ABLATED] Skipping generation gap analysis.")

    # Build S_t with ONLY enabled labels
    S_t = {"speaker": speaker}
    if ablation_config.get("use_intent", True):
        S_t["intent"] = intent if intent else "none"
    if ablation_config.get("use_dialogue_act", True):
        S_t["dialogue_act"] = dialogue_act if dialogue_act else "none"
    if ablation_config.get("use_emotion", True):
        S_t["emotion"] = emotion if emotion else "none"
    if ablation_config.get("use_interruption", True):
        S_t["interruption_role"] = interruption["interruption_role"] if interruption else "none"
        S_t["interruption_type"] = interruption["interruption_type"] if interruption else "none"
    if ablation_config.get("use_gap", True):
        S_t["generation_gap_indicator"] = gap if gap else "none"

    if not all_errors:
        print(f"   [FACILITATOR SUCCESS] Labels: {json.dumps(S_t, ensure_ascii=False)}")
    else:
        print(f"   [FACILITATOR WARNINGS] {len(all_errors)} issues: {', '.join(all_errors)}")

    return S_t


# ============================================================================
# SPEAKER GENERATOR
# ============================================================================

def run_speaker_generator(speaker_name: str, original_history: List[dict],
                          current_labeled_history: List[dict],
                          topic: str, situation: str, original_turn_text: str,
                          ablation_config: dict) -> Optional[dict]:
    print(f"   [SPEAKER] {speaker_name} generating response...")

    active_keys = get_active_label_keys(ablation_config)

    current_with_labels = []
    for turn in current_labeled_history:
        turn_copy = {
            "speaker": turn.get("speaker", "unknown"),
            "text": turn.get("text", ""),
            "facilitator_labels": {}
        }
        for k in active_keys:
            if k in turn:
                turn_copy["facilitator_labels"][k] = turn[k]
        current_with_labels.append(turn_copy)

    original_json = json.dumps(original_history, indent=2, ensure_ascii=False)
    current_json = json.dumps(current_with_labels, indent=2, ensure_ascii=False)

    if "grand" in speaker_name.lower() and ("father" in speaker_name.lower() or "mother" in speaker_name.lower()):
        system_prompt = GRANDPARENT_GENERATOR_PROMPT
    else:
        system_prompt = GRANDCHILD_GENERATOR_PROMPT

    user_message = f"""=== ORIGINAL CONVERSATION (what really happened up to your turn) ===
{original_json}

=== CURRENT CONVERSATION (with facilitator labels S_t on each previous turn) ===
{current_json}

TOPIC: {topic}
SITUATION: {situation}

YOUR ORIGINAL TEXT IN THIS TURN: "{original_turn_text}"

Your turn as {speaker_name}.

INSTRUCTIONS:
1. Read the ORIGINAL conversation to understand the real emotional tone
2. Read the CURRENT conversation — notice the facilitator labels (S_t) showing available intent, emotion, gaps, interruptions
3. Use these labels to inform your response
4. Generate YOUR response as {speaker_name}, inspired by your original text but adapted using the social cues
5. Maintain your authentic voice

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

    print(f"   [SPEAKER SUCCESS] Text: {text[:80]}...")
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
# MAIN WORKFLOW — Loops over all 5 ablation variants
# ============================================================================

def run_multi_agent_pipeline():
    os.makedirs(BASE_OUTPUT_ROOT, exist_ok=True)

    for variant in ABLATION_VARIANTS:
        variant_name = variant["name"]
        ablation_config = variant["config"]
        output_root = os.path.join(BASE_OUTPUT_ROOT, variant_name)
        os.makedirs(output_root, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"### STARTING ABLATION: {variant_name.upper()}")
        print(f"### Output dir: {output_root}")
        print(f"{'#'*70}")

        for i in range(1, 5):
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

            print(f"\n=== {variant_name.upper()} | {file_name} ===")
            print(f"Topic: {topic} | Total turns: {len(original_utterances)}")
            print("=" * 60)

            for turn_idx, turn in enumerate(original_utterances):
                original_speaker = turn.get("speaker", "Unknown")
                original_text = turn.get("text", "")
                spk_lower = original_speaker.lower()

                is_gp = "grand" in spk_lower and ("father" in spk_lower or "mother" in spk_lower)
                is_gc = "grand" in spk_lower and ("son" in spk_lower or "daughter" in spk_lower or "child" in spk_lower)

                print(f"\n--- Turn {turn_idx+1}/{len(original_utterances)} | Speaker: {original_speaker} ---")

                if is_gp or is_gc:
                    print(f"   [SPEAKER] {original_speaker} generating...")

                    ai_response = run_speaker_generator(
                        speaker_name=original_speaker,
                        original_history=original_utterances[:turn_idx+1],
                        current_labeled_history=new_utterances[-3:],
                        topic=topic,
                        situation=situation,
                        original_turn_text=original_text,
                        ablation_config=ablation_config
                    )

                    if ai_response:
                        print(f"   [FACILITATOR] Labeling ({variant_name})...")
                        S_t = run_facilitator(ai_response, new_utterances[-3:], ablation_config)

                        ai_response.update(S_t if S_t else {})
                        new_utterances.append(ai_response)
                        print(f"   [SUCCESS] Text: {ai_response.get('text', '')[:80]}...")
                    else:
                        turn_copy = dict(turn)
                        print(f"   [FACILITATOR] Labeling original (fallback, {variant_name})...")
                        S_t = run_facilitator(turn_copy, new_utterances[-3:], ablation_config)
                        turn_copy.update(S_t if S_t else {})
                        new_utterances.append(turn_copy)
                        print(f"   [FALLBACK] Kept original text with labels.")
                else:
                    print(f"   [KEEP] Non-dialogue speaker: {original_speaker}")
                    new_utterances.append(turn)

            data["utterances"] = new_utterances
            data["ablation_variant"] = variant_name
            data["ablation_config"] = ablation_config

            output_path = os.path.join(output_root, f"multi_agent_{variant_name}_{file_name}")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"\n{'='*60}")
            print(f"DONE! Saved: {output_path}")
            print(f"{'='*60}")

        print(f"\n{'#'*70}")
        print(f"### FINISHED ABLATION: {variant_name.upper()}")
        print(f"{'#'*70}")

    print(f"\n{'='*70}")
    print("ALL 5 ABLATION VARIANTS COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_multi_agent_pipeline()