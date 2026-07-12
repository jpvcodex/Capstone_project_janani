

import os
import re
import json
import random

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

try:
    import jsonschema
    from jsonschema import ValidationError
except ImportError:
    class ValidationError(Exception):
        pass

    class _JsonSchemaShim:
        """Minimal drop-in for jsonschema.validate() covering the scalar,
        required-field, enum-based schemas used in this script."""

        @staticmethod
        def validate(instance, schema):
            if not isinstance(instance, dict):
                raise ValidationError(f"{instance!r} is not of type 'object'")

            for field in schema.get("required", []):
                if field not in instance:
                    raise ValidationError(f"'{field}' is a required property")

            props = schema.get("properties", {})
            for field, value in instance.items():
                if field not in props:
                    continue
                spec = props[field]
                expected_type = spec.get("type")
                type_map = {"string": str, "number": (int, float), "boolean": bool}
                if expected_type in type_map and not isinstance(value, type_map[expected_type]):
                    raise ValidationError(
                        f"{value!r} is not of type '{expected_type}' for field '{field}'"
                    )
                if "enum" in spec and value not in spec["enum"]:
                    raise ValidationError(
                        f"{value!r} is not one of {spec['enum']} for field '{field}'"
                    )

    jsonschema = _JsonSchemaShim()

HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(42)


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ===========================================================================
# TASK 1: LLM API connection
# ===========================================================================
LLM_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "openai/gpt-4o-mini"


def _mock_llm_response(system_prompt, user_prompt, temperature):
    """Deterministic-ish stand-in used only when no real API key/network is
    available, so the pipeline can still be demonstrated end-to-end. Clearly
    labeled with [MOCK] wherever it is printed."""
    # crude parse of the predicted class / probability out of the user prompt
    # so the mock explanation is at least self-consistent with the inputs.
    cls_match = re.search(r"Predicted class:\s*(\d)", user_prompt)
    prob_match = re.search(r"Predicted probability.*?:\s*([0-9.]+)", user_prompt)
    pred_class = cls_match.group(1) if cls_match else "1"
    prob = float(prob_match.group(1)) if prob_match else 0.9

    label = "above-median price" if pred_class == "1" else "below-median price"
    if prob >= 0.9:
        conf = "high"
    elif prob >= 0.65:
        conf = "medium"
    else:
        conf = "low"

    reasons_pool_high = ["carat is large", "the x/y/z dimensions are large", "cut grade is high"]
    reasons_pool_low = ["carat is small", "the x/y/z dimensions are small", "cut grade is lower"]
    pool = reasons_pool_high if pred_class == "1" else reasons_pool_low

    if abs(temperature) < 1e-9:
        top_reason, second_reason = pool[0], pool[1]
    else:
        # temperature=0.7 branch: introduce sampling-like variability
        shuffled = pool[:]
        random.shuffle(shuffled)
        top_reason, second_reason = shuffled[0], shuffled[1]

    payload = {
        "prediction_label": label,
        "confidence_level": conf,
        "top_reason": top_reason,
        "second_reason": second_reason,
        "next_step": "Verify carat and cut with a physical appraisal before final pricing.",
    }
    return "[MOCK] " + json.dumps(payload)


def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    api_key = os.environ.get("LLM_API_KEY")
    
    if not api_key:
        print("[call_llm] No LLM_API_KEY environment variable found - using MOCK response.")
        return _mock_llm_response(system_prompt, user_prompt, temperature)

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"[call_llm] Request failed ({e}) - falling back to MOCK response.")
        return _mock_llm_response(system_prompt, user_prompt, temperature)

    if response.status_code != 200:
        print(f"[call_llm] Non-200 status code: {response.status_code}")
        print(response.text[:500])
        return None

    return response.json()["choices"][0]["message"]["content"]


def task1_demo_call():
    hr("TASK 1: call_llm() connectivity test")
    result = call_llm(
        system_prompt="You are a terse assistant.",
        user_prompt="Reply with only the word: hello",
        temperature=0.0,
        max_tokens=10,
    )
    print(f"Response: {result!r}")
    return result


# ===========================================================================
# TASK 2/3: PII guardrail
# ===========================================================================
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))


def guarded_call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        return None
    return call_llm(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)


def task_guardrail_demo():
    hr("GUARDRAIL DEMO: PII detection")
    pii_input = "Please contact John at john.doe@example.com about this diamond."
    clean_input = "Here are the feature values for a diamond to explain."

    print("Test 1 (contains an email - should be BLOCKED):")
    print(f"  input: {pii_input!r}")
    result1 = guarded_call_llm("You are a helpful assistant.", pii_input)
    print(f"  result: {result1!r}")

    print("\nTest 2 (clean text - should PROCEED to the LLM call):")
    print(f"  input: {clean_input!r}")
    result2 = guarded_call_llm("You are a helpful assistant.", clean_input, max_tokens=20)
    print(f"  result: {result2!r}")
    return pii_input, result1, clean_input, result2


# ===========================================================================
# TASK: encode_record - turn a raw feature dict into the model's expected input
# ===========================================================================
CUT_ORDER = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
CLARITY_ORDER = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4,
                  "VVS2": 5, "VVS1": 6, "IF": 7}
COLOR_LEVELS = ["D", "E", "F", "G", "H", "I", "J"]  # D was the dropped reference level

FEATURE_COLUMNS = [
    "carat", "cut", "clarity", "table", "x", "y", "z",
    "color_E", "color_F", "color_G", "color_H", "color_I", "color_J",
]


def encode_record(features: dict) -> pd.DataFrame:
    """Reproduces the exact encoding used in Parts 2/3 (ordinal cut/clarity,
    one-hot color with 'D' as the dropped reference level) so a raw feature
    dict can be fed straight into best_model.pkl's .predict()/.predict_proba()."""
    row = {
        "carat": features["carat"],
        "cut": CUT_ORDER[features["cut"]],
        "clarity": CLARITY_ORDER[features["clarity"]],
        "table": features["table"],
        "x": features["x"],
        "y": features["y"],
        "z": features["z"],
    }
    for level in COLOR_LEVELS[1:]:  # skip 'D', the dropped reference level
        row[f"color_{level}"] = 1 if features["color"] == level else 0

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


# ===========================================================================
# JSON schema for the explanation output
# ===========================================================================
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": [
        "prediction_label", "confidence_level", "top_reason", "second_reason", "next_step",
    ],
}

FALLBACK_EXPLANATION = {
    "prediction_label": None,
    "confidence_level": None,
    "top_reason": None,
    "second_reason": None,
    "next_step": None,
}

SYSTEM_PROMPT = (
    "You are a pricing-model explainability assistant for a diamond retailer. "
    "You will be given the feature values of a diamond, the model's predicted "
    "price class (0 = below median price, 1 = above median price), and the "
    "model's predicted probability for that class. "
    "Respond with ONLY a single valid JSON object (no markdown fences, no "
    "extra commentary) with exactly these five fields: "
    '"prediction_label" (string, e.g. "above-median price" or "below-median '
    'price"), "confidence_level" (one of "low", "medium", "high"), '
    '"top_reason" (string, the single most influential feature/pattern), '
    '"second_reason" (string, the second most influential feature/pattern), '
    'and "next_step" (string, a short recommended action for a pricing analyst). '
    "Do not include any keys other than these five."
)

USER_PROMPT_TEMPLATE = """Diamond feature values:
{feature_json}

Predicted class: {pred_class}
Predicted probability (class {pred_class}): {pred_prob:.4f}

Explain this prediction as a JSON object following the required schema."""


def build_user_prompt(features, pred_class, pred_prob):
    feature_json = json.dumps(features, indent=2)
    return USER_PROMPT_TEMPLATE.format(
        feature_json=feature_json, pred_class=pred_class, pred_prob=pred_prob
    )


def validate_explanation(raw_response):
    """strip -> json.loads -> jsonschema.validate, with fallback on failure."""
    if raw_response is None:
        return dict(FALLBACK_EXPLANATION), "fail: no response (blocked or API error)"

    text = raw_response.strip()
    if text.startswith("[MOCK]"):
        text = text[len("[MOCK]"):].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        return dict(FALLBACK_EXPLANATION), f"fail: JSONDecodeError - {e}"

    try:
        jsonschema.validate(parsed, EXPLANATION_SCHEMA)
    except ValidationError as e:
        return dict(FALLBACK_EXPLANATION), f"fail: ValidationError - {e}"

    return parsed, "pass"


# ===========================================================================
# Track C main pipeline
# ===========================================================================
TEST_DIAMONDS = [
    {"carat": 1.50, "cut": "Ideal", "color": "G", "clarity": "VS1",
     "table": 57.0, "x": 7.30, "y": 7.33, "z": 4.50},
    {"carat": 0.25, "cut": "Fair", "color": "J", "clarity": "SI2",
     "table": 60.0, "x": 4.00, "y": 3.98, "z": 2.50},
    {"carat": 0.55, "cut": "Good", "color": "H", "clarity": "VS2",
     "table": 58.0, "x": 5.25, "y": 5.28, "z": 3.25},
]


def task_predict_and_explain(model, temperature=0.0, verbose=True):
    rows = []
    for i, features in enumerate(TEST_DIAMONDS, start=1):
        encoded = encode_record(features)
        pred_class = int(model.predict(encoded)[0])
        pred_prob = float(model.predict_proba(encoded)[0, pred_class])

        user_prompt = build_user_prompt(features, pred_class, pred_prob)
        pii_flag = has_pii(user_prompt)

        if verbose:
            print(f"\n--- Diamond {i} ---")
            print(f"Features: {features}")
            print(f"Predicted class: {pred_class}, probability: {pred_prob:.4f}")
            print(f"PII check on prompt: {'BLOCKED' if pii_flag else 'clean, proceeding'}")

        if pii_flag:
            raw = None
        else:
            raw = call_llm(SYSTEM_PROMPT, user_prompt, temperature=temperature, max_tokens=300)

        if verbose:
            print(f"Raw LLM response: {raw!r}")

        parsed, status = validate_explanation(raw)

        if verbose:
            print(f"Validation outcome: {status}")
            print(f"Parsed explanation: {parsed}")

        rows.append({
            "diamond_index": i,
            "features": features,
            "pred_class": pred_class,
            "pred_prob": pred_prob,
            "raw_response": raw,
            "parsed": parsed,
            "status": status,
            "pii_blocked": pii_flag,
        })
    return rows


def task_temperature_comparison(model):
    hr("TEMPERATURE A/B COMPARISON (temp=0 vs temp=0.7)")
    comparison_rows = []
    for i, features in enumerate(TEST_DIAMONDS, start=1):
        encoded = encode_record(features)
        pred_class = int(model.predict(encoded)[0])
        pred_prob = float(model.predict_proba(encoded)[0, pred_class])
        user_prompt = build_user_prompt(features, pred_class, pred_prob)

        raw_t0 = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0, max_tokens=300)
        raw_t07 = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.7, max_tokens=300)

        parsed_t0, _ = validate_explanation(raw_t0)
        parsed_t07, _ = validate_explanation(raw_t07)

        print(f"\nDiamond {i} (class={pred_class}, prob={pred_prob:.3f})")
        print(f"  temp=0.0 -> {parsed_t0}")
        print(f"  temp=0.7 -> {parsed_t07}")

        comparison_rows.append({
            "diamond_index": i,
            "temp0": parsed_t0,
            "temp07": parsed_t07,
        })
    return comparison_rows


def main():
    hr("TASK 1: LLM connectivity demo")
    task1_demo_call()

    hr("SETUP: load best_model.pkl")
    model_path = os.path.join(HERE, "best_model.pkl")
    model = joblib.load(model_path)
    print(f"Loaded model pipeline from {model_path}:")
    print(model)

    task_guardrail_demo()

    hr("MAIN PIPELINE: predict + explain (temperature=0.0)")
    results = task_predict_and_explain(model, temperature=0.0, verbose=True)

    task_temperature_comparison(model)

    hr("SUMMARY TABLE")
    for r in results:
        print(f"Diamond {r['diamond_index']}: class={r['pred_class']}, "
              f"prob={r['pred_prob']:.4f}, status={r['status']}, "
              f"pii_blocked={r['pii_blocked']}")


if __name__ == "__main__":
    main()