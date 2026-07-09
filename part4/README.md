# Part 4 — LLM-Based Explainability Layer
**Track chosen: (C) Model Prediction Explanation Pipeline**
## What this does

`part_4.py` takes the Random Forest pipeline from Part 3 (`best_model.pkl`)
and wraps it with an LLM that turns a raw prediction into a plain-language
explanation for someone like a pricing analyst — basically "here's what the
model predicted and why," in a validated JSON format instead of free text.

- `call_llm()` — wrapper around the OpenRouter chat completions API
  (`openai/gpt-4o-mini`). If `LLM_API_KEY` isn't set, or the request fails
  or times out, it just falls back to a deterministic mock response
  (prefixed `[MOCK]` so it's obvious) — so the whole thing still runs
  end-to-end without a key or network access.
- PII guardrail (`has_pii` / `guarded_call_llm`) — regex check for emails
  and phone numbers in anything about to get sent to the LLM. If it finds
  something, it blocks the call before it goes out and returns `None`.
- `encode_record()` — redoes the exact ordinal/one-hot encoding from Parts
  2–3 (ordinal for `cut`/`clarity`, one-hot for `color` with `D` dropped as
  reference) so a raw feature dict can go straight into `best_model.pkl`.
- Explanation schema — strict JSON schema with 5 required fields:
  `prediction_label`, `confidence_level`, `top_reason`, `second_reason`,
  `next_step`. `validate_explanation()` parses whatever the LLM returns,
  checks it against the schema, and falls back to a safe all-`None`
  structure if anything goes wrong (bad JSON, missing field, wrong enum
  value, blocked call, whatever).
- Main pipeline: for 3 test diamonds I picked by hand — predict class +
  probability with the Part 3 model, build a prompt describing the
  features and prediction, run it through the PII guardrail, call the LLM,
  validate the response.
- Temperature A/B: run the same 3 diamonds through the LLM at temp=0.0 and
  temp=0.7 to see how much the reasoning text actually changes with more
  sampling randomness.

## Running it

### Dependencies
```bash
pip install pandas joblib requests jsonschema
```
(`jsonschema` isn't strictly required — there's a small built-in fallback
validator in the script if you don't have it installed.)

### Prerequisite
Part 3 needs to run first to produce `best_model.pkl`. This script expects
it at `../part3/best_model.pkl` relative to itself — adjust `model_path` in
`main()` if your folders are set up differently.

### API key

This one actually calls a real LLM, so it needs an API key. **Nothing is
hardcoded in this repo.** To run it for real:

1. Make a `.env` file in this folder (already in `.gitignore`, won't get
   committed) with:
   ```
   LLM_API_KEY=your_openrouter_api_key_here
   ```
2. Or just export it directly before running — on Windows:
   ```cmd
   set LLM_API_KEY=your_openrouter_api_key_here
   python part_4.py
   ```
   on macOS/Linux:
   ```bash
   export LLM_API_KEY=your_openrouter_api_key_here
   python part_4.py
   ```

If `LLM_API_KEY` isn't set at all, it just falls back to the mock response
(clearly labeled `[MOCK]`), so it's still fully runnable and gradeable
without any key.

### Command
```bash
python part_4.py
```

## Findings

- With a real key set, all 3 test diamonds came back with valid,
  schema-conformant JSON (`status=pass` across the board), and the
  reasoning was actually specific to the diamond — things like "Ideal cut
  quality" or "Low clarity (SI2)" rather than generic filler text.
- The PII guardrail did what it was supposed to: it blocked a prompt that
  had an email address in it before that ever reached the LLM, and let a
  clean prompt through fine — so it's reacting to actual content, not
  something unrelated about the request.
- Bumping temperature from 0.0 to 0.7 visibly changed which feature got
  picked as the "top reason" for a couple of the diamonds (it'd swap
  between clarity and color as the second reason), which is basically what
  you'd expect from sampling temperature. The prediction label and
  confidence level stayed the same across both temperatures though, since
  those come straight from the deterministic model, not from the LLM.
- The mock fallback (kicks in automatically with no API key) produces the
  same structure as the real LLM path, so I could still verify the rest of
  the pipeline — guardrail, encoding, validation — end to end even without
  hitting the actual API.
