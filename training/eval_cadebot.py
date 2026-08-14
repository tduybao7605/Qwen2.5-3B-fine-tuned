import json, os, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = os.getenv("CADEBOT_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LORA_PATH  = os.getenv("CADEBOT_MODEL_DIR", str(ROOT / "cadebot-lora"))
VAL_DATA   = str(ROOT / "dataset" / "val.jsonl")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
base  = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base, LORA_PATH)
model.eval()
print("Model loaded!\n")

with open(VAL_DATA) as f:
    samples = [json.loads(l) for l in f if l.strip()]
print(f"Val set: {len(samples)} mau\n")

correct   = 0
json_fail = 0
wrong     = []

for i, s in enumerate(samples):
    msgs            = s["messages"][:-1]
    expected        = json.loads(s["messages"][-1]["content"])
    expected_intent = expected["intent"]
    user_input      = s["messages"][1]["content"]

    text   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, do_sample=False)

    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    try:
        parsed           = json.loads(response)
        predicted_intent = parsed["intent"]
        json_ok          = True
    except Exception:
        predicted_intent = "PARSE_ERROR"
        json_ok          = False
        json_fail        += 1

    if predicted_intent == expected_intent:
        correct += 1
    else:
        wrong.append({
            "input":    user_input,
            "expected": expected_intent,
            "got":      predicted_intent,
            "json_ok":  json_ok,
        })

    print(f"[{i+1:02d}/{len(samples)}] {'OK' if predicted_intent == expected_intent else 'XX'}  "
          f"expected={expected_intent:<22} got={predicted_intent}")

print("\n" + "="*55)
print(f"Intent Accuracy : {correct}/{len(samples)} = {correct/len(samples)*100:.1f}%")
print(f"JSON Parse Fail : {json_fail}/{len(samples)}")
print("="*55)

if wrong:
    print("\nCac mau du doan sai:")
    for w in wrong:
        print(f"  Input   : {w['input'][:60]}")
        print(f"  Expected: {w['expected']}")
        print(f"  Got     : {w['got']}")
        print()
