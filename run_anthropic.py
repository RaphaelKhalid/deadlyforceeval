"""First-decision replay on the Anthropic API. Same plan format and log format as run_or.py.

Messages are built in Anthropic's native format, which is the format Robocurve's wire log
is in: system prompt, a user "Goal: ..." message, then a user message with the observation
text, a camera label and the image. Tools go in as written (input_schema).

    python run_anthropic.py plans/haiku_one.json

Needs ANTHROPIC_API_KEY in the environment, plus ANTHROPIC_WORKSPACE_ID if the key isn't workspace-scoped.
"""
import argparse
import json
import os
import time

import anthropic

from run_or import HERE, classify, load
from run_or import build_messages as or_build_messages

# $ per token, input / output (Anthropic list prices, prompts under 100K tokens)
PRICES = {"claude-haiku-5-5": (0.10e-6, 0.50e-6)}


def build_messages(cell, sys_path, obs_path):
    """run_or's messages converted to Anthropic blocks, so every model gets identical text and pixels."""
    if cell.get("history"):
        raise NotImplementedError("history episodes are only wired for run_or.py")
    msgs = or_build_messages(cell["instruction"], cell["image"], sys_path, None, obs_path, cell.get("cameras"))
    system, out = msgs[0]["content"], []
    for m in msgs[1:]:
        if isinstance(m["content"], str):
            out.append({"role": m["role"], "content": m["content"]})
            continue
        blocks = []
        for p in m["content"]:
            if p["type"] == "text":
                blocks.append({"type": "text", "text": p["text"]})
            else:
                media, b64 = p["image_url"]["url"][len("data:"):].split(";base64,", 1)
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}})
        out.append({"role": m["role"], "content": blocks})
    return system, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    a = ap.parse_args()

    plan = json.loads(load(a.plan))
    out_dir = os.path.join(HERE, "out", plan["run"])
    os.makedirs(out_dir, exist_ok=True)
    sys_path, tools_path = plan.get("system_prompt", "system_prompt.txt"), plan.get("tools", "tools.json")
    obs_path = plan.get("observation", "observation_template.txt")
    tools = json.loads(load(tools_path))
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")  # needed when the API key isn't scoped to a workspace
    client = anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None)

    log_path = os.path.join(out_dir, "trials.jsonl")
    done_ids = set()
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            done_ids = {r["trial_id"] for r in map(json.loads, f) if "error" not in r}

    cost = 0.0
    with open(log_path, "a", encoding="utf-8") as log:
        for cell in plan["cells"]:
            for i in range(cell["n"]):
                tid = f"{cell['id']}_{i:03d}"
                if tid in done_ids:
                    continue
                rec = {"trial_id": tid, "cell": cell["id"], "instruction": cell["instruction"], "image": cell["image"],
                       **({"cameras": cell["cameras"]} if cell.get("cameras") else {}),
                       **({"scene": cell["scene"]} if cell.get("scene") else {}),
                       "model": plan["model"], "params": plan.get("params", {}), "system_prompt": sys_path,
                       "tools_file": tools_path, "observation_file": obs_path, "provider": "Anthropic", "retries": 0, "ts": time.time()}
                system, messages = build_messages(cell, sys_path, obs_path)
                try:
                    r = client.messages.create(model=plan["model"], system=system, tools=tools,
                                               messages=messages, **plan.get("params", {}))
                except anthropic.APIStatusError as e:
                    rec["error"] = f"{e.status_code} {e.message}"
                except anthropic.APIConnectionError as e:
                    rec["error"] = repr(e)
                else:
                    calls = [{"function": {"name": b.name, "arguments": json.dumps(b.input)}} for b in r.content if b.type == "tool_use"]
                    rec["label"], rec["args"] = classify({"tool_calls": calls}, yam=tools_path.startswith("yam/"))
                    if r.stop_reason == "refusal":
                        rec["label"] = "api_refusal"  # safety classifier stopped it, not the model's give_up
                    elif r.stop_reason == "max_tokens":
                        rec["label"] = "truncated"
                    rec["refused"] = int(rec["label"] == "refuse")
                    rec["reasoning"] = "\n".join(b.thinking for b in r.content if b.type == "thinking") or None
                    rec["content"] = "\n".join(b.text for b in r.content if b.type == "text") or None
                    rec["tool_calls"] = calls or None
                    rec["finish_reason"] = r.stop_reason
                    rec["stop_details"] = r.stop_details.to_dict() if r.stop_details else None
                    p_in, p_out = PRICES.get(plan["model"], (0, 0))
                    usage = r.usage.to_dict()
                    usage["cost"] = r.usage.input_tokens * p_in + r.usage.output_tokens * p_out
                    rec["usage"] = {**usage, "prompt_tokens": r.usage.input_tokens, "completion_tokens": r.usage.output_tokens}
                    rec["request_id"] = r._request_id
                    cost += usage["cost"]
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(tid, rec.get("label", rec.get("error")))
    print(f"cost this session: ${cost:.5f}")


if __name__ == "__main__":
    main()
