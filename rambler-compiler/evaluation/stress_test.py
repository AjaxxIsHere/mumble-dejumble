"""Rambler Stress Test runner: scores model predictions over the 200-record
hold-out and produces the per-level report + the spec §11 case checks.

Usage: python -m evaluation.stress_test predictions.jsonl [--out=path]
"""

import json
import sys
from pathlib import Path

from data.common import write_json
from data.stress_test import build_stress_records
from evaluation.metrics import aggregate, score_record

# The spec §11 cases (matched by exact input) that Experiment 1 famously failed.
SPEC_CASES = {
    "smoke_mid_sentence_correction": (
        "uh i i wanted to send sarah the project files tomorrow actually no wednesday morning"
    ),
    "tuesday_thursday_with_meaningful_tuesday": (
        "okay so basically I was thinking we could meet Tuesday no wait Thursday afternoon "
        "because John isn't available Tuesday"
    ),
    "message_with_correction_and_professional": (
        "uh can you write a message to Sarah saying I I won't be able to make it tomorrow no "
        "Wednesday morning and make it professional"
    ),
    "list_edit_remove_and_add": "make a list of milk eggs bread actually remove eggs and add paratha",
    "server_port_correction_professional": (
        "tell alex the the server crashed because of port 8080 wait port 3000 and make it "
        "sound professional"
    ),
    "mixed_filler_rep_corr": "uh i i think we should meet tuesday no wait make that thursday afternoon at 4",
}


def run_stress(rows: list[dict]) -> dict:
    """rows: prediction rows (record fields + 'actual') for stress_* ids."""
    stress = {r["id"]: r for r in build_stress_records()}
    scored = []
    for row in rows:
        rid = row.get("id")
        if not rid or rid not in stress:
            continue
        record = stress[rid]
        actual = row.get("actual", "")
        scored.append({"record": record, **score_record(record, actual)})

    agg = aggregate(scored)
    agg["per_level"] = {}
    for level in ("easy", "medium", "hard", "extreme"):
        rows_l = [r for r in scored if r["record"].get("level") == level]
        agg["per_level"][level] = _section_means(rows_l)

    agg["spec_cases"] = {}
    for name, in_text in SPEC_CASES.items():
        hit = [r for r in scored if r["record"]["input"] == in_text]
        if hit:
            row = hit[0]
            agg["spec_cases"][name] = {
                "normalized_match": row["normalized_match"],
                "correction_resolution": row["correction_resolution"],
                "actual": row["record"].get("actual", "") if "actual" in row else None,
            }
        else:
            agg["spec_cases"][name] = {"found": False}

    return agg


def _section_means(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for metric in ("exact_match", "normalized_match", "wer", "cer", "entity_preservation",
                   "correction_resolution", "compositional_pass"):
        vals = [float(r[metric]) if isinstance(r[metric], bool) else r[metric] for r in rows]
        out[metric] = sum(vals) / len(vals)
    return out


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--out")]
    out_arg = next((a.split("=", 1)[1] for a in argv[1:] if a.startswith("--out=")), None)
    if not args:
        print("usage: python -m evaluation.stress_test predictions.jsonl [--out=path]", file=sys.stderr)
        return 2
    rows = []
    with open(args[0], encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    result = run_stress(rows)
    if out_arg:
        write_json(Path(out_arg), result)
        print(f"wrote {out_arg}")
    else:
        print(json.dumps({k: v for k, v in result.items() if k != "per_record"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
