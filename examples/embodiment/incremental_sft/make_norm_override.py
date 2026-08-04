#!/usr/bin/env python3
"""Regenerate the libero_130 action-norm override used to align incremental SFT to the teacher.

The incremental SFT normalizes each suite's actions with the libero_130 q01/q99 (not the per-suite
stats) so the SFT'd student shares the 130 teacher's 256-bin action tokenization. This script
extracts the `libero_130_no_noops_trajall` stats block from the 130-Base-Lora dataset_statistics
and writes it as a SINGLE-dataset stats file (the shape make_dataset_from_rlds / our sft_aligned.py
monkeypatch expect: {action:{mean,std,q01,q99,mask,...}, proprio:{...}, num_transitions, ...}).
"""
import argparse
import json

SRC = "/share/fanruochen-local/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora/dataset_statistics.json"
OUT = "/share/fanruochen-local/checkpoints/norm_override_libero130.json"
KEY = "libero_130_no_noops_trajall"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--key", default=KEY)
    a = ap.parse_args()
    inner = json.load(open(a.src))[a.key]
    json.dump(inner, open(a.out, "w"))
    print(f"wrote {a.out}")
    print(f"  action q01[:3]={inner['action']['q01'][:3]}  q99[:3]={inner['action']['q99'][:3]}")
