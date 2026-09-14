"""Capture/compare tool for checking whether a code change (e.g. to
timegan.py's embedder/recovery/generator/discriminator, now the only
copy of that logic after folding in timegan_static.py's static-feature
support and deleting that file) altered training behavior.

Why this exists instead of a golden-value test: verified directly that
this TF1 graph-mode code is NOT bit-reproducible in this environment even
with every seed fixed and execution forced single-threaded -- two
consecutive runs of the identical script, same seed, same config, produced
different generated output (see tests/test_training.py's module docstring
for the actual numbers). A single before/after run is indistinguishable
from ordinary noise. This tool instead captures several repeated runs on
each side to characterize that noise, then reports whether the "after"
side's values look like more of the same noise, or like a real shift.

Two-step workflow, run manually at refactor time -- NOT wired into pytest
or CI, since it does real training and isn't asserting anything on its
own, just presenting numbers for a human to judge:

    # Before touching the code:
    python3 report_training_comparison.py capture --label before

    # After the change:
    python3 report_training_comparison.py capture --label after

    # Compare:
    python3 report_training_comparison.py compare before after

Tracks generated-output summary statistics (mean/std/min/max), the one
thing train_timegan_timed and train_timegan (with or without
ori_data_static) all return in common.
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import tensorflow as tf

from timegan.data_loading import real_data_loading
from timegan.timegan import train_timegan
from timegan.timegan import train_timegan_timed

OUT_DIR = ".training_comparison"
TINY_PARAMS = dict(hidden_dim=4, num_layer=2, batch_size=8, module="gru", iterations=2)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _summary_stats(generated):
    flat = np.concatenate([np.asarray(seq).ravel() for seq in generated])
    return dict(
        mean=float(flat.mean()), std=float(flat.std()),
        min=float(flat.min()), max=float(flat.max()),
    )


def _run_once(fn_name, seed):
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    if fn_name in ("timed", "base"):
        ori_data = np.asarray(real_data_loading("stock", seq_len=8)[:40], dtype=np.float32)
        if fn_name == "timed":
            phase, generated = train_timegan_timed(
                ori_data, dict(TINY_PARAMS), in_filename=f"/tmp/report_cmp_{fn_name}_{seed}",
                seconds=120, phase=1, new=True, num_generate=10,
            )
            if phase != 4:
                raise RuntimeError(f"train_timegan_timed didn't finish in one call (phase={phase})")
        else:
            generated = train_timegan(
                ori_data, dict(TINY_PARAMS), filename=f"/tmp/report_cmp_{fn_name}_{seed}",
            )
    elif fn_name == "static":
        no, seq_len, dim = 32, 6, 3
        ori_data = np.random.uniform(0, 1, [no, seq_len, dim]).astype(np.float32)
        ori_data_static = np.random.randint(0, 2, size=(no, 1)).astype(np.float32)
        generated = train_timegan(
            ori_data, dict(TINY_PARAMS), filename=f"/tmp/report_cmp_{fn_name}_{seed}",
            ori_data_static=ori_data_static,
        )
    else:
        raise ValueError(f"unknown fn: {fn_name}")

    return _summary_stats(generated)


def capture(label, fn_name, runs):
    os.makedirs(OUT_DIR, exist_ok=True)
    run_stats = []
    for i in range(runs):
        seed = 1000 * hash(label) % 10_000 + i  # varies by label so before/after don't share exact seeds
        seed = abs(seed) % (2**31)
        stats = _run_once(fn_name, seed)
        print(f"  run {i + 1}/{runs} (seed={seed}): {stats}")
        run_stats.append(stats)

    record = dict(
        label=label, fn=fn_name, runs=runs, git_commit=_git_commit(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), params=TINY_PARAMS,
        run_stats=run_stats,
    )
    out_path = os.path.join(OUT_DIR, f"{label}.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"wrote {out_path} ({runs} runs)")


def compare(label_a, label_b):
    with open(os.path.join(OUT_DIR, f"{label_a}.json")) as f:
        rec_a = json.load(f)
    with open(os.path.join(OUT_DIR, f"{label_b}.json")) as f:
        rec_b = json.load(f)

    if rec_a["fn"] != rec_b["fn"]:
        print(f"WARNING: comparing different functions ({rec_a['fn']} vs {rec_b['fn']}) -- "
              f"not a meaningful before/after comparison")

    print(f"{label_a}: fn={rec_a['fn']} commit={rec_a.get('git_commit')} runs={rec_a['runs']}")
    print(f"{label_b}: fn={rec_b['fn']} commit={rec_b.get('git_commit')} runs={rec_b['runs']}")
    print()

    metrics = ["mean", "std", "min", "max"]
    any_no_overlap = False
    for metric in metrics:
        a_vals = [r[metric] for r in rec_a["run_stats"]]
        b_vals = [r[metric] for r in rec_b["run_stats"]]
        a_lo, a_hi = min(a_vals), max(a_vals)
        b_lo, b_hi = min(b_vals), max(b_vals)
        overlap = max(a_lo, b_lo) <= min(a_hi, b_hi)
        flag = "" if overlap else "  <-- NO OVERLAP, worth a closer look"
        if not overlap:
            any_no_overlap = True
        print(f"{metric:5s}  {label_a} range=[{a_lo:.4f}, {a_hi:.4f}]  "
              f"{label_b} range=[{b_lo:.4f}, {b_hi:.4f}]{flag}")

    print()
    if any_no_overlap:
        print("At least one metric's ranges don't overlap across runs -- run more repeats on "
              "each side before concluding this is a real behavior change rather than the "
              "handful-of-runs sample just missing each other by chance.")
    else:
        print("All tracked metrics' ranges overlap -- no evidence of a behavior change beyond "
              "ordinary run-to-run noise, at this sample size.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_capture = sub.add_parser("capture", help="run repeated trainings and save their stats")
    p_capture.add_argument("--label", required=True, help="e.g. 'before' or 'after'")
    p_capture.add_argument("--fn", choices=["timed", "base", "static"], default="timed")
    p_capture.add_argument("--runs", type=int, default=10)

    p_compare = sub.add_parser("compare", help="compare two captured labels")
    p_compare.add_argument("label_a")
    p_compare.add_argument("label_b")

    args = parser.parse_args()
    if args.command == "capture":
        capture(args.label, args.fn, args.runs)
    elif args.command == "compare":
        compare(args.label_a, args.label_b)
