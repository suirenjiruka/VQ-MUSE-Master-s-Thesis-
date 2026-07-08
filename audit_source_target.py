"""Source->target audit for MotionFix edit pairs.

Question it answers: does the SAME source motion appear with DIFFERENT edit texts
producing DIFFERENT targets? If not, text is redundant given source, and no
architecture/loss can force the model to use text.

It groups edit pairs by the CONTENT of their source motion (a stable hash), so it
does not need any source-id metadata. Mirror pairs (M-prefix) hash differently
(mirrored array), so they are naturally treated as distinct sources.

Run (in your WSL env, where the data lives):
    python audit_source_target.py \
        --root_dir /home/imlab/HumanML3D/repo \
        --split /home/imlab/HumanML3D/repo/HumanML3D/train.txt

Optional overrides: --motion_dir, --text_dir, --motionfix_start_id (default 400000)
"""
import os
import argparse
import hashlib
from collections import defaultdict, Counter

import numpy as np


def stable_hash(arr, decimals=3):
    a = np.ascontiguousarray(np.round(np.asarray(arr, dtype=np.float64), decimals))
    return hashlib.md5(a.tobytes()).hexdigest()


def is_edit_id(name, start_id):
    raw = name[1:] if name.startswith("M") else name
    return raw.isdigit() and int(raw) >= start_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--split", required=True, help="split .txt (train/val) to audit")
    ap.add_argument("--motion_dir", default=None)
    ap.add_argument("--text_dir", default=None)
    ap.add_argument("--motionfix_start_id", type=int, default=400000)
    ap.add_argument("--decimals", type=int, default=3, help="rounding for source hash")
    args = ap.parse_args()

    motion_dir = args.motion_dir or os.path.join(args.root_dir, "HumanML3D", "new_joint_vecs")
    text_dir = args.text_dir or os.path.join(args.root_dir, "HumanML3D", "texts")

    with open(args.split) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    edit_names = [n for n in names if is_edit_id(n, args.motionfix_start_id)]
    print(f"split: {args.split}")
    print(f"total ids: {len(names)}, edit ids: {len(edit_names)}")

    # source_hash -> list of {name, target_hash, text}
    by_source = defaultdict(list)
    n_ok, n_fail = 0, 0
    for name in edit_names:
        mpath = os.path.join(motion_dir, name + ".npy")
        try:
            m = np.load(mpath, allow_pickle=True).item()
            src, tgt = m["source"], m["target"]
        except Exception as e:
            n_fail += 1
            continue
        try:
            with open(os.path.join(text_dir, name + ".txt"), encoding="utf-8") as tf:
                text = tf.read().strip().split("\n")[0].split("#")[0].strip()
        except Exception:
            text = ""
        by_source[stable_hash(src, args.decimals)].append(
            {"name": name, "tgt": stable_hash(tgt, args.decimals), "text": text}
        )
        n_ok += 1

    print(f"loaded pairs: {n_ok} (failed to load: {n_fail})")
    if n_ok == 0:
        print("No pairs loaded -- check paths.")
        return

    # per-source stats
    pairs_per_source = Counter()
    multi_target_sources = 0        # same source, >=2 DISTINCT targets  (genuine multi-edit)
    multi_pair_sources = 0          # same source, >=2 pairs (may be duplicates)
    genuine_multi_edit_pairs = 0    # pairs living in a genuine multi-target source group
    examples = []
    for shash, items in by_source.items():
        pairs_per_source[len(items)] += 1
        if len(items) >= 2:
            multi_pair_sources += 1
        distinct_tgts = {it["tgt"] for it in items}
        distinct_texts = {it["text"] for it in items}
        if len(distinct_tgts) >= 2:
            multi_target_sources += 1
            genuine_multi_edit_pairs += len(items)
            if len(examples) < 8:
                examples.append((len(items), len(distinct_tgts), len(distinct_texts),
                                 [it["name"] for it in items][:4],
                                 [it["text"][:40] for it in items][:3]))

    n_src = len(by_source)
    print("\n==== SOURCE-TARGET AUDIT ====")
    print(f"unique sources: {n_src}   (pairs/source ratio: {n_ok / n_src:.2f})")
    print(f"pairs-per-source histogram: {dict(sorted(pairs_per_source.items()))}")
    print(f"sources with >=2 pairs:            {multi_pair_sources} "
          f"({100*multi_pair_sources/n_src:.1f}% of sources)")
    print(f"sources with >=2 DISTINCT targets: {multi_target_sources} "
          f"({100*multi_target_sources/n_src:.1f}% of sources)   <-- genuine multi-edit")
    print(f"pairs inside genuine multi-edit groups: {genuine_multi_edit_pairs} "
          f"({100*genuine_multi_edit_pairs/n_ok:.1f}% of all edit pairs)")

    print("\n---- verdict ----")
    frac = genuine_multi_edit_pairs / n_ok
    if frac < 0.05:
        print(f"TEXT IS REDUNDANT: only {100*frac:.1f}% of pairs share a source with a different target.")
        print("-> source almost always determines target; loss/arch cannot force text.")
        print("-> the real lever is DATA: synthesize same-source multi-edit pairs.")
    elif frac < 0.25:
        print(f"WEAK text necessity: {100*frac:.1f}% of pairs are genuine multi-edit.")
        print("-> text is only sometimes needed; expect limited ceiling from arch/loss alone.")
    else:
        print(f"TEXT IS NEEDED: {100*frac:.1f}% of pairs are genuine multi-edit.")
        print("-> source does NOT determine target; failure is arch/loss/quantity, not redundancy.")

    if examples:
        print("\n---- example genuine multi-edit groups (source shared, targets differ) ----")
        for npair, ntgt, ntext, nms, txts in examples:
            print(f"  {npair} pairs / {ntgt} targets / {ntext} texts  names={nms}")
            for t in txts:
                print(f"      text: {t}")


if __name__ == "__main__":
    main()
