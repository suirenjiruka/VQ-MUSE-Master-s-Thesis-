# Full-Mask Text Supervision Experiment - 2026-07-08

## Base

Start from `951649759c1d31dc1c1cabc0199455b6f8bad65d` with active latent-logit mixing removed. Keep the stable architecture:

- main AdaLN branch
- source ControlNet-style branch
- VQ delta residual branch
- normal random-mask CE / VQ latent losses

## Problem

The previous text advantage loss was computed on the normal random-mask training state. It decreased, but one-hop diagnosis still showed:

```text
correct text << null text
```

The mismatch is that training saw partially visible target tokens, while the diagnosis and early BERT generation start from a full/high-mask state.

## Change

Keep the normal MoMask random-mask CE unchanged. Add a separate full-mask text branch:

```text
full-mask input + source + correct text  -> C
full-mask input + source + null text     -> B
full-mask input + source + shuffled text -> S

full_adv  = ReLU(m1 - (logp_C(target) - logp_B(target)))
full_rank = ReLU(m2 - (logp_C(target) - logp_S(target)))
full_cfg  = CE(B + scale * (C - B), target)
```

This directly trains the same state measured by the one-hop diagnosis.

## Expected Signal

The main check is:

```text
one-hop correct-null should move from strongly negative to positive
```

If `full_adv/full_cfg` decrease but one-hop still fails, the issue is not just teacher-forcing mismatch and we should inspect branch mismatch or source/text fusion itself.

