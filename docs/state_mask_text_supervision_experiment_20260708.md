# State-Mask Text Supervision Experiment - 2026-07-08

## Base

Use the stable `951649759c1d31dc1c1cabc0199455b6f8bad65d` line with active latent-logit mixing removed. Keep:

- main AdaLN branch
- source ControlNet-style branch
- VQ delta residual branch
- normal MoMask random-mask CE

## Problem

The previous random-mask text advantage loss decreased, but one-hop diagnosis still showed correct text much worse than null text. Pure full-mask supervision is too extreme because real BERT inference moves from all-mask to almost no-mask over multiple refinement steps.

## Change

Add a light inference-state text supervision branch. It samples a per-sample mask ratio instead of always using full mask:

```text
mask_ratio ~ Uniform(0.45, 0.95)
with 0.15 probability: mask_ratio = 1.0
```

Then train the text path at that sampled state:

```text
state input + source + correct text  -> C
state input + source + null text     -> B
state_adv  = ReLU(m1 - (logp_C(target) - logp_B(target)))
state_cfg  = CE(B + scale * (C - B), target)
```

Only sampled masked tokens are supervised in this branch. The normal random-mask CE remains unchanged.

The shuffled-text branch was removed to save training time. The current bottleneck is correct text vs null text, so the experiment focuses on making the edit instruction outperform the source-only/null-text branch.

Current weights:

```text
state_adv_margin = 1.0
weight_state_adv = 0.8
weight_state_cfg = 0.1
```

`state_gap = logp_C(target) - logp_B(target)` is logged directly. The target is to push this gap above 1.0 instead of inferring it indirectly from the hinge loss.

## Why Not Rollout Yet

Full rollout-level training would match inference better, but it is much heavier because it requires repeated autoregressive/BERT refinement inside each training step. This experiment uses a cheaper single-step approximation of rollout states first.

## Expected Check

The one-hop logp diagnosis should improve first:

```text
correct-null should move upward
```

The rollout diagnosis should improve only if the single-step state supervision transfers through iterative refinement.
