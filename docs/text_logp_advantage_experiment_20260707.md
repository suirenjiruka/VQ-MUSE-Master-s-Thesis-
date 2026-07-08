# Text LogP Advantage Experiment - 2026-07-07

## Base

Based on the VQ latent + logit-change line after removing active latent-logit mixing. Keep:

- main AdaLN branch
- source ControlNet-style branch
- VQ delta residual branch
- scale-aware CE

## Problem

Rollout diagnosis showed:

- correct text is better than shuffled text
- null text is still better than correct text for target token match / latent distance

So the model learns some edit-text direction, but the correct edit text is not a necessary condition for target-token prediction.

## Change

Replace the previous `cfg_res / cfg_dir` losses with direct text log-prob advantage:

```text
B = source + null text
C = source + correct text
S = source + shuffled text

text_adv  = ReLU(m - (logp_C(target) - logp_B(target)))
text_rank = ReLU(m - (logp_C(target) - logp_S(target)))
cfg_res   = CE(B + scale * (C - B), target)
```

The loss is only applied to edit samples, changed tokens, and predicted tokens. Scale-aware weights are reused, so fine-scale tokens receive stronger late-stage pressure.

The null branch follows the real CFG branch B (`source + null text`, no VQ-delta text residual). The shuffled branch uses the same text-active path as the correct branch, including the VQ-delta residual path, but it is computed with no gradient and only serves as a negative baseline.

`cfg_res` is kept because it directly supervises the same boosted text residual used by CFG inference. `cfg_dir` is removed because target-vs-source gain alone was too weak and mostly behaved like a monitor.

## Expected Signal

This directly trains the observed failure case:

```text
correct text target logp should be higher than null/shuffled text target logp
```

It does not add a new branch. It changes the training pressure on the existing text path.
