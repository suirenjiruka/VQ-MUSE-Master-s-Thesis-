# EMA + Gen-Dominant Regularization Experiment

## Goal

Test whether the edit bottleneck is mainly overfitting instead of model capacity or source/text balance.

Base model:

```text
951649759c1d31dc1c1cabc0199455b6f8bad65d style
+ VQ delta residual branch
+ latent-logit classifier path disabled
+ same-source contrast disabled
```

This experiment keeps the model architecture stable and only changes the training protocol.

## Diagnosis Behind This Run

Previous same-source contrast showed:

```text
train same_gap opens
val same_gap stays near 0
```

So the model can use text on train samples, but the behavior does not generalize.
This points to memorization / overfitting rather than insufficient capacity.

The old sampler used:

```text
gen : edit = 50 : 50
```

This oversampled the small MotionFix edit set and likely amplified memorization in shared weights.

## Training Changes

### EMA

EMA is now active in `VA_trainer.py`.

```text
after optimizer step -> update EMA
before validation/eval -> copy EMA weights to model
after validation/eval -> restore raw training weights
checkpoint -> save EMA state
```

Old checkpoints without EMA initialize EMA from the loaded model.

### Gen-Dominant Task Mix

`Train_motion.py` now reads:

```yaml
training:
  edit_sample_ratio: 0.30
```

The sampler becomes:

```text
gen  = 0.70
edit = 0.30
```

This is close to the natural train split:

```text
gen 24546
edit 10774
edit ratio ~= 0.305
```

### Disabled Same-Source Contrast

```yaml
loss:
  weight_same_src: 0.0
```

Reason:

```text
same-source contrast is useful as diagnosis,
but train opens / val does not,
so it is currently a memorization-prone loss.
```

## Active Losses

```text
L =
  CE_target
  + w_info   * InfoNCE(target_motion, text)
  + w_delta  * VQ_delta
  + w_latent * latent_CE
  + w_rank   * latent_rank
```

Current config:

```yaml
weight_transformer_loss: 1.0
weight_motion_text_InfoNCE: 0.2
weight_delta: 0.5
weight_latent: 0.05
weight_rank: 0.05
weight_same_src: 0.0
```

No new branch is added.
No source branch weakening is used.
No operator contrast is used yet.

## Expected Signal

This run should answer:

```text
Can EMA + gen-dominant mixed training reduce edit overfitting
without changing the model architecture?
```

Useful metrics:

```text
Gen R@1/FID
Edit G2T R@1
Edit G2S R@1
TMR-FID
rollout correct-null gap
val same_gap if available
```

If validation improves:

```text
overfitting was a major bottleneck
```

If validation remains flat:

```text
move to instruction-recurrence operator learning
or synthetic multi-edit data
```

## Next Step If This Fails

Do not add more generic logp losses.

Next candidate:

```text
instruction-recurrence operator loss
same instruction_key
different source
different target
```

It must be paired with this anti-overfitting setup:

```text
EMA
gen-dominant mix
small / stable model
```

