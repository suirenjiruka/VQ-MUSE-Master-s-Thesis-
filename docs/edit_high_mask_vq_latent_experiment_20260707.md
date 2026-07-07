# Edit High-Mask VQ Latent Experiment

Date: 2026-07-07

Base:

```text
vq latent + logit change
```

## Diagnosis

The bottleneck diagnostic showed:

```text
train visible changed-token leakage: about 0.35
all-mask argmax target hit: about 0.04
```

So the edit model can exploit visible target tokens during training, but inference starts from almost all masked tokens.

This creates a mismatch:

```text
training: visible target context + source + text -> target
inference: source + text -> target
```

## Core Change

Do not add a new branch.

Change edit-sample masking only:

```text
generation samples: keep original MoMask masking
editing samples: high-mask / all-mask biased masking
```

Config:

```yaml
training:
  m_drop: 0.15
  e_min: 0.75
  e_max: 1.0
  e_full: 0.3
```

Meaning:

```text
e_min/e_max: edit samples sample target mask ratio from this range
e_full: extra edit all-mask probability
```

For edit samples, masked target tokens no longer keep the original target id:

```text
gen:  random / mask / keep target
edit: random / mask only
```

This reduces answer leakage from target tokens.

## VQ Delta Setting

Remove final logit change:

```yaml
model:
  delta_alpha: 0.0
```

Keep VQ latent residual:

```yaml
model:
  delta_beta: 0.3
```

Reason:

```text
latent logit bias did not clearly break the 0.4 bottleneck.
VQ latent residual is kept as a hidden control signal.
```

## Expected Signal

Good sign:

```text
edit all-mask behavior improves
edit R@1 rises beyond the old 0.40 range
visual edit amplitude becomes more complete
gen R@1 stays close to the vq latent baseline
```

Bad sign:

```text
edit R@1 drops or stays near 0.35 - 0.40
gen stays good but edit all-mask remains weak
```

Interpretation if bad:

```text
target leakage is not the only bottleneck.
The next change must modify the text-conditioned token decision path itself.
```

## Result

Tested conservative setting:

```yaml
training:
  m_drop: 0.15
  e_min: 0.3
  e_max: 0.99
  e_full: 0.1

model:
  delta_alpha: 0.0
  delta_beta: 0.3
```

Run result:

```text
epoch 23
val loss: 5.925
vq_delta: 0.039
vq_latent: 6.133
vq_rank: 0.369
accuracy: 0.132

Gen R@1/2/3: 0.4439 / 0.6282 / 0.7328
Gen Matching: 3.4473
Gen FID: 1.0100

Edit G2T R@1/2/3: 0.3500 / 0.5141 / 0.6297
Edit G2S R@1/2/3: 0.3531 / 0.5312 / 0.6234
TMR-FID: 0.1419
```

Conclusion:

```text
failed.
```

Reason:

```text
even the conservative high-mask schedule reduced edit performance.
Removing latent logit change and forcing edit samples toward higher target masking did not improve text-conditioned editing.
```

Decision:

```text
do not carry high-mask into the next latent-driven classifier experiment.
rollback to vq latent + logit change before the next change.
```
