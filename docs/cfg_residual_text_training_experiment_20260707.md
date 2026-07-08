# CFG Residual Text Training

Date: 2026-07-07

Base:

```text
vq latent + logit change checkpoint/code line,
but remove latent-driven final logit mix.
```

## Motivation

Rollout diagnosis showed:

```text
correct text is better than shuffled text,
but source + null text can still match target as well or better.
```

This means the model learns some instruction direction,
but the edit text is not yet a necessary target-token decision signal.

## Removed

Latent-driven final classifier is disabled:

```yaml
model:
  use_latent_classifier: False
  delta_alpha: 0.0
```

So edit final logits are no longer:

```text
latent_w * latent_logits + raw_w * raw_logits
```

The VQ delta branch is still kept as hidden/control and latent supervision:

```text
delta_beta: 0.3
weight_delta: 0.5
weight_latent: 0.1
weight_rank: 0.03
```

## Kept

Scale-aware loss stays active:

```yaml
model:
  sc_loss: True
  sc_w_early: [1.2, 1.1, 1.0, 0.9]
  sc_w_late: [0.8, 1.0, 1.2, 1.4]
```

It applies to:

```text
1. final CE
2. latent CE
```

Generation samples keep weight 1.0.

## Core Change

Train the same residual used by edit CFG:

```text
B = source + null text
C = source + correct edit text
CFG = B + s * (C - B)
```

The source/null branch is detached:

```text
cfg_logits = B.detach() + cfg_res_scale * (C - B.detach())
```

This avoids damaging the source-preservation branch.
Only the correct-text branch is pushed to create a useful edit residual.

## Loss

Main loss remains:

```text
CE(C, target)
+ InfoNCE(target motion, text)
+ VQ delta / latent / rank losses
```

New losses:

```text
L_cfg_res = CE(B.detach() + s * (C - B.detach()), target)
```

and:

```text
delta = C - B.detach()
L_cfg_dir = max(0, margin - (delta[target_id] - delta[source_id]))
```

Meaning:

```text
correct edit text must make target token gain more than source token gain.
```

Important detail:

```text
L_cfg_dir is computed only where target_id != source_id.
Otherwise target/source are the same token and the margin becomes a constant with no useful gradient.
```

Config:

```yaml
loss:
  weight_cfg_res: 0.2
  weight_cfg_dir: 0.05
  cfg_res_scale: 2.0
  cfg_res_margin: 0.2
```

## Why This Is Different

Previous auxiliary losses often supervised a side signal:

```text
edit map
similarity
latent residual
extra text branch
```

but did not directly supervise the actual inference edit residual.

This experiment directly trains:

```text
C - B
```

which is the text edit direction used by dual CFG.

## Expected Good Sign

Rollout diagnosis should change from:

```text
correct ≈ null or correct < null
```

to:

```text
correct > null > shuffle
```

especially in:

```text
target token match
target latent distance
coarse scale target match
```

Visual output should move less randomly away from source and more consistently toward target.

## Stop Rule

If `cfg_res` decreases but rollout still shows:

```text
correct <= null
```

then the residual loss is being absorbed as another token-prior objective.
Do not increase weight blindly; move to a stronger text-conditioned classifier/teacher setup.
