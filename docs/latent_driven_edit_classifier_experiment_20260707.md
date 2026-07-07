# Latent-Driven Edit Classifier

Date: 2026-07-07

Base:

```text
vq latent + logit change
```

Architecture figure:

```text
docs/latent_driven_edit_classifier_framework_20260707.png
```

## Motivation

Dynamic diagnosis showed that direct VQ token edit transition is unstable:

```text
source and target motions can look similar,
but source/target VQ ids differ on most tokens.
```

Previous VQ delta design used latent logits only as a small residual bias:

```text
final_logits = raw_logits + alpha * (latent_logits - source_logits)
```

This kept the raw token classifier as the main decision path.

## Core Change

Keep the existing architecture and existing VQ delta branch.

Do not add high-mask training.
Do not add a new text branch.

Change the edit final classifier:

```text
generation:
  final_logits = raw_logits

editing:
  final_logits = latent_w * latent_logits + raw_w * raw_logits
```

where:

```text
latent_w = latent_w_min + (latent_w_max - latent_w_min) * mask_ratio
raw_w = raw_w_sum - latent_w
```

Initial config:

```yaml
model:
  use_latent_classifier: True
  delta_alpha: 0.0
  delta_beta: 0.3
  delta_temp: 0.15
  latent_w_min: 0.5
  latent_w_max: 0.9
  raw_w_sum: 1.4
```

Interpretation:

```text
high mask ratio:
  latent branch leads edit direction

low mask ratio:
  raw branch contributes more token consistency and motion prior
```

## VQ Latent Branch

The delta encoder is still the existing 8-layer AdaLN ControlBranch.

It is a hidden-space condition integrator:

```text
transformer_input
text_cross
source_cross
z_src_proj
  -> delta_control_proj
  -> delta_encoder
  -> delta_head
```

The VQ latent prediction happens after the hidden encoder:

```text
z_src = codebook[aligned_source_ids]
pred_delta_z = delta_head(delta_hidden)
pred_target_z = z_src + pred_delta_z
latent_logits = cosine(pred_target_z, codebook) / delta_temp
```

## Loss

Generation samples:

```text
CE(raw_logits, target_id)
+ InfoNCE(target_motion, text)
```

Editing samples:

```text
CE(latent_w * latent_logits + raw_w * raw_logits, target_id)
+ weight_latent * CE(latent_logits, target_id)
+ weight_delta * SmoothL1(pred_delta_z, z_tgt - z_src)
+ weight_rank * rank(pred_target_z closer to z_tgt than z_src)
+ InfoNCE(target_motion, text)
```

Initial weights:

```yaml
loss:
  weight_delta: 0.5
  weight_latent: 0.1
  weight_rank: 0.03
```

## Why This Is Different

Failed VQ delta experiments:

```text
raw token classifier was still the main path.
latent branch was only a small residual or auxiliary signal.
```

This experiment:

```text
latent logits become the main edit classifier.
raw logits are kept as a motion/token prior.
```

## Expected Signal

Good sign:

```text
edit R@1 breaks past the old 0.40 - 0.43 region
visual edit amplitude becomes more complete
latent CE decreases meaningfully
gen remains close to baseline because gen path is raw_logits only
```

Bad sign:

```text
edit stays near or below 0.40
latent CE stays high
visual output remains partial/conservative
```

Interpretation if bad:

```text
VQ codebook latent state is also insufficient as an edit semantic space.
Stop repeating VQ latent residual variants and move to a different edit representation or classifier.
```

## Implementation Check

Checked files:

```text
model.py
VA_trainer.py
Adaln_encoder.py
configs/train_vamotion_hml.yaml
```

Compile / style check:

```text
python -m py_compile model.py VA_trainer.py Adaln_encoder.py
git diff --check -- model.py VA_trainer.py Adaln_encoder.py configs/train_vamotion_hml.yaml
```

Result:

```text
passed.
```

## Active Code Path

Training forward:

```text
target/source ids
  -> masked_motion_tokens
  -> text_cross
  -> source_cross
  -> transformer_input
  -> source control residuals
  -> VQ delta branch
  -> main AdaLN
  -> raw_logits
```

VQ delta branch:

```text
aligned_source_ids -> z_src
z_src -> z_src_proj
[transformer_input, text_cross, source_cross, z_src_proj]
  -> delta_control_proj
  -> delta_encoder
  -> delta_head
  -> pred_delta_z

pred_target_z = z_src + pred_delta_z
latent_logits = cosine(pred_target_z, codebook) / delta_temp
```

Final classifier:

```text
if generation sample:
  final_logits = raw_logits

if editing sample:
  final_logits = latent_w * latent_logits + raw_w * raw_logits
```

where:

```text
latent_w = 0.5 + 0.4 * mask_ratio
raw_w = 1.4 - latent_w
```

Important note:

```text
raw_logits means the main AdaLN output logits.
For edit samples, this main path still receives source control residuals and delta_residuals when delta_beta > 0.
So raw_logits is not a pure baseline-only branch; it is the existing main prediction path used as the motion/token prior.
```

Inference forward:

```text
raw branch:
  A = null source + null text
  B = source + null text
  C = source + text
  raw_scaled = A + source_scale * (B - A) + cond_scale * (C - B)

latent branch:
  use C branch latent_logits only

edit final:
  final_logits = latent_w * latent_full_logits + raw_w * raw_scaled
```

This keeps latent logits from being directly extrapolated by CFG.

## Config Check

Current config:

```yaml
model:
  use_latent_classifier: True
  delta_alpha: 0.0
  delta_beta: 0.3
  delta_temp: 0.15
  latent_w_min: 0.65
  latent_w_max: 1.0
  raw_w_sum: 1.3

training:
  max_epoch: 250
  m_drop: 0.1

loss:
  weight_delta: 0.5
  weight_latent: 0.1
  weight_rank: 0.03
```

No high-mask parameters are active:

```text
no e_min
no e_max
no e_full
```

## Param Sweep

The first run with:

```yaml
latent_w_min: 0.5
latent_w_max: 0.9
raw_w_sum: 1.4
```

showed unstable early behavior:

```text
early G2S was very high, suggesting the latent classifier starts as a source-attractor.
later metrics moved back toward the old 0.35 - 0.40 basin.
```

Next run strengthens latent dominance:

```yaml
latent_w_min: 0.65
latent_w_max: 1.0
raw_w_sum: 1.3
```

Effective ratios:

```text
mask_ratio = 1.0:
  latent_w = 1.00, raw_w = 0.30

mask_ratio = 0.5:
  latent_w = 0.825, raw_w = 0.475

mask_ratio = 0.0:
  latent_w = 0.65, raw_w = 0.65
```

Purpose:

```text
test whether the latent classifier can break the old basin when it clearly leads the edit decision.
```

## Verification Conclusion

The implementation matches the intended experiment:

```text
1. generation path remains raw main logits
2. edit path uses latent logits as the main classifier
3. raw branch remains as prior through dynamic mixing
4. old residual-logit bias is disabled by delta_alpha = 0.0
5. high-mask experiment is not carried into this run
```
