# Latent Semantic Residual Experiment

## Status

Planned / running. Do not treat as validated until training and eval results are added.

Base branch:

```text
vq-latent / claude rewind
```

This experiment starts from the restored source-branch baseline and adds an edit-only semantic VQ residual path.

## Motivation

Previous attempts showed that the model can use source motion, but edit text is still weak. Source branch and source inpainting can improve preservation, while text-control branches and final-logit margins did not reliably break the edit R@1 ceiling.

The current hypothesis is:

```text
edit text needs a direct path into token decision,
but that path should also train the main AdaLN hidden state.
```

So this experiment makes text-conditioned VQ latent residuals directly affect final logits, while allowing part of the latent loss gradient to flow back into the main branch.

## Architecture

Original source branch is kept:

```text
masked target tokens
    -> text_cross(masked target, text tokens)
    -> source_cross(text_cross, source tokens)
    -> transformer_input = masked target tokens + source_cross
    -> source control branch(aligned source)
    -> main AdaLN
    -> output hidden
    -> raw_logits
```

New semantic VQ residual path:

```text
source_id -> VQ codebook lookup -> z_src
z_src -> latent_MLP -> e_src

output hidden + text_cross + source_cross + e_src
    -> edit_delta_head
    -> pred_delta_e

pred_e = e_src + pred_delta_e
semantic_codebook = latent_MLP(VQ codebook)

latent_logits = cosine(pred_e, semantic_codebook) / tau
src_logits = cosine(e_src, semantic_codebook) / tau
latent_residual_logits = latent_logits - src_logits

final_logits = raw_logits + latent_alpha * latent_residual_logits
```

The subtraction is intentional:

```text
if pred_delta_e = 0, latent residual logits ~= 0
```

This prevents the latent path from simply adding a source-copy bias at initialization.

## Gating

Latent residual logits are active only for:

```text
editing sample
+ real source
+ real text branch
```

Generation samples keep:

```text
final_logits = raw_logits
```

In CFG:

```text
A: null source + null text -> no latent residual
B: source + null text      -> no latent residual
C: source + text           -> latent residual enabled
```

So CFG amplifies the edit-text direction instead of the source-only branch.

## Main-Branch Feedback

The latent residual head receives `output hidden` from main AdaLN. This lets `latent_ce`, `latent_delta`, and `latent_rank` push semantic edit information back into the main branch.

Gradient is controlled by:

```yaml
latent_main_grad: 0.3
```

Implementation:

```python
main_hidden = output.detach() + latent_main_grad * (output - output.detach())
```

This keeps the forward value unchanged but limits how strongly latent losses update the main branch.

## Loss

Existing losses:

```text
CE(final_logits, target token)
+ InfoNCE(target motion tokens, text)
```

Additional losses:

```text
raw edit CE:
  CE(raw_logits, target token) on edit masked tokens

latent CE:
  CE(latent_logits, target token)

latent delta:
  SmoothL1(pred_delta_e, e_tgt - e_src)

latent rank:
  max(0, margin + dist(pred_e, e_tgt) - dist(pred_e, e_src))
```

Current weights:

```yaml
latent_alpha: 0.4
latent_main_grad: 0.3

weight_raw_edit_ce: 0.2
weight_latent_ce: 0.5
weight_latent_delta: 0.5
weight_latent_rank: 0.2
latent_tau: 0.1
latent_rank_margin: 0.2
latent_weight_floor: 0.2
```

## Expected Signals

Early useful signs:

```text
latent_ce decreases
latent_delta decreases
latent_rank decreases
edit G2T R@1 rises beyond the source-branch ceiling
generation R-precision does not collapse
```

If latent losses decrease but edit metrics do not improve:

```text
latent path learns but does not push final token enough
```

Possible next adjustment:

```yaml
latent_alpha: 0.6
```

If generation degrades:

```text
shared main branch is being pulled too much by edit latent losses
```

Possible next adjustment:

```yaml
latent_main_grad: 0.1
weight_latent_delta: 0.25
```

## Related Ideas

This is inspired by:

- source-to-target motion mapping in MotionFix / MotionLab-style editing
- similarity / semantic representation guidance in SimMotionEdit
- latent semantic alignment ideas from MotionCLIP-like methods
- residual conditional control from ControlNet-like designs

The key difference from previous failed branches is:

```text
the text-conditioned latent residual directly enters final token logits
and latent losses can update the main AdaLN hidden state.
```

## Result

Failed.

Best observed around epoch 18:

```text
validation loss: 6.973
latent_ce: 4.425
latent_delta: 0.009
latent_rank: 0.274
accuracy: 0.164

Gen R@1/2/3: 0.4401 / 0.6084 / 0.7360
Gen Matching: 3.4306
Gen FID: 1.0255
Gen Diversity: 9.6787

Edit G2T R@1/2/3: 0.2984 / 0.4656 / 0.5531
Edit G2T AvgR: 5.22
Edit G2S R@1/2/3: 0.2500 / 0.4016 / 0.5125
Edit G2S AvgR: 6.11
TMR-FID: 0.1659
TMR-Diversity: 1.2993

Global monitor G2T R@1/2/3: 0.0742 / 0.1545 / 0.2015
Global monitor G2S R@1/2/3: 0.0591 / 0.0894 / 0.1318
```

Outcome:

```text
edit R@1 stayed around 0.30 and was worse than the claude-rewind source-branch baseline.
generation was acceptable, but editing quality did not improve.
```

Failure interpretation:

- Semantic VQ residual logits did not break the source/text bottleneck.
- Letting latent losses flow into main AdaLN with `latent_main_grad=0.3` did not make the main branch learn instruction semantics.
- `latent_delta` stayed very small, so the residual supervision was weak in practice.
- `latent_ce` decreased to a reasonable scale, but it did not translate into better editing retrieval.
- The design likely remained a late logit correction instead of changing how the model reasons about source-to-target edits.

Do not repeat this exact design without a major change, such as:

- a stronger latent objective that actually separates target/source edit directions,
- a direct edit action representation before token decoding,
- or a training objective closer to source-to-target flow / denoising rather than late token residual logits.
