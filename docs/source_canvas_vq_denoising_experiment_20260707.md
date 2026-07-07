# Source Canvas VQ Denoising Experiment

Date: 2026-07-07

Base:

```text
mix-vq / vq latent + logit change
```

Status:

```text
running / not validated yet
```

Architecture figure:

![Edit Canvas VQ Denoising Architecture](./edit_canvas_vq_denoising_architecture.png)

## Goal

The VQ latent + logit branch can slightly push samples away from source and toward target, but it does not reliably break the edit R@1 ceiling. The likely bottleneck is that training is still mostly target denoising:

```text
corrupt(target tokens) + source + edit text -> target tokens
```

This can let the model learn motion prior and source preservation without learning a strong source-to-target editing transition.

This experiment changes the edit training canvas so the model must rewrite source-like tokens into target tokens:

```text
corrupt(mix(source tokens, target tokens)) + source + edit text -> target tokens
```

Generation remains the normal target-denoising path.

## Kept From VQ Latent + Logit Change

The stable source branch is kept:

```text
masked motion tokens
    -> text_cross(masked motion, text tokens)
    -> source_cross(text_cross, source tokens)
    -> transformer_input = masked motion tokens + source_cross
    -> source control branch(aligned source)
    -> main AdaLN
    -> raw logits
```

The VQ delta branch is also kept:

```text
aligned source ids -> z_src
transformer_input + text_cross + source_cross + z_src
    -> delta encoder
    -> layer-wise delta residuals
    -> pred_z
    -> VQ latent residual logits
```

Final prediction still uses:

```text
final_logits = raw_logits + delta_alpha * latent_residual_logits
```

## New Edit Canvas Training

For editing samples with real source and real text:

```text
edit_progress = (1 - mask_ratio) ^ edit_progress_power
```

Then each valid token chooses its visible canvas source:

```text
early denoising:
  mostly source tokens

late denoising:
  mostly target tokens
```

So the training trajectory becomes:

```text
source-like canvas -> mixed canvas -> target-like canvas
```

This is meant to match iterative BERT refinement better than a one-step source hint.

## CE On Source-Visible Edit Tokens

When a source token is visible in the edit canvas, it is not ignored. The model is still asked to predict the target token there, with a VQ-latent change weight:

```text
edit_weight = normalized || z_tgt - z_src ||
```

Loss mask:

```text
normal predict_mask
+ source-visible edit positions
```

This gives direct pressure:

```text
source token visible + edit text -> target token
```

## Edit Advantage Loss

A light auxiliary term pushes the target token logit above the source token logit:

```text
max(0, margin - (logit_target - logit_source))
```

This is edit-only and weighted by the latent source-target difference.

Current config:

```yaml
weight_edit_adv: 0.2
edit_adv_margin: 0.5
edit_visible_weight: 0.5
```

## Inference Change

For editing samples, generation starts from an aligned source canvas instead of pure mask:

```text
ids[source positions] = aligned_source_ids
scores[source positions] = edit_canvas_score
```

The initial mask ratio is capped:

```yaml
edit_canvas_start_ratio: 0.7
edit_canvas_score: 0.5
```

Source canvas tokens are not hard locked. They fade through the BERT refinement process and can be replaced when the model becomes confident.

Generation samples remain unchanged:

```text
all mask -> target tokens
```

## Current Config

```yaml
edit_canvas_prob: 1.0
edit_progress_power: 1.5

weight_delta: 0.5
weight_latent: 0.02
weight_rank: 0.0
weight_edit_adv: 0.2
edit_visible_weight: 0.5
edit_adv_margin: 0.5

delta_alpha: 0.2
delta_beta: 0.3
```

## Expected Signals

Useful signs:

```text
edit R@1 improves beyond the old 0.40 - 0.43 range
visual motion moves farther from source but remains natural
motion amplitude becomes more complete
gen R@1 does not collapse
```

Failure signs:

```text
edit R@1 stays near the old ceiling
gen R@1 drops strongly
visual output becomes source-copy with small edits
edit_adv decreases but retrieval does not improve
```

## Core Risk

This can help only if the model learns a real rewriting transition:

```text
source canvas + edit text -> target motion
```

If it only learns another noisy corruption type, it may split the training distribution and hurt generation without improving instruction grounding.

## Result Note

This broad source-canvas route failed in later testing:

```text
edit R@1 did not improve beyond the old ceiling,
and generation/motion prior became worse.
```

Do not repeat this exact path.

If source-to-target corruption is revisited, it should be narrower:

```text
1. edit-only training corruption
2. no inference canvas change at first
3. source-token replacement only on selected predicted positions
4. stronger focus on likely changed tokens
5. generation branch untouched
```

This narrower idea is tracked as:

```text
Plan D in docs/next_edit_instruction_experiments_20260707.md
```

## Interpretation Plan

If this improves edit but hurts gen:

```text
reduce edit_canvas_prob or edit_visible_weight,
and keep the canvas path edit-only.
```

If it improves visualization but not R@1:

```text
the method may improve kinematic motion without matching TMR retrieval strongly.
check qualitative samples and global monitor together.
```

If it does not improve either:

```text
the bottleneck is not source initialization or canvas trajectory.
focus next on cleaner text residual responsibility and stronger edit-text supervision.
```

## Result Update

Epoch:

```text
15
```

Validation:

```text
loss: 5.663
vq_delta: 0.030
vq_latent: 6.158
vq_rank: 0.573
edit_adv: 1.140
accuracy: 0.138
```

Generation:

```text
R@1/2/3: 0.4305 / 0.6129 / 0.7251
Matching: 3.4813
FID: 1.0310
Diversity: 9.5379
```

Motion editing:

```text
G2T R@1/2/3: 0.3141 / 0.5000 / 0.6250
G2T AvgR: 4.96
G2S R@1/2/3: 0.3203 / 0.5031 / 0.6016
G2S AvgR: 5.37
TMR-FID: 0.1409
TMR-Diversity: 1.3305
```

Global monitor:

```text
G2T R@1/2/3: 0.1091 / 0.1712 / 0.2379
G2T AvgR: 61.60
G2S R@1/2/3: 0.0636 / 0.1303 / 0.1712
G2S AvgR: 70.56
```

Conclusion:

```text
failed.
Source canvas rewriting did not improve early edit learning and degraded/stressed the shared training distribution.
Do not continue this path as a main strategy.
```

Reasonable takeaway:

```text
source initialization/canvas rewriting is not the missing mechanism.
The next direction should focus on edit text learning source-to-target transition directly.
```
