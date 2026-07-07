# Text Gain + Edit Effect Calibration

Date: 2026-07-07

Base:

```text
claude rewind source branch
```

## Goal

Current visual diagnosis:

```text
text is not dead.
It pushes motion toward a related target direction,
but the edit is weak, conservative, and often inaccurate.
```

So this experiment does not add another text branch. It tests whether the existing text path can be made stronger and trained more directly.

## Core Change

No new branch:

```text
no new text encoder branch
no extra ControlNet copy
no final external text logits branch
```

Keep the claude rewind path:

```text
masked target tokens
  -> text_cross
  -> source_cross
  -> source control branch
  -> main AdaLN
  -> token logits
```

Add text gain only inside the existing path:

```text
transformer_input =
    motion_tokens
  + source_cross
  + (text_gain - 1) * (text_cross - motion_tokens)
```

AdaLN text condition is also lightly amplified for edit samples:

```text
cond = text_cross * adaln_text_gain
```

Both gains are gated by real editing samples, so generation samples stay structurally close to the old path.

## Loss

Main loss stays:

```text
CE(final_logits, target_ids)
InfoNCE(target_motion, text)
```

New edit-only loss uses final logits directly:

```text
p = softmax(final_logits)
pred_z = sum_k p(k) * codebook_z(k)

delta_pred = pred_z - z_src
delta_gt   = z_tgt - z_src
```

Edit-effect calibration:

```text
L_effect =
    direction loss on changed tokens only
  + effect_mag * magnitude loss on valid tokens
```

Implementation detail:

```text
direction loss uses target_id != source_id only.
unchanged tokens have delta_gt = 0, so direction is undefined.
unchanged tokens are handled by magnitude loss, which pushes ||delta_pred|| toward 0.
```

Limit:

```text
pred_z is a token-codebook proxy from softmax(logits) @ codebook.
It is not the full HRVQ reconstruction latent, because HRVQ combines multi-scale residual codes.
So edit_effect should be read as a token-level edit-direction signal, not an absolute motion-latent fidelity metric.
```

Small counterfactual text ratio:

```text
L_ratio =
  -log sigmoid((logit_correct[target] - logit_correct[source]) / tau)
  -log sigmoid((logit_correct[target] - logit_wrong[target]) / tau)
```

This is not the main signal. It only checks that correct text pushes target more than source-copy or shuffled text.

## Config

```yaml
model:
  text_gain: 1.5
  adaln_text_gain: 1.2

loss:
  weight_edit_effect: 0.3
  weight_text_ratio: 0.05
  effect_mag: 0.5
  text_ratio_tau: 0.2
```

## Why This Is Different From Failed Branch Designs

Failed directions often had:

```text
new branch -> residual/logit bias -> main model can ignore it
```

This experiment instead:

```text
1. keeps the same main path
2. increases existing text signal strength
3. applies edit-effect loss to final logits
```

So if it fails, the interpretation is clearer:

```text
the existing text path capacity is not enough,
and stronger architecture-level text conditioning is required.
```

## Expected Signal

Good sign:

```text
edit_effect decreases
edit R@1 improves beyond old 0.40 - 0.43 range
visual motion has larger and more complete edit amplitude
gen R@1 does not collapse
```

Bad sign:

```text
edit_effect decreases but edit R@1 stays flat
text_ratio decreases but visualization remains conservative
gen drops strongly
```

Interpretation if bad:

```text
loss-only and gain-only correction is insufficient.
Next step must change the main classifier or text-conditioning architecture itself.
```

## Result

Run result:

```text
epoch 21
val loss: 5.450
edit_effect: 0.725
text_ratio: 4.836
accuracy: 0.174

Gen R@1/2/3: 0.4483 / 0.6403 / 0.7526
Gen Matching: 3.4004
Gen FID: 0.9870

Edit G2T R@1/2/3: 0.3047 / 0.4813 / 0.5922
Edit G2S R@1/2/3: 0.2594 / 0.4375 / 0.5391
TMR-FID: 0.1605
```

Conclusion:

```text
failed.
```

Reason:

```text
text gain and edit-effect loss made the training heavier,
but did not make correct edit text a reliable decision factor.
Edit R@1 dropped far below the 0.40 range.
```

Interpretation:

```text
1. simply amplifying the existing text_cross path is not enough
2. final-logit latent/effect supervision is still auxiliary
3. the model can keep using source + motion prior + visible target context
4. text signal becomes noisy rather than precise
```

Do not repeat this exact design unless the main decision path is changed.
