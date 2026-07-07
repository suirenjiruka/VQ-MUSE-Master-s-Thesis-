# Next Edit Instruction Experiments

Date: 2026-07-07

Base to start from:

```text
best stable VQ latent + logit change / claude rewind style source branch
```

Do not start from source canvas unless it unexpectedly improves. Source canvas is currently considered a risky side path.

## Problem To Attack

The current model can learn motion prior and can use source motion, but edit text is still weak. The main failure pattern is:

```text
source branch gives plausible motion
text branch only nudges it
output moves slightly away from source but does not fully follow instruction
```

So the next experiments should directly test:

```text
can edit text learn source -> target change?
```

## Plan A: MotionFix-Only Edit Finetune

Purpose:

```text
check whether mixed HML3D generation data dilutes edit instruction learning
```

Training:

```text
start from current best stable checkpoint
use only MotionFix / id >= 400000 editing samples
keep source branch and VQ latent + logit design
disable generation samples during this stage
```

Implemented config:

```yaml
training:
  edit_only_e: 50
```

Current behavior:

```text
epoch 0 - 49:
  train on MotionFix/editing ids only
  validate on editing ids only
  run motion-edit evaluator only
  skip generation evaluator

epoch 50+:
  restore mixed HML3D + MotionFix training loader
  restore mixed validation loader
  run both generation and motion-edit evaluators
```

Checkpoint usage:

```text
--OnGoing_model path/to/best.tar loads model weights only and restarts this curriculum from epoch 0.
cfg.exp.is_continue=True still means full latest.tar resume with optimizer/scheduler/epoch.
```

Expected interpretation:

```text
if edit R@1 breaks 0.45:
  mixed training ratio is a major bottleneck

if edit stays around 0.40:
  bottleneck is model/representation, not only dataset mixture
```

After MotionFix-only stabilizes:

```text
mix HML3D back with lower LR
keep edit sampling ratio high
monitor gen R@1/FID and edit G2T together
```

Do not treat this as two separate final models. It is curriculum training for the same shared model.

## Plan B: Text-Conditioned Token Transition Prior

Purpose:

```text
make edit text directly learn source token -> target token change
```

This is different from the previous VQ latent residual branch. The old branch was:

```text
mixed hidden + source_cross + text_cross + z_src
  -> pred_delta_z
  -> small residual bias
```

It may be too source/main-branch dominated.

The new branch should be cleaner:

```text
source_id / z_src
+ pure edit text feature
+ scale or level embedding
+ optional current mask-ratio embedding
  -> transition_logits over VQ codebook
```

Loss:

```text
L_transition = CE(transition_logits, target_id)
L_margin = max(0, margin - (logit_target - logit_source))
```

Inference integration:

```text
transition_residual_logits =
    transition_logits(edit text)
  - transition_logits(null text)

final_logits = main_logits + alpha * transition_residual_logits
```

Important design rule:

```text
the branch should not depend heavily on source_cross or main hidden
```

Otherwise it can repeat the old failure: source-dominant safe correction instead of text-driven edit.

## Priority

Run order:

```text
1. MotionFix-only finetune
2. Text-conditioned token transition prior
```

Reason:

```text
MotionFix-only is cheaper and tells us whether the edit signal is diluted by mixed training.
The transition prior is the stronger architectural change if MotionFix-only shows potential.
```

## Stop Rules

For MotionFix-only:

```text
if edit R@1 does not improve beyond 0.42 - 0.44 after a reasonable warmup,
do not spend a full long run.
```

Result:

```text
failed.
```

Epoch:

```text
52
```

Validation:

```text
loss: 5.967
vq_delta: 0.042
vq_latent: 6.020
vq_rank: 0.401
accuracy: 0.141
```

Generation after mixed data returns:

```text
R@1/2/3: 0.3399 / 0.5357 / 0.6582
Matching: 4.1006
FID: 2.8583
Diversity: 9.0803
```

Motion editing:

```text
G2T R@1/2/3: 0.3563 / 0.5250 / 0.6453
G2T AvgR: 4.38
G2S R@1/2/3: 0.3516 / 0.5312 / 0.6297
G2S AvgR: 4.43
TMR-FID: 0.1508
TMR-Diversity: 1.3107
```

Conclusion:

```text
MotionFix-only curriculum did not improve edit instruction learning.
It also damaged generation/motion prior after returning to mixed training.
Mixed HML3D data is not merely diluting edit signal; it also regularizes motion prior and text-motion alignment.
```

Next:

```text
do not continue MotionFix-only as the main path.
move to text-conditioned source_id -> target_id transition prior.
```

For transition prior:

```text
if transition CE decreases but edit R@1 and visualization do not improve,
the branch is learning token statistics but not useful motion semantics.
```

## What Not To Repeat

Avoid repeating these as main strategies:

```text
1. another generic text branch without direct token transition responsibility
2. final-logit margin only
3. edit-map auxiliary that is not used in inference
4. heavy source canvas rewriting unless current run proves otherwise
```

## Plan C: Text Gain + Edit Effect Calibration

Status:

```text
active experiment.
```

Base:

```text
claude rewind source branch
```

Reason:

```text
visualization suggests text is not ignored,
but its effect is weak, conservative, and not accurate enough.
```

Change:

```text
do not add another text branch.
amplify the existing text_cross / AdaLN text path for edit samples.
train final logits to produce the correct source -> target VQ-code effect.
```

Loss:

```text
CE(final_logits, target_ids)
+ edit-effect direction/magnitude loss from final logits
+ small correct-vs-shuffled text ratio loss
```

Detailed record:

```text
docs/text_gain_edit_effect_experiment_20260707.md
```

## Plan D: Edit-Specific Source-To-Target Corruption

Status:

```text
idea only, not implemented.
```

Motivation from claude rewind static analysis:

```text
editing training is still mostly target-token denoising.
When many target tokens are visible, the model can use target context + source prior
instead of learning source + instruction -> target transformation.
```

Important caution:

```text
This is not the same as the failed source canvas experiment.
The old source canvas path was too broad and hurt generation/motion prior.
If this is tested, it must be narrow, edit-only, and designed to remove the target-context shortcut.
```

Proposed training change:

```text
generation samples:
  keep normal MoMask target denoising

editing samples:
  for selected predicted positions,
  replace current target input token with aligned source token,
  still train CE to target token.
```

Purpose:

```text
force the main prediction path to see source-like current tokens
and learn how edit text transforms them into target tokens.
```

Key difference from old source canvas:

```text
1. no inference path change at first
2. no full source-canvas trajectory
3. only selected edit training positions are source-corrupted
4. changed-token positions should be sampled more often than unchanged positions
5. generation samples are untouched
```

Risk:

```text
If source/target changed-token localization is noisy in VQ ID space,
this may still become another corruption type rather than true edit learning.
```

Stop rule:

```text
If edit R@1 and visualization do not improve within early epochs,
drop it and do not repeat source-canvas variants.
```

## Other Candidate Ideas

### Idea E: Text-Conditioned Output Classifier

Status:

```text
candidate if Plan C/D fail.
```

Reason:

```text
Current text enters as cross-attention/AdaLN modulation,
but the final token classifier is static.
If text is always a weak hidden-state perturbation,
make the classifier itself text-conditioned.
```

Sketch:

```text
hidden -> text-conditioned scale/shift -> output logits
```

or:

```text
classifier_weight = W + low_rank(text) 
```

This changes the main decision surface, not just hidden residuals.

Risk:

```text
Can hurt generation if applied globally.
Should be gated edit-only or use small low-rank adaptation.
```

### Idea F: Changed-Token Focused Training

Status:

```text
candidate.
```

Reason:

```text
MotionFix edits may change only a minority of tokens.
Normal CE is dominated by easy unchanged / motion-prior tokens.
```

Sketch:

```text
use VQ-id or latent-distance change mask
increase CE weight on changed tokens
reduce weight on unchanged tokens
```

This is simpler than source canvas and directly attacks gradient dilution.

Risk:

```text
VQ-id change is noisy.
Need latent-distance or multi-scale smoothing, not raw id mismatch only.
```

### Idea G: Source Branch Temperature / Capacity Control

Status:

```text
candidate.
```

Reason:

```text
Source branch is much stronger than text path.
If text is consistently suppressed, reduce source dominance during edit training.
```

Sketch:

```text
randomly weaken source control residuals on edit samples
or apply source residual dropout per layer
```

Goal:

```text
make text necessary without removing source completely.
```

Risk:

```text
Too much source weakening hurts preservation and motion naturalness.
```

## Current Priority After Static Analysis

Static diagnosis of claude rewind suggests the bottleneck is not that text is completely ignored. It is:

```text
text is a weak likelihood shift,
while source prior + visible target context + motion prior can explain most CE.
```

So the next ideas should be judged by whether they change this:

```text
Does the method make edit text necessary for predicting target tokens?
Does it reduce easy target-context/source-prior shortcuts?
Does it improve changed-token gradients without damaging generation?
```

Current priority:

```text
1. Plan C: Text Gain + Edit Effect Calibration
   already active; tests whether existing text path is enough if supervised better.

2. Idea F: Changed-Token Focused Training
   cleanest next step; attacks CE dilution without changing inference or input distribution.

3. Idea E: Text-Conditioned Output Classifier
   strongest architectural next step; changes the main token decision surface.

4. Plan D: Narrow Edit-Specific Source-To-Target Corruption
   plausible but risky; old broad source canvas failed, so only try if done narrowly.

5. Idea G: Source Branch Capacity Control
   useful as a stabilizer or ablation, not a standalone breakthrough.
```

## Decision Rules

If Plan C improves visualization/R@1:

```text
keep text gain / edit-effect loss,
then try Idea F with low weight to sharpen changed-token learning.
```

If Plan C lowers edit_effect but edit R@1 stays flat:

```text
loss signal is not enough.
move to Idea E because the main classifier likely needs explicit text conditioning.
```

If Plan C hurts generation:

```text
reduce text_gain / adaln_text_gain first.
If generation remains unstable, gate gains/edit losses more strictly to edit samples.
```

If Idea F is tested:

```text
do not use raw source_id != target_id alone as a hard changed mask.
Use VQ latent/codebook distance or smoothed multi-scale change weight.
```

If Plan D is tested:

```text
do not change inference first.
do not use full source canvas.
only replace selected edit-training input tokens with aligned source tokens.
stop early if it behaves like the failed source canvas run.
```

## Key Lessons To Preserve

Do not re-propose these as new main ideas:

```text
1. generic extra text branch
2. auxiliary-only edit-map/sim/part loss
3. final-logit bias without changing main decision responsibility
4. broad source canvas denoising
5. MotionFix-only curriculum
6. heavy VQ latent delta branch as the main solution
```

Keep this framing:

```text
The target is not just stronger text representation.
The target is making correct edit text necessary for source -> target token prediction.
```
