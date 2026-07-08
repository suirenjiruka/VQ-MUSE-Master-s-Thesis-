# Same-source contrast experiment

## Why

MotionFix audit shows 49.4% of edit pairs live in genuine multi-edit groups:

```
same source + text A -> target A
same source + text B -> target B
```

So text is not redundant. The previous failure is likely that training did not explicitly use this same-source disambiguation signal.

## Difference from failed trials

- Not random shuffled text.
- Not a new text branch.
- Not latent-logit driven output.
- The wrong text comes from a real edit pair with the same source motion but a different target/text.

## Design

Dataset builds a source hash from MotionFix `source` motion content.
For each edit sample, it samples one hard negative caption from:

```
same source
different target
different text
```

Model adds an inline margin loss under an all-mask state:

```
logp(target_i | source_i, text_i) > logp(target_i | source_i, wrong_text_same_source)
```

This keeps source fixed, so text is the only variable explaining the target.

The loss is only computed on changed tokens after aligning the hard-negative target to the current target timeline:

```
target_i_token != target_j_token
```

Shared tokens are ignored because both texts should predict the same token there.

For each same-source pair, the model runs two all-mask text branches:

```
f(source, text_i)
f(source, text_j)
```

These two logits score both `target_i` and aligned `target_j`, giving a symmetric hard-negative signal without target leakage.

## Config

```
weight_same_src: 0.5
same_src_margin: 1.0
same_src_max: 16
```

`same_src_max` limits the extra wrong-text branch per batch to control speed.

## Metrics

- `same_src`: margin loss, lower is better.
- `same_gap`: correct text target logp minus same-source wrong text target logp, higher is better.

If `same_gap` does not open on train, text authority/architecture is likely still too weak.
If train opens but eval does not, the next issue is language/generalization, then paraphrase or in-context text should be tested.
