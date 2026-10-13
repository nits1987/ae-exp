# Improve a small character-level name model

Build a better next-character model for the deduplicated names recorded in the
frozen data manifest. The current corpus comes from the combined common dataset
(merging supplied ranked names in data.txt with Karpathy's makemore names.txt,
~29,495 unique names). The seed is an embedding plus MLP with an eight-character
causal context. PyTorch trains the weights; evolve only the architecture in the
single EVOLVE-BLOCK.

Contract: build_model(config) returns a torch.nn.Module. forward(context) accepts
CPU int64 tensors [B, 8] and returns finite unnormalized logits [B, vocab_size].
The config contains vocab_size, embedding_size=32, hidden_size=128, and
context_length=8. ID 0 is padding/start and the end-of-name target. Characters
have IDs 1..vocab_size-1. No future characters or targets are passed to the model.

Allowed: hidden widths/depths, nonlinearities, normalization, residual blocks,
gating, and embedding changes. Only torch and math imports. No file/network I/O,
subprocesses, runtime introspection, pretrained weights, dataset lookup tables,
environment access, or changes outside the evolution block. The model must be
stateless between independent batches in evaluation mode.

Frozen training: 2,000 AdamW updates, batch 128, learning rate .001, weight decay
.01, seeds 3407 and 42. Each seed starts from scratch. The evaluator owns all
training, targets, and scoring; do not compute or return your own loss/score.

Maximize exp(-mean validation NLL), where NLL is summed over real character
targets including EOS and divided by their count, then averaged across seeds.
Constraints: 1..200,000 parameters and batch-one CPU inference p95 no greater
than the calibrated latency cap (up to 10x baseline / 0.60ms floor on CPU to accommodate
per-operator runtime dispatch overhead, 2x on GPU). Evaluation insights report
val_NLL, train_NLL, overfit_gap (val_NLL - train_NLL), parameter usage, and latency
headroom so you can balance model capacity against generalization. Two CPU threads,
sequential evaluations. Invalid or timed-out candidates fail; they still consume
the generation allowance. Prefer architectures (e.g., WaveNet-style hierarchical
mixing, causal convolutions, gated/residual MLPs, normalization, dropout) that
improve validation NLL under the fixed 2,000-step training allowance.

The final test split is never used for search. The final experiment compares
the evolved model against the seed and ordinary architecture tuning. A better
validation score is evidence for further testing, not a guarantee of improvement.
