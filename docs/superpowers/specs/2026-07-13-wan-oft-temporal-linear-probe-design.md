# WanOFT Temporal Linear Probe Design

## Objective

Determine whether WanOFT SFT makes Flappy Bird timing information more linearly decodable from the frozen Wan representation. The experiment compares the released pre-SFT WanOFT checkpoint with a post-SFT checkpoint under one fixed protocol. It does not train or modify the policy.

## Experimental Unit

Each example is one converted Flappy decision row containing:

- the exact five-frame clip used by WanOFT, ordered `[t-4, t-3, t-2, t-1, t]`;
- the exact training prompt and state;
- `episode_index`, `frame_index`, and the original `decision_step`;
- `action_id` and `latency`.

Train and validation examples come from the existing converted train and validation directories. Episode selection is deterministic and independent of labels. The two checkpoints and every control use the same selected rows in the same order.

`decision_step` is required because `clean_v1` removes background rows from the training split. A re-numbered `frame_index` cannot measure real time between flap events. The Flappy converter therefore preserves the original decision step as an additional column without changing any existing training field.

## Frozen Features

For every batch, the probe path reproduces WanOFT preprocessing, including the configured image resize, state tokens, Wan VAE encoding, clean timestep, Wan Transformer forward pass, action-query projection, and action head. All parameters are frozen and inference runs without gradients.

The five input frames produce two Wan temporal token groups. The implementation validates this from the VAE latent shape and the transformer's temporal patch size instead of assuming a token layout.

Four feature families are extracted:

1. `vae_temporal_delta`: spatially pooled VAE groups `concat(z0, z1, z1-z0)`. This is an input-representation control because the VAE is not SFT-supervised.
2. `dit_global_mean`: the global mean of the final Wan Transformer tokens, matching the default WanOFT pooling source before projection.
3. `dit_temporal_delta`: spatially pooled final Wan groups `concat(h0, h1, h1-h0)`. This is the main temporal representation.
4. `action_query`: the first projected action query consumed by the action MLP. This tests whether timing information reaches the policy bottleneck.

Current action logits are retained only for direct policy classification metrics; they are not probe inputs.

Wan VAE sampling is stochastic. Feature extraction runs inside a forked RNG context with a deterministic per-batch seed. The same latent-noise sequence is therefore used for pre-SFT, post-SFT, and input controls without changing global RNG state.

## Labels

The fixed probe suite contains:

- `current_action`: `NOOP` or `FLAP` from the current row;
- `time_to_next_flap`: exact decision-step distance to the next flap, bucketed as `0, 1, 2, 3, 4+`;
- `time_since_last_flap`: exact decision-step distance from the previous flap, bucketed as `0, 1, 2, 3, 4+`;
- `latency_id`: the recorded environment latency.

Distance zero means that the current action is flap. Missing previous or next events belong to `4+`. A latency probe is mathematically unavailable for a single-latency dataset; the report records that status and reason explicitly instead of emitting a score.

## Controls

The complete suite extracts four input conditions:

- `normal`: original frames and prompt;
- `shuffled_frames`: a deterministic per-example permutation of all five frames;
- `repeated_last_frame`: five copies of frame `t`;
- `latency_neutral_prompt`: original frames with the explicit latency sentence removed using the repository's existing prompt rule.

Frame controls test whether decodability depends on temporal evidence rather than the current image. The neutral-prompt control separates visual latency evidence from direct textual disclosure.

## Linear Probe

Every checkpoint, input condition, feature family, and evaluable label receives an independent multinomial linear classifier. Features are standardized using train statistics only. The classifier uses fixed AdamW hyperparameters, inverse-frequency train class weights, a fixed epoch count, and a fixed seed. Hyperparameters are shared by all comparisons and no checkpoint is selected from validation performance.

Validation reports:

- macro-F1 as the primary metric;
- balanced accuracy, accuracy, and confusion matrix;
- per-class precision, recall, F1, and support;
- direct action-logit metrics;
- score deltas from `normal` to each control;
- pre-SFT to post-SFT score deltas under matching conditions.

The report also includes label distributions, selected episode files, feature dimensions, checkpoint paths, extraction seeds, and probe hyperparameters.

## Storage And Failure Semantics

The pipeline processes one checkpoint and one condition at a time. Features remain in CPU memory only long enough to train the corresponding probes. It writes compact probe state files and JSON reports, then releases model and feature memory. Large intermediate feature caches are not persisted.

The run fails explicitly when:

- a required parquet column or task prompt is missing;
- train and validation sample identities differ across conditions;
- the clip does not contain five frames;
- Wan does not produce exactly two temporal token groups;
- hidden token counts do not match the VAE/patch-derived layout;
- a required timing class is absent from the selected probe training data;
- checkpoint keys do not match the configured WanOFT architecture.

## Repository Boundaries

The implementation adds an isolated probing package and command entry point. Existing WanOFT training and rollout evaluation behavior is unchanged. The only shared data-path change is preservation of `decision_step` in newly converted Flappy parquet files.

Local validation covers label construction, frame controls, temporal token grouping, metric calculations, deterministic linear-probe training, converter schema preservation, and CLI syntax. Full feature extraction is deferred to the remote GPU environment.
