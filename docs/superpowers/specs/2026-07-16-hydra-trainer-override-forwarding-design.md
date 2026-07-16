# Hydra Trainer Override Forwarding Design

## Objective

Allow the RL-games launcher to forward a dynamically added `trainer.*` leaf through its second Hydra composition without losing the field-addition semantics.

## Root Cause

WanOFT command wrappers add `trainer.learning_rate.action_query_proj` during the launcher's first Hydra composition. The launcher then serializes the resolved configuration into a command for `train_starvla_hydra.py`. During serialization, it removes the Hydra override prefix from every `trainer.*` leaf. The second composition therefore treats the added learning-rate group as an override of a declared field and rejects it in struct mode.

## Design

Every serialized leaf override will use Hydra's `++` operator. This operator updates an existing field or adds a missing field, so it is idempotent across the launcher's two composition stages. The change restores the launcher's original generic forwarding behavior and keeps module-specific optimizer groups extensible.

The WanOFT command and model configuration remain unchanged. `action_query_proj` continues to receive its explicitly requested learning rate instead of falling into the base optimizer group.

## Validation

A regression test will:

1. compose the WanOFT Demon Attack configuration with a newly added `trainer.learning_rate.action_query_proj` field;
2. build the inner trainer command;
3. assert that the field is forwarded with `++`;
4. compose the inner overrides again and verify that the learning rate is preserved.

Existing RL-games command and WanOFT tests will run locally. Full model training remains remote-only.

## Publishing

The same reviewed implementation commit will be applied to both remote branches: `master` and `WanOFT`. No compatibility path, fallback, or unrelated refactor is included.
