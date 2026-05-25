import numpy as np


ACTION_OUTPUT_TYPES = {
    "l1": "env_action",
    "ce": "logits",
    "factorized_ce": "logits",
}


def postprocess_actions(normalized: np.ndarray, processor, loss_type: str) -> np.ndarray:
    output_fns = {
        "l1": lambda: np.stack(
            [processor.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        ),
        "ce": lambda: normalized,
        "factorized_ce": lambda: normalized,
    }
    return output_fns[loss_type]()
