"""
# A python module that manipulates torch checkpoint file in a hacky way.
Each function should be used with caution and should be used only when thoughtfully considered.
---
Args:
    source_state_dict: the state_dict loaded using torch.load
    algo_state_dict: the algorithm state_dict summarized from algorithm as an example
---
Returns:
    new_state_dict: the state_dict that has been manipulated or directly saved as a checkpoint file.
"""
import torch
from collections import OrderedDict


def reinitialize_height_encoders(source_state_dict, algo_state_dict):
    """Keep the walking policy while reinitializing both height encoders."""
    encoder_prefixes = ("encoders.0.", "critic_encoders.0.")
    source_model = source_state_dict["model_state_dict"]
    target_model = algo_state_dict["model_state_dict"]

    source_keys = set(source_model)
    target_keys = set(target_model)
    if source_keys != target_keys:
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        raise KeyError(
            "Checkpoint model keys do not match the target model. "
            f"Missing: {missing}; unexpected: {unexpected}."
        )

    encoder_keys = [
        key for key in target_model if key.startswith(encoder_prefixes)
    ]
    for prefix in encoder_prefixes:
        if not any(key.startswith(prefix) for key in encoder_keys):
            raise KeyError(f"Target model has no parameters under {prefix!r}.")

    new_model_state_dict = OrderedDict()
    for key, target_value in target_model.items():
        source_value = source_model[key]
        if key.startswith(encoder_prefixes):
            new_model_state_dict[key] = target_value
            continue
        if source_value.shape != target_value.shape:
            raise ValueError(
                f"Non-encoder parameter {key!r} has incompatible shapes: "
                f"checkpoint {tuple(source_value.shape)} versus target "
                f"{tuple(target_value.shape)}."
            )
        new_model_state_dict[key] = source_value

    print(
        "\033[1;36m Reinitialized Actor/Critic height encoders; "
        "kept all other checkpoint model parameters. \033[0m"
    )
    return dict(
        model_state_dict=new_model_state_dict,
        iter=source_state_dict["iter"],
        infos=source_state_dict.get("infos"),
    )

def replace_encoder0(source_state_dict, algo_state_dict):
    print("\033[1;36m Replacing encoder.0 weights with untrained weights and avoid critic_encoder.0 \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if "critic_encoders.0" in key:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
        elif "encoders.0" in key:
            print(
                "key:", key,
                "shape:", algo_state_dict["model_state_dict"][key].shape,
                "using untrained module weights.")
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict= new_model_state_dict,
        # No optimizer_state_dict
        iter= source_state_dict["iter"],
        infos= source_state_dict["infos"],
    )
    return new_state_dict
