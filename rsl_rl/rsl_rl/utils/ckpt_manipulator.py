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


def reset_optimizer_state(source_state_dict, algo_state_dict):
    """Keep every model parameter while discarding optimizer state."""
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

    new_model_state_dict = OrderedDict()
    for key, target_value in target_model.items():
        source_value = source_model[key]
        if source_value.shape != target_value.shape:
            raise ValueError(
                f"Parameter {key!r} has incompatible shapes: checkpoint "
                f"{tuple(source_value.shape)} versus target "
                f"{tuple(target_value.shape)}."
            )
        new_model_state_dict[key] = source_value

    print(
        "\033[1;36m Kept all checkpoint model parameters; "
        "reset optimizer and scheduler state. \033[0m"
    )
    return dict(
        model_state_dict=new_model_state_dict,
        iter=source_state_dict["iter"],
        infos=source_state_dict.get("infos"),
    )


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


def expand_height_encoder_inputs(
    source_state_dict,
    algo_state_dict,
    source_grid_shape=(21, 11),
    target_grid_shape=(36, 17),
):
    """Embed old height-grid inputs into a larger aligned encoder input."""
    encoder_prefixes = ("encoders.0.", "critic_encoders.0.")
    input_weight_keys = tuple(
        prefix + "model.0.weight" for prefix in encoder_prefixes
    )
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
    for key in input_weight_keys:
        if key not in target_model:
            raise KeyError(f"Target model has no encoder input weight {key!r}.")

    source_x, source_y = map(int, source_grid_shape)
    target_x, target_y = map(int, target_grid_shape)
    if min(source_x, source_y, target_x, target_y) <= 0:
        raise ValueError("Height-grid dimensions must be positive.")
    if target_x < source_x or target_y < source_y:
        raise ValueError("Target height grid must contain the source grid.")
    lateral_padding = target_y - source_y
    if lateral_padding % 2 != 0:
        raise ValueError(
            "Target height grid must add equal padding on both lateral sides."
        )
    target_y_start = lateral_padding // 2

    new_model_state_dict = OrderedDict()
    for key, target_value in target_model.items():
        source_value = source_model[key]
        if key in input_weight_keys:
            expected_source_shape = (
                target_value.shape[0],
                source_x * source_y,
            )
            expected_target_shape = (
                target_value.shape[0],
                target_x * target_y,
            )
            if tuple(source_value.shape) != expected_source_shape:
                raise ValueError(
                    f"Encoder input {key!r} has source shape "
                    f"{tuple(source_value.shape)}, expected "
                    f"{expected_source_shape}."
                )
            if tuple(target_value.shape) != expected_target_shape:
                raise ValueError(
                    f"Encoder input {key!r} has target shape "
                    f"{tuple(target_value.shape)}, expected "
                    f"{expected_target_shape}."
                )

            expanded_value = torch.zeros_like(target_value)
            expanded_grid = expanded_value.reshape(
                target_value.shape[0], target_x, target_y
            )
            source_grid = source_value.to(
                device=target_value.device,
                dtype=target_value.dtype,
            ).reshape(source_value.shape[0], source_x, source_y)
            expanded_grid[
                :, :source_x, target_y_start : target_y_start + source_y
            ] = source_grid
            new_model_state_dict[key] = expanded_value
            continue

        if source_value.shape != target_value.shape:
            raise ValueError(
                f"Parameter {key!r} has incompatible shapes: checkpoint "
                f"{tuple(source_value.shape)} versus target "
                f"{tuple(target_value.shape)}."
            )
        new_model_state_dict[key] = source_value

    print(
        "\033[1;36m Expanded Actor/Critic height encoder inputs; "
        "preserved the aligned source grid and all other model parameters. "
        "\033[0m"
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
