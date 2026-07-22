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
import copy
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


def reset_critic_and_optimizer(source_state_dict, algo_state_dict):
    """Keep the complete Actor side while reinitializing all Critic modules."""
    critic_prefixes = ("critic.", "memory_c.", "critic_encoders.")
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
    for prefix in critic_prefixes:
        if not any(key.startswith(prefix) for key in target_model):
            raise KeyError(f"Target model has no parameters under {prefix!r}.")

    new_model_state_dict = OrderedDict()
    for key, target_value in target_model.items():
        source_value = source_model[key]
        if key.startswith(critic_prefixes):
            new_model_state_dict[key] = target_value
            continue
        if source_value.shape != target_value.shape:
            raise ValueError(
                f"Preserved parameter {key!r} has incompatible shapes: "
                f"checkpoint {tuple(source_value.shape)} versus target "
                f"{tuple(target_value.shape)}."
            )
        new_model_state_dict[key] = source_value

    print(
        "\033[1;36m Kept Actor, Actor memory, encoders, estimator, and "
        "action noise; reinitialized all Critic modules and optimizer state. "
        "\033[0m"
    )
    return dict(
        model_state_dict=new_model_state_dict,
        iter=source_state_dict["iter"],
        infos=source_state_dict.get("infos"),
    )


def reset_one_box_critic_from4000(source_state_dict, algo_state_dict):
    """Reset the one-box Critic while preserving curriculum and reference."""
    if int(source_state_dict.get("iter", -1)) != 4000:
        raise ValueError(
            "The one-box Critic reset requires the accepted model_4000.pt."
        )
    task_state = source_state_dict.get("task_curriculum_state_dict")
    if not isinstance(task_state, dict):
        raise ValueError("The one-box checkpoint has no curriculum state.")
    if (
        int(task_state.get("version", -1)) != 3
        or int(task_state.get("stage", -1)) != 3
        or abs(float(task_state.get("landing_blend", -1.0)) - 0.4) > 1e-6
    ):
        raise ValueError(
            "The one-box Critic reset requires Stage 3 at landing blend 0.4."
        )
    if source_state_dict.get("reference_model_state_dict") is None:
        raise ValueError(
            "The one-box checkpoint has no frozen walking reference policy."
        )

    reset_state = reset_critic_and_optimizer(
        source_state_dict, algo_state_dict
    )
    migrated = copy.deepcopy(source_state_dict)
    migrated["model_state_dict"] = reset_state["model_state_dict"]
    migrated.pop("optimizer_state_dict", None)
    migrated.pop("lr_scheduler_state_dict", None)
    print(
        "\033[1;36m Preserved the model_4000 Actor, one-box curriculum, "
        "and frozen walking reference; reset all Critic modules and the "
        "optimizer. \033[0m"
    )
    return migrated


def initialize_one_box_from_rough2000(
    source_state_dict,
    algo_state_dict,
    source_grid_shape=(21, 11),
    target_grid_shape=(36, 17),
):
    """Expand the rough Actor scan and reset the complete Critic side."""
    if int(source_state_dict.get("iter", -1)) != 2000:
        raise ValueError(
            "One-box initialization requires the rough model_2000.pt "
            "checkpoint."
        )

    critic_prefixes = ("critic.", "memory_c.", "critic_encoders.")
    actor_input_key = "encoders.0.model.0.weight"
    source_model = source_state_dict["model_state_dict"]
    target_model = algo_state_dict["model_state_dict"]
    source_keys = set(source_model)
    target_keys = set(target_model)
    if source_keys != target_keys:
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        raise KeyError(
            "Checkpoint model keys do not match the one-box model. "
            f"Missing: {missing}; unexpected: {unexpected}."
        )
    if actor_input_key not in target_model:
        raise KeyError(
            f"Target model has no Actor height input {actor_input_key!r}."
        )
    for prefix in critic_prefixes:
        if not any(key.startswith(prefix) for key in target_model):
            raise KeyError(f"Target model has no parameters under {prefix!r}.")

    source_x, source_y = map(int, source_grid_shape)
    target_x, target_y = map(int, target_grid_shape)
    lateral_padding = target_y - source_y
    if target_x < source_x or lateral_padding < 0 or lateral_padding % 2:
        raise ValueError(
            "The one-box height grid must contain the centered rough grid."
        )
    target_y_start = lateral_padding // 2
    source_input = source_model[actor_input_key]
    target_input = target_model[actor_input_key]
    expected_source_shape = (target_input.shape[0], source_x * source_y)
    expected_target_shape = (target_input.shape[0], target_x * target_y)
    if tuple(source_input.shape) != expected_source_shape:
        raise ValueError(
            f"Actor height input has source shape {tuple(source_input.shape)}, "
            f"expected {expected_source_shape}."
        )
    if tuple(target_input.shape) != expected_target_shape:
        raise ValueError(
            f"Actor height input has target shape {tuple(target_input.shape)}, "
            f"expected {expected_target_shape}."
        )

    new_model_state_dict = OrderedDict()
    for key, target_value in target_model.items():
        source_value = source_model[key]
        if key.startswith(critic_prefixes):
            new_model_state_dict[key] = target_value
            continue
        if key == actor_input_key:
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
                f"Preserved parameter {key!r} has incompatible shapes: "
                f"checkpoint {tuple(source_value.shape)} versus target "
                f"{tuple(target_value.shape)}."
            )
        new_model_state_dict[key] = source_value

    print(
        "\033[1;36m Expanded the rough Actor height encoder; kept Actor, "
        "Actor memory, estimator, and action noise; reinitialized the "
        "complete Critic side and optimizer. \033[0m"
    )
    return dict(
        model_state_dict=new_model_state_dict,
        iter=source_state_dict["iter"],
        infos=source_state_dict.get("infos"),
    )


def initialize_go2_field_wide_scan_from_rough2000(
    source_state_dict,
    algo_state_dict,
):
    """Expand the rough Actor scan and reset Critic for the Go2 field task."""
    migrated = initialize_one_box_from_rough2000(
        source_state_dict,
        algo_state_dict,
        source_grid_shape=(21, 11),
        target_grid_shape=(36, 17),
    )
    print(
        "\033[1;36m Initialized the 36x17 Go2 field policy from rough "
        "model_2000: retained the aligned Actor scan, reset Critic and "
        "optimizer. \033[0m"
    )
    return migrated


def initialize_one_box_lift_from_warmup2100(
    source_state_dict, algo_state_dict
):
    """Keep the one-box warmup model while applying the light lift-stage KL."""
    if int(source_state_dict.get("iter", -1)) != 2100:
        raise ValueError(
            "One-box lift initialization requires model_2100.pt from the "
            "completed Critic warmup."
        )
    source_algorithm = source_state_dict.get("algorithm_state_dict", {})
    if source_algorithm.get("critic_warmup_until_iteration") != 2100:
        raise ValueError(
            "The source checkpoint is not the completed one-box warmup."
        )
    if source_algorithm.get("actor_finetune_active", False):
        raise ValueError(
            "The source checkpoint already contains Actor fine-tuning."
        )
    if source_state_dict.get("reference_model_state_dict") is None:
        raise ValueError("The one-box warmup has no frozen reference Actor.")

    source_model = source_state_dict["model_state_dict"]
    target_model = algo_state_dict["model_state_dict"]
    if source_model.keys() != target_model.keys():
        raise KeyError("One-box warmup and lift model keys do not match.")
    for name, source_value in source_model.items():
        if source_value.shape != target_model[name].shape:
            raise ValueError(
                f"Parameter {name!r} has incompatible shapes: "
                f"{tuple(source_value.shape)} versus "
                f"{tuple(target_model[name].shape)}."
            )

    target_algorithm = algo_state_dict["algorithm_state_dict"]
    migrated = copy.deepcopy(source_state_dict)
    migrated_algorithm = migrated["algorithm_state_dict"]
    for name in (
        "reference_kl_min_coef",
        "reference_kl_max_coef",
        "current_reference_kl_coef",
    ):
        migrated_algorithm[name] = copy.deepcopy(target_algorithm[name])
    migrated_algorithm["reference_kl_stable_window_count"] = 0
    migrated_algorithm["curriculum_stable_windows"] = 0
    migrated_algorithm["curriculum_regression_windows"] = 0
    migrated_algorithm["collapse_windows"] = 0
    migrated_algorithm["collapse_warning"] = False
    migrated_algorithm["quality_stage_start_iteration"] = 2100
    print(
        "\033[1;36m Preserved the one-box warmup Actor, Critic, reference "
        "Actor, and optimizer; capped lift-stage reference KL at "
        f"{migrated_algorithm['reference_kl_max_coef']}. \033[0m"
    )
    return migrated


def enable_flat_reference_kl_from_one_box2500(
    source_state_dict, algo_state_dict
):
    """Enable masked rough-walking KL without changing the 2500 policy."""
    if int(source_state_dict.get("iter", -1)) != 2500:
        raise ValueError(
            "Flat-reference KL initialization requires the accepted one-box "
            "model_2500.pt checkpoint."
        )
    source_model = source_state_dict["model_state_dict"]
    target_model = algo_state_dict["model_state_dict"]
    if source_model.keys() != target_model.keys():
        raise KeyError("One-box 2500 and flat-KL model keys do not match.")
    for name, source_value in source_model.items():
        if source_value.shape != target_model[name].shape:
            raise ValueError(
                f"Parameter {name!r} has incompatible shapes: "
                f"{tuple(source_value.shape)} versus "
                f"{tuple(target_model[name].shape)}."
            )

    target_algorithm = algo_state_dict["algorithm_state_dict"]
    migrated = copy.deepcopy(source_state_dict)
    migrated["reference_model_state_dict"] = None
    migrated_algorithm = migrated.setdefault("algorithm_state_dict", {})
    for name in (
        "reference_kl_min_coef",
        "reference_kl_max_coef",
        "current_reference_kl_coef",
    ):
        migrated_algorithm[name] = copy.deepcopy(target_algorithm[name])
    migrated_algorithm["reference_kl_stable_window_count"] = 0
    print(
        "\033[1;36m Kept the complete one-box 2500 policy, Critic, "
        "optimizer, and curriculum; enabled flat-only reference KL at "
        f"{migrated_algorithm['current_reference_kl_coef']}. The runner will "
        "load the frozen rough Actor separately. "
        "\033[0m"
    )
    return migrated


def initialize_v11_from_v10_warmup(source_state_dict, algo_state_dict):
    """Initialize v11 curriculum state from the verified v10 warmup boundary."""
    if int(source_state_dict.get("iter", -1)) != 11800:
        raise ValueError(
            "v11 initialization requires model_11800_warmup.pt at iteration "
            "11800."
        )
    source_algorithm = source_state_dict.get("algorithm_state_dict", {})
    if source_algorithm.get("critic_warmup_until_iteration") != 11800:
        raise ValueError(
            "The source checkpoint is not the verified 11800 warmup boundary."
        )
    if source_algorithm.get("actor_finetune_active", False):
        raise ValueError(
            "The source checkpoint already contains Actor fine-tuning updates."
        )
    if source_state_dict.get("reference_model_state_dict") is None:
        raise ValueError("The source checkpoint has no frozen reference Actor.")

    source_model = source_state_dict["model_state_dict"]
    target_model = algo_state_dict["model_state_dict"]
    if source_model.keys() != target_model.keys():
        raise KeyError("v10 and v11 model parameter names do not match.")
    for name, source_value in source_model.items():
        if source_value.shape != target_model[name].shape:
            raise ValueError(
                f"Parameter {name!r} has incompatible shapes: "
                f"{tuple(source_value.shape)} versus "
                f"{tuple(target_model[name].shape)}."
            )

    target_algorithm = algo_state_dict["algorithm_state_dict"]
    curriculum_keys = (
        "curriculum_state_version",
        "quality_phase",
        "speed_penalty_level",
        "motion_quality_level",
        "curriculum_stable_windows",
        "curriculum_regression_windows",
        "speed_master_windows",
        "motion_master_windows",
        "reference_kl_stable_window_count",
        "collapse_windows",
        "collapse_warning",
        "reward_order_warning",
        "reference_kl_min_coef",
        "reference_kl_max_coef",
        "current_reference_kl_coef",
    )
    migrated = copy.deepcopy(source_state_dict)
    migrated_algorithm = migrated.setdefault("algorithm_state_dict", {})
    for key in curriculum_keys:
        migrated_algorithm[key] = copy.deepcopy(target_algorithm[key])
    migrated_algorithm["quality_stage_start_iteration"] = 11800
    print(
        "\033[1;36m Preserved the v10 warmup Actor, Critic, reference Actor, "
        "and optimizer; initialized explicit v11 curriculum state. \033[0m"
    )
    return migrated


def _enable_v11_finetune_with_kl_floor(
    source_state_dict,
    algo_state_dict,
    required_minimum,
    description,
):
    """Preserve a healthy v11 checkpoint and replace only its KL floor."""
    source_algorithm = source_state_dict.get("algorithm_state_dict", {})
    if source_algorithm.get("curriculum_state_version") != 11:
        raise ValueError(f"{description} requires a v11 checkpoint.")
    if not source_algorithm.get("actor_finetune_active", False):
        raise ValueError(
            f"{description} requires an active Actor checkpoint."
        )
    if source_state_dict.get("reference_model_state_dict") is None:
        raise ValueError("The source checkpoint has no frozen reference Actor.")

    source_model = source_state_dict["model_state_dict"]
    target_model = algo_state_dict["model_state_dict"]
    if source_model.keys() != target_model.keys():
        raise KeyError("Source and target model parameter names do not match.")
    for name, source_value in source_model.items():
        if source_value.shape != target_model[name].shape:
            raise ValueError(
                f"Parameter {name!r} has incompatible shapes: "
                f"{tuple(source_value.shape)} versus "
                f"{tuple(target_model[name].shape)}."
            )

    target_algorithm = algo_state_dict["algorithm_state_dict"]
    target_minimum = float(target_algorithm["reference_kl_min_coef"])
    if abs(target_minimum - required_minimum) > 1e-12:
        raise ValueError(
            f"{description} requires reference_kl_min_coef="
            f"{required_minimum}."
        )

    migrated = copy.deepcopy(source_state_dict)
    migrated_algorithm = migrated["algorithm_state_dict"]
    migrated_algorithm["reference_kl_min_coef"] = target_minimum
    migrated_algorithm["reference_kl_max_coef"] = float(
        target_algorithm["reference_kl_max_coef"]
    )
    migrated_algorithm["current_reference_kl_coef"] = target_minimum
    migrated_algorithm["reference_kl_stable_window_count"] = 0
    print(
        "\033[1;36m Preserved the complete v11 checkpoint and enabled "
        f"{description} with KL minimum/current coefficient "
        f"{target_minimum}. "
        "\033[0m"
    )
    return migrated


def enable_v11_target_speed_finetune(source_state_dict, algo_state_dict):
    """Preserve a v11 checkpoint and use the historical 0.02 KL floor."""
    return _enable_v11_finetune_with_kl_floor(
        source_state_dict,
        algo_state_dict,
        required_minimum=0.02,
        description="target-speed fine-tuning",
    )


def enable_v14_speed_priority_finetune(source_state_dict, algo_state_dict):
    """Preserve a v11 checkpoint and start v14 with a 0.002 KL floor."""
    return _enable_v11_finetune_with_kl_floor(
        source_state_dict,
        algo_state_dict,
        required_minimum=0.002,
        description="v14 speed-priority fine-tuning",
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
