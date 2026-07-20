"""Training configuration for the Go2 five-box parkour task."""

from copy import deepcopy
import os.path as osp

import numpy as np

from legged_gym.envs.go2.debug_go2_box_config import DebugGo2BoxCfg
from legged_gym.envs.go2.go2_config import Go2RoughCfgPPO


class Go2BoxParkourCfg(DebugGo2BoxCfg):
    """Reuse the verified box geometry with policy actions enabled."""

    class env(DebugGo2BoxCfg.env):
        num_envs = 4096
        episode_length_s = 45
        debug_geometry = False
        debug_zero_actions = False

    class init_state(DebugGo2BoxCfg.init_state):
        zero_actions = False

    class terrain(DebugGo2BoxCfg.terrain):
        # Keep training and geometry-debug configuration mutations isolated.
        RandomBoxTrack_kwargs = deepcopy(
            DebugGo2BoxCfg.terrain.RandomBoxTrack_kwargs
        )
        # Spread 4096 actors over 512 physical tracks while repeating only the
        # same four logical layouts. This avoids GPU broad-phase pair overflow.
        num_rows = 8
        num_cols = 64
        max_init_terrain_level = 7
        RandomBoxTrack_kwargs["num_unique_layouts"] = 4
        # Oracle scan aligned with the existing 3 m forward-depth camera range.
        measured_points_x = np.linspace(-0.5, 3.0, 36).tolist()
        measured_points_y = np.linspace(-0.8, 0.8, 17).tolist()

    class commands(DebugGo2BoxCfg.commands):
        heading_command = False
        # A command is sampled by the existing reset path and remains fixed.
        resampling_time = 1e16

        class ranges(DebugGo2BoxCfg.commands.ranges):
            lin_vel_x = [0.45, 0.55]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class asset(DebugGo2BoxCfg.asset):
        terminate_after_contacts_on = []
        penalize_contacts_on = ["thigh", "calf", "base"]

    class termination(DebugGo2BoxCfg.termination):
        roll_kwargs = dict(threshold=1.4)
        pitch_kwargs = dict(threshold=1.6)
        timeout_at_border = False

    class rewards(DebugGo2BoxCfg.rewards):
        class scales:
            tracking_ang_vel = 0.2
            forward_speed_tracking = 1.0
            course_progress = 1000.0
            speed_error_square = -1.0
            overspeed = -1.5
            action_rate = -0.01
            flat_orientation = -0.2
            lin_pos_y = -0.1
            yaw_abs = -0.1
            energy_substeps = -2e-7
            torques = -1e-7
            dof_error_named = -1.0
            dof_error = -0.005
            # Base contact is a hard safety cost. Leg rubbing is introduced by
            # the action-quality curriculum to preserve the source behavior.
            body_collision = -0.05
            thigh_collision = -0.05
            calf_collision = -0.05
            exceed_dof_pos_limits = -0.1
            exceed_torque_limits_l1norm = -0.1
            box_first_foot_contact = 5.0
            box_second_foot_contact = 10.0
            box_passed = 25.0
            success = 500.0
            termination = -2000.0
            incomplete = -2000.0

        only_positive_rewards = False
        forward_speed_tracking_sigma = 0.02
        speed_penalty_initial_level = 0.1
        motion_quality_initial_level = 0.0
        # Track the command closely on flat ground. Near a box, allow a brief
        # positive speed error for jumping or climbing without rewarding it.
        # Smoothly blend the local speed limit over these distances. Only the
        # current target box can activate this window.
        box_speed_ramp_up_distance = 0.5
        box_speed_ramp_down_distance = 0.5
        box_lateral_margin = 0.2
        box_speed_allowance = 0.5
        flat_speed_limit = 0.7
        flat_severe_speed_limit = 0.8
        box_speed_limit = 1.2
        # Task failure remains costly even after all boxes are passed but the
        # landing has not been completed.
        failure_progress_floor = 0.5
        leg_contact_force_threshold = 0.1
        body_collision_force_threshold = 1.0
        dof_near_limit_fraction = 0.15
        action_saturation_threshold = 0.95

    class box_progress:
        required_boxes = 5
        min_landing_zone_length = 0.85
        pass_margin = 0.15
        top_contact_tolerance = 0.06
        contact_force_threshold = 1.0
        required_distinct_feet = 2
        landing_steps = 10
        body_contact_steps = 15
        roll_threshold = 1.4
        pitch_threshold = 1.6
        base_height_threshold = 0.15
        lateral_limit = 0.8


class Go2BoxParkourCfgPPO(Go2RoughCfgPPO):
    class policy(Go2RoughCfgPPO.policy):
        # Keep perception replaceable: height or depth must produce 32 values.
        encoder_component_names = ["height_measurements"]
        critic_encoder_component_names = ["height_measurements"]
        encoder_output_size = 32

    class algorithm(Go2RoughCfgPPO.algorithm):
        # Use conservative fixed settings while fine-tuning a learned policy.
        schedule = "fixed"
        learning_rate = 5e-5
        entropy_coef = 0.003
        gamma = 0.999
        lam = 0.95
        critic_warmup_iterations = 100
        actor_finetune_learning_rate = 1e-5
        actor_finetune_clip_param = 0.1
        actor_finetune_entropy_coef = 0.001
        # Adapt behavior protection from task capability rather than coupling
        # it directly to the active penalty level.
        reference_kl_min_coef = 0.02
        reference_kl_max_coef = 1.0
        reference_kl_start_coef = 0.20
        reference_kl_stable_windows = 2
        reference_kl_stable_decrease = 0.02
        reference_kl_regression_increase = 0.10
        reference_kl_regression_floor = 0.50
        # Current and frozen Actors are compared on the same one-step GRU path.
        # Padded batch replay has a separate diagnostic because CUDA float32
        # accumulation order can differ without any parameter mutation.
        actor_parameter_equivalence_tolerance = 0.0
        actor_output_equivalence_tolerance = 1e-5
        actor_std_equivalence_tolerance = 1e-7
        require_v11_curriculum_state = True
        quality_min_episodes = 256
        speed_penalty_initial_level = 0.10
        motion_quality_initial_level = 0.0
        curriculum_level_step = 0.10
        curriculum_stage_min_iterations = 200
        curriculum_stable_windows = 3
        curriculum_regression_windows = 2
        speed_success_up = 0.90
        speed_box_pass_up = 0.92
        speed_fall_up = 0.10
        curriculum_success_down = 0.85
        curriculum_box_pass_down = 0.88
        curriculum_fall_down = 0.15
        speed_master_windows = 5
        speed_master_flat_min = 0.45
        speed_master_flat_max = 0.70
        speed_master_severe_overspeed = 0.05
        motion_success_up = 0.88
        motion_box_pass_up = 0.90
        motion_fall_up = 0.12
        motion_flat_speed_max = 0.75
        motion_severe_overspeed_max = 0.10
        motion_master_windows = 5
        motion_master_action_rate = 2.0
        motion_master_action_saturation = 0.20
        motion_master_dof_near_limit = 0.10
        collapse_success_threshold = 0.60
        collapse_fall_threshold = 0.50

    class runner(Go2RoughCfgPPO.runner):
        experiment_name = "go2_box_parkour"
        run_name = "five_box_v11_speed_phase_from11800"
        resume = True
        load_run = osp.join(
            osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))),
            "logs",
            "go2_box_parkour",
            "Jul20_18-38-38_five_box_v10_retry3_warmup100_train900",
        )
        checkpoint = 11800
        # Initial Critic migration is selected explicitly from the CLI. Keeping
        # this disabled prevents later v10 resumes from resetting Critic again.
        ckpt_manipulator = None
        max_iterations = 200
        save_interval = 100
        log_interval = 10


class Go2BoxParkour1BoxCfg(Go2BoxParkourCfg):
    """One-box curriculum stage on the unchanged five-box terrain."""

    class env(Go2BoxParkourCfg.env):
        episode_length_s = 15

    class box_progress(Go2BoxParkourCfg.box_progress):
        required_boxes = 1


class Go2BoxParkour1BoxCfgPPO(Go2BoxParkourCfgPPO):
    class runner(Go2BoxParkourCfgPPO.runner):
        run_name = "one_box_v9_critic_warmup_from11700"
        # The source checkpoint must be selected explicitly on the CLI.
        resume = False
        load_run = -1
        checkpoint = -1
        ckpt_manipulator = None
        max_iterations = 1000


class Go2BoxParkour3BoxCfg(Go2BoxParkourCfg):
    """Three-box curriculum stage on the unchanged five-box terrain."""

    class env(Go2BoxParkourCfg.env):
        episode_length_s = 30

    class box_progress(Go2BoxParkourCfg.box_progress):
        required_boxes = 3


class Go2BoxParkour3BoxCfgPPO(Go2BoxParkourCfgPPO):
    class runner(Go2BoxParkourCfgPPO.runner):
        run_name = "three_box_v9_curriculum"
        # Continue from the accepted one-box checkpoint without migration.
        resume = False
        load_run = -1
        checkpoint = -1
        ckpt_manipulator = None
        max_iterations = 1000
