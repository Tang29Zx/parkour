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
        # Add 2.5 m after the final box while the policy learns to stop.
        RandomBoxTrack_kwargs["track_length"] = 18.0
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

    class viewer(DebugGo2BoxCfg.viewer):
        # Center the static view on the 18 m formal track. play.py follows the
        # robot by default, using this vector as its camera offset.
        pos = [3.0, 2.0, 9.0]
        lookat = [14.0, 6.0, 0.2]

    class rewards(DebugGo2BoxCfg.rewards):
        class scales:
            tracking_ang_vel = 0.2
            forward_speed_tracking = 0.3
            course_progress = 1000.0
            landing_quality_progress = 150.0
            landing_hold_progress = 350.0
            landing_deceleration_progress = 200.0
            landing_alignment_progress = 250.0
            # V18 prioritizes clean landing and four-leg motion over exact speed.
            speed_error_square = 0.0
            overspeed = -0.2
            action_rate = -0.005
            flat_orientation = -0.2
            flat_base_height = 0.0
            dof_vel = -5e-5
            lin_pos_y = -0.1
            yaw_abs = -0.1
            dof_error_named = -0.2
            dof_error = -0.001
            body_collision = -5.0
            thigh_collision = -0.2
            calf_collision = -0.2
            rear_support_missing = -0.2
            flat_airborne = -0.1
            lateral_velocity_square = 0.0
            flat_lateral_position = 0.0
            flat_yaw_abs = 0.0
            world_overspeed = 0.0
            rear_upper_joint_excursion = 0.0
            exceed_torque_limits_l1norm = -1.0
            box_front_foot_contact = 5.0
            # Disabled on the five-box task. The dedicated one-box stage uses
            # a non-repeatable approach-window lift signal.
            front_foot_lift_progress = 0.0
            front_foot_reach_progress = 0.0
            rear_foot_lift_progress = 0.0
            rear_foot_reach_progress = 0.0
            box_approach_progress = 0.0
            front_foot_clearance_progress = 0.0
            post_front_base_progress = 0.0
            rear_foot_clearance_progress = 0.0
            box_rear_foot_contact = 10.0
            box_passed = 25.0
            success = 1250.0
            termination = -2000.0
            landing_overrun = -1000.0
            landing_lateral_exit = -1250.0
            landing_timeout = -750.0
            incomplete = -2000.0
            severe_body_impact = 0.0
            stagnation = 0.0

        only_positive_rewards = False
        forward_speed_tracking_sigma = 0.25
        speed_penalty_initial_level = 0.0
        motion_quality_initial_level = 0.0
        body_collision_floor = 1.0
        leg_collision_floor = 0.25
        flat_orientation_floor = 0.25
        flat_base_height_floor = 1.0
        action_rate_floor = 0.10
        dof_error_floor = 0.10
        dof_vel_floor = 0.10
        speed_error_floor = 0.0
        overspeed_floor = 1.0
        rear_support_window_steps = 25
        rear_support_target_ratio = 0.20
        flat_airborne_free_ratio = 0.40
        flat_height_min = 0.28
        flat_height_max = 0.38
        flat_height_normalization = 0.08
        flat_hip_allowance = 0.35
        flat_thigh_allowance = 0.75
        box_hip_allowance = 0.55
        box_thigh_allowance = 1.25
        # Track the command closely on flat ground. Near a box, allow a brief
        # positive speed error for jumping or climbing without rewarding it.
        # Smoothly blend the local speed limit over these distances. Only the
        # current target box can activate this window.
        box_speed_ramp_up_distance = 0.5
        box_speed_ramp_down_distance = 0.5
        box_lateral_margin = 0.2
        box_speed_allowance = 0.5
        flat_speed_limit = 1.2
        flat_severe_speed_limit = 1.5
        box_speed_limit = 1.8
        leg_contact_force_threshold = 0.1
        body_collision_force_threshold = 1.0
        dof_near_limit_fraction = 0.15
        action_saturation_threshold = 0.95
        front_foot_lift_approach_distance = 0.5
        front_foot_lift_clearance = 0.03
        front_foot_lift_lateral_margin = 0.2
        front_foot_reach_start_distance = 0.25
        front_foot_reach_target_inset = 0.125
        front_foot_reach_height_tolerance = 0.02
        front_foot_reach_base_overrun = 0.15
        foot_guidance_min_forward_speed = 0.05

    class box_progress:
        required_boxes = 5
        min_landing_zone_length = 0.85
        pass_margin = 0.15
        top_contact_tolerance = 0.06
        contact_force_threshold = 1.0
        front_contact_required_steps = 2
        rear_contact_required_steps = 2
        landing_steps = 10
        landing_min_current_feet = 2
        landing_require_rear_foot = True
        landing_roll_threshold = 0.35
        landing_pitch_threshold = 0.45
        landing_base_height_threshold = 0.22
        landing_vertical_speed_threshold = 0.5
        landing_horizontal_speed_threshold = 0.35
        landing_lateral_speed_threshold = 0.20
        landing_lateral_offset_threshold = 0.40
        landing_yaw_threshold = 0.35
        landing_deceleration_start_speed = 1.5
        landing_deadline_steps = 150
        landing_command_ramp_steps = 20
        body_contact_window_steps = 25
        body_contact_failure_steps = 8
        severe_body_impact_force = 80.0
        roll_threshold = 1.4
        pitch_threshold = 1.6
        base_height_threshold = 0.15
        flat_low_base_height_threshold = 0.20
        flat_low_base_height_steps = 25
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
        # The 11300 Actor initializes V17 but is not a behavior reference.
        reference_kl_min_coef = 0.0
        reference_kl_max_coef = 0.0
        reference_kl_start_coef = 0.0
        reference_kl_stable_windows = 2
        reference_kl_stable_decrease = 0.02
        reference_kl_regression_increase = 0.10
        reference_kl_regression_floor = 0.50
        freeze_actor_encoder_iterations = 200
        # Keep the checkpoint schema compatible; with KL max set to zero no
        # frozen reference policy is created and these limits remain inactive.
        actor_parameter_equivalence_tolerance = 0.0
        actor_output_equivalence_tolerance = 1e-5
        actor_std_equivalence_tolerance = 1e-7
        require_v11_curriculum_state = True
        quality_min_episodes = 256
        speed_penalty_initial_level = 0.0
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
        run_name = "five_box_v18_landing_guidance_from11300"
        # Partial episodes generated at process startup do not represent the
        # checkpoint policy and must not drive the box curriculum or KL state.
        init_at_random_ep_len = False
        resume = True
        load_run = osp.join(
            osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))),
            "logs",
            "go2_box_parkour",
            "Jul19_22-22-00_five_box_v4_from11200",
        )
        checkpoint = 11300
        # Select Critic reset explicitly on the first CLI launch. Later V17
        # resumes must not reset the learned Critic again.
        ckpt_manipulator = None
        max_iterations = 200
        save_interval = 50
        log_interval = 10


class Go2BoxParkour1BoxCfg(Go2BoxParkourCfg):
    """Dedicated one-box task for learning clean obstacle traversal."""

    class env(Go2BoxParkourCfg.env):
        episode_length_s = 15

    class terrain(Go2BoxParkourCfg.terrain):
        RandomBoxTrack_kwargs = deepcopy(
            Go2BoxParkourCfg.terrain.RandomBoxTrack_kwargs
        )
        RandomBoxTrack_kwargs.update(
            randomize=True,
            seed=0,
            num_unique_layouts=7,
            track_length=6.2,
            track_width=2.0,
            spawn_margin=0.6,
            first_gap_range=(1.2, 1.5),
            boxes=[
                dict(
                    gap=1.2,
                    length=1.6,
                    width=1.6,
                    height=0.15,
                    height_choices=(
                        0.08,
                        0.10,
                        0.12,
                        0.14,
                        0.16,
                        0.18,
                        0.20,
                    ),
                    lateral_offset=0.0,
                ),
            ],
        )

    class rewards(Go2BoxParkourCfg.rewards):
        class scales(Go2BoxParkourCfg.rewards.scales):
            tracking_ang_vel = 0.2
            forward_speed_tracking = 0.5
            course_progress = 0.0
            landing_quality_progress = 0.0
            landing_hold_progress = 0.0
            landing_deceleration_progress = 0.0
            landing_alignment_progress = 0.0
            speed_error_square = 0.0
            overspeed = -0.2
            action_rate = -0.005
            flat_orientation = -0.1
            dof_vel = 0.0
            lin_pos_y = -0.1
            yaw_abs = -0.1
            energy_substeps = 0.0
            torques = 0.0
            dof_error_named = 0.0
            dof_error = 0.0
            body_collision = -5.0
            thigh_collision = 0.0
            calf_collision = 0.0
            rear_support_missing = -0.2
            flat_airborne = -0.2
            lateral_velocity_square = -0.3
            flat_lateral_position = -0.2
            flat_yaw_abs = -0.2
            world_overspeed = -0.5
            # Each progress reward is a normalized, non-repeatable high-water
            # increment. The configured values are the requested actual return
            # multiplied by 1 / dt = 50.
            front_foot_lift_progress = 0.0
            front_foot_reach_progress = 0.0
            rear_foot_lift_progress = 0.0
            rear_foot_reach_progress = 0.0
            box_approach_progress = 100.0
            front_foot_clearance_progress = 100.0
            box_front_foot_contact = 300.0
            post_front_base_progress = 150.0
            rear_foot_clearance_progress = 100.0
            box_rear_foot_contact = 400.0
            box_passed = 500.0
            success = 750.0
            termination = -2250.0
            severe_body_impact = -2500.0
            stagnation = -2000.0
            landing_overrun = -2250.0
            landing_lateral_exit = -2250.0
            landing_timeout = -2250.0
            incomplete = -2000.0

        failure_progress_scaling = False
        reward_order_mode = "success_above_failures"
        foot_clearance_height = 0.03
        foot_clearance_start_distance = 0.30
        foot_clearance_target_inset = 0.12
        post_front_base_target_fraction = 0.65
        # Enable gait repair only after the front/rear contact stages have
        # completed. The active box window remains unconstrained.
        quality_repair_min_curriculum_stage = 2
        action_rate_flat_only = True
        quality_repair_flat_speed_limit = 1.2
        action_rate_floor = 1.0
        flat_airborne_free_ratio = 0.20

    class box_progress(Go2BoxParkourCfg.box_progress):
        required_boxes = 1
        min_landing_zone_length = 2.0
        landing_min_forward_distance = 0.6
        landing_horizontal_speed_threshold = 1.2
        landing_lateral_speed_threshold = 0.35
        landing_deadline_steps = 200
        stop_command_after_course = False
        stagnation_region_distance = 0.6
        stagnation_steps = 125

    class one_box_curriculum:
        enabled = True
        state_version = 1
        stage_names = ("front_contact", "rear_contact", "full_traversal")
        low_height_layouts = (0, 1, 2)
        full_height_layouts = (2, 3, 4, 5, 6)
        minimum_stage_iterations = 100
        minimum_episodes = 256
        required_stable_windows = 2
        promotion_success_rate = 0.65


class Go2BoxParkour1BoxCfgPPO(Go2BoxParkourCfgPPO):
    class algorithm(Go2BoxParkourCfgPPO.algorithm):
        freeze_actor_encoder_iterations = 100
        actor_finetune_learning_rate = 2e-5
        actor_finetune_clip_param = 0.1
        actor_finetune_entropy_coef = 0.002
        # A rough walking policy cannot be a behavior reference for box
        # traversal. Preserve it only as initialization and disable KL.
        reference_kl_min_coef = 0.0
        reference_kl_max_coef = 0.0
        reference_kl_start_coef = 0.0

    class runner(Go2BoxParkourCfgPPO.runner):
        run_name = "one_box_v184_gait_repair_from2500"
        init_at_random_ep_len = False
        resume = True
        load_run = osp.join(
            osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))),
            "logs",
            "go2_box_parkour",
            "Jul21_18-05-59_one_box_v183_from_rough2000",
        )
        checkpoint = 2500
        ckpt_manipulator = None
        max_iterations = 200
        save_interval = 50
        log_interval = 50


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
