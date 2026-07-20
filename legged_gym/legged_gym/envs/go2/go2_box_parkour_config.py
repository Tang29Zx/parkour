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
            speed_error_square = -1.0
            overspeed = -1.5
            lin_pos_y = -0.1
            yaw_abs = -0.1
            energy_substeps = -2e-7
            torques = -1e-7
            dof_error_named = -1.0
            dof_error = -0.005
            collision = -0.05
            exceed_dof_pos_limits = -0.1
            exceed_torque_limits_l1norm = -0.1
            box_first_foot_contact = 5.0
            box_second_foot_contact = 10.0
            box_passed = 25.0
            success = 100.0
            termination = -100.0
            episode_timeout = -150.0

        only_positive_rewards = False
        # Track the command closely on flat ground. Near a box, allow a brief
        # positive speed error for jumping or climbing without rewarding it.
        box_approach_distance = 0.5
        box_exit_distance = 0.2
        box_lateral_margin = 0.2
        box_speed_allowance = 0.5
        flat_speed_limit = 0.7
        flat_severe_speed_limit = 0.8
        box_speed_limit = 1.2

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

    class runner(Go2RoughCfgPPO.runner):
        experiment_name = "go2_box_parkour"
        run_name = "five_box_v8_pre_curriculum_from11700"
        resume = True
        load_run = osp.join(
            osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))),
            "logs",
            "go2_box_parkour",
            "Jul19_23-03-03_five_box_v5_stable_from11000",
        )
        checkpoint = 11700
        # Initial Critic migration is selected explicitly from the CLI. Keeping
        # this disabled prevents later v8 resumes from resetting Critic again.
        ckpt_manipulator = None
        max_iterations = 2000
        save_interval = 250
        log_interval = 50


class Go2BoxParkour1BoxCfg(Go2BoxParkourCfg):
    """One-box curriculum stage on the unchanged five-box terrain."""

    class env(Go2BoxParkourCfg.env):
        episode_length_s = 15

    class box_progress(Go2BoxParkourCfg.box_progress):
        required_boxes = 1


class Go2BoxParkour1BoxCfgPPO(Go2BoxParkourCfgPPO):
    class runner(Go2BoxParkourCfgPPO.runner):
        run_name = "one_box_v8_curriculum"
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
        run_name = "three_box_v8_curriculum"
        # Continue from the accepted one-box checkpoint without migration.
        resume = False
        load_run = -1
        checkpoint = -1
        ckpt_manipulator = None
        max_iterations = 1000
