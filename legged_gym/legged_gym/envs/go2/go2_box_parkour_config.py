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
        episode_length_s = 30
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
            # Stage-one range; restore [0.5, 1.2] after box-top contact emerges.
            lin_vel_x = [0.4, 0.8]
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
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.2
            # Keep progress useful but cap its raw reward at 1.2 m/s.
            lin_vel_x = 0.5
            # Allow short jump bursts up to 0.4 m/s above the command.
            overspeed = -1.0
            lin_pos_y = -0.1
            yaw_abs = -0.1
            energy_substeps = -2e-7
            torques = -1e-7
            dof_error_named = -1.0
            dof_error = -0.005
            collision = -0.05
            exceed_dof_pos_limits = -0.1
            exceed_torque_limits_l1norm = -0.1
            box_first_foot_contact = 25.0
            box_second_foot_contact = 100.0
            box_passed = 250.0
            success = 500.0
            termination = -100.0
            episode_timeout = -250.0

        only_positive_rewards = False

    class box_progress:
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

    class runner(Go2RoughCfgPPO.runner):
        experiment_name = "go2_box_parkour"
        run_name = "five_box_landing_speed_v2_from10900"
        resume = True
        load_run = osp.join(
            osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))),
            "logs",
            "go2_box_parkour",
            "Jul19_21-37-13_five_box_reward_v3_from10200",
        )
        checkpoint = 10900
        ckpt_manipulator = None
        max_iterations = 2000
