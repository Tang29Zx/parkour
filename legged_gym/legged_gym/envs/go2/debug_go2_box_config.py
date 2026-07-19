"""Fixed Go2 box geometry for viewer and collision debugging."""

from legged_gym.envs.go2.go2_config import Go2RoughCfg, Go2RoughCfgPPO


class DebugGo2BoxCfg(Go2RoughCfg):
    class env(Go2RoughCfg.env):
        num_envs = 4
        episode_length_s = 1000
        debug_geometry = True
        debug_zero_actions = True

    class init_state(Go2RoughCfg.init_state):
        # Start at the normal standing height on flat ground before the course.
        pos = [0.0, 0.0, 0.5]
        zero_actions = True

    class terrain(Go2RoughCfg.terrain):
        selected = "RandomBoxTrack"
        mesh_type = None
        num_rows = 1
        num_cols = 4
        max_init_terrain_level = 0
        curriculum = False
        measure_heights = True
        RandomBoxTrack_kwargs = dict(
            randomize=True,
            seed=0,
            track_length=15.5,
            track_width=2.0,
            spawn_margin=0.6,
            first_gap_range=(0.8, 1.2),
            gap_distributions=[
                dict(name="dense", range=(0.45, 0.75), weight=0.2),
                dict(name="normal", range=(0.75, 1.20), weight=0.6),
                dict(name="sparse", range=(1.20, 1.60), weight=0.2),
            ],
            high_box_threshold=0.4,
            post_high_min_gap=0.8,
            boxes=[
                dict(gap=0.6, length=1.2, width=1.2, height=0.20, lateral_offset=0.0),
                dict(gap=0.7, length=1.2, width=1.2, height=0.30, lateral_offset=0.0),
                dict(gap=0.8, length=1.2, width=1.2, height=0.40, lateral_offset=0.0),
                dict(gap=0.6, length=1.2, width=1.2, height=0.40, lateral_offset=0.0),
                dict(gap=0.8, length=1.2, width=1.2, height=0.50, lateral_offset=0.0),
            ],
        )

    class commands(Go2RoughCfg.commands):
        heading_command = False
        resampling_time = 1e16

        class ranges(Go2RoughCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class domain_rand(Go2RoughCfg.domain_rand):
        randomize_com = False
        randomize_motor = False
        randomize_base_mass = False
        randomize_friction = False
        push_robots = False
        # first_gap_range controls the full run-up from this spawn position.
        init_base_pos_range = dict(x=[0.0, 0.0], y=[0.0, 0.0])
        init_base_rot_range = dict(
            roll=[0.0, 0.0],
            pitch=[0.0, 0.0],
            yaw=[0.0, 0.0],
        )
        init_base_vel_range = [0.0, 0.0]
        init_dof_vel_range = [0.0, 0.0]
        init_dof_pos_ratio_range = [1.0, 1.0]

    class termination(Go2RoughCfg.termination):
        timeout_at_border = False

    class viewer(Go2RoughCfg.viewer):
        pos = [3.0, 2.0, 9.0]
        lookat = [12.0, 9.0, 0.2]


class DebugGo2BoxCfgPPO(Go2RoughCfgPPO):
    class runner(Go2RoughCfgPPO.runner):
        experiment_name = "debug_go2_box"
        run_name = "viewer"
        resume = False
        load_run = -1
        checkpoint = -1
        max_iterations = 1
