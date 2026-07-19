"""Fixed Go2 hurdle geometry for viewer and collision debugging."""

from legged_gym.envs.go2.go2_field_config import Go2FieldCfg, Go2FieldCfgPPO


class DebugGo2BoxCfg(Go2FieldCfg):
    class env(Go2FieldCfg.env):
        num_envs = 4
        episode_length_s = 1000
        debug_geometry = True
        debug_zero_actions = True

    class init_state(Go2FieldCfg.init_state):
        pos = [0.0, 0.0, 0.5]
        zero_actions = True

    class terrain(Go2FieldCfg.terrain):
        selected = "BarrierTrack"
        mesh_type = None
        num_rows = 1
        num_cols = 4
        max_init_terrain_level = 0
        curriculum = False
        measure_heights = True
        BarrierTrack_kwargs = dict(
            options=["hurdle"],
            randomize_obstacle_order=False,
            n_obstacles_per_track=1,
            track_width=1.6,
            # Hurdle starts one 0.025 m cell after this block boundary.
            track_block_length=1.475,
            wall_thickness=0.0,
            wall_height=0.0,
            hurdle=dict(
                height=0.10,
                depth=0.40,
                curved_top_rate=0.0,
            ),
            add_perlin_noise=False,
            border_perlin_noise=False,
            virtual_terrain=False,
            draw_virtual_terrain=False,
            engaging_next_threshold=0.0,
            engaging_finish_threshold=0.0,
        )

    class commands(Go2FieldCfg.commands):
        is_goal_based = False
        heading_command = False
        resampling_time = 1e16

        class ranges(Go2FieldCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class domain_rand(Go2FieldCfg.domain_rand):
        randomize_com = False
        randomize_motor = False
        randomize_base_mass = False
        randomize_friction = False
        push_robots = False
        init_base_pos_range = dict(x=[0.0, 0.0], y=[0.0, 0.0])
        init_base_rot_range = dict(
            roll=[0.0, 0.0],
            pitch=[0.0, 0.0],
            yaw=[0.0, 0.0],
        )
        init_base_vel_range = [0.0, 0.0]
        init_dof_vel_range = [0.0, 0.0]
        init_dof_pos_ratio_range = [1.0, 1.0]

    class termination(Go2FieldCfg.termination):
        timeout_at_border = False
        timeout_at_finished = False

    class viewer(Go2FieldCfg.viewer):
        pos = [-1.5, -2.5, 1.3]
        lookat = [1.3, 0.0, 0.25]


class DebugGo2BoxCfgPPO(Go2FieldCfgPPO):
    class runner(Go2FieldCfgPPO.runner):
        experiment_name = "debug_go2_box"
        run_name = "viewer"
        resume = False
        load_run = -1
        checkpoint = -1
        max_iterations = 1
