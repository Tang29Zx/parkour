"""Tests for five-box reward configuration and checkpoint migration."""

import importlib.util
from collections import OrderedDict
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest


ISAAC_GYM_AVAILABLE = importlib.util.find_spec("isaacgym") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if ISAAC_GYM_AVAILABLE:
    import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.
if TORCH_AVAILABLE:
    import torch

if importlib.util.find_spec("tensorboardX") is None:
    tensorboard_module = ModuleType("tensorboardX")
    tensorboard_module.SummaryWriter = object
    sys.modules["tensorboardX"] = tensorboard_module


@unittest.skipUnless(TORCH_AVAILABLE, "Checkpoint tests require PyTorch.")
class HeightEncoderMigrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module_path = (
            Path(__file__).resolve().parents[3]
            / "rsl_rl"
            / "rsl_rl"
            / "utils"
            / "ckpt_manipulator.py"
        )
        spec = importlib.util.spec_from_file_location(
            "ckpt_manipulator", module_path
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def make_states(self):
        source_model = OrderedDict(
            [
                ("actor.weight", torch.full((2, 2), 1.0)),
                ("encoders.0.weight", torch.full((2, 3), 2.0)),
                ("critic_encoders.0.weight", torch.full((2, 3), 3.0)),
                ("memory_a.rnn.weight", torch.full((2, 2), 4.0)),
            ]
        )
        target_model = OrderedDict(
            [
                ("actor.weight", torch.full((2, 2), 10.0)),
                ("encoders.0.weight", torch.full((2, 5), 20.0)),
                ("critic_encoders.0.weight", torch.full((2, 5), 30.0)),
                ("memory_a.rnn.weight", torch.full((2, 2), 40.0)),
            ]
        )
        source = dict(
            model_state_dict=source_model,
            optimizer_state_dict={"old": True},
            iter=8400,
            infos={"source": "walking"},
        )
        target = dict(model_state_dict=target_model)
        return source, target

    def make_expand_states(self):
        source_model = OrderedDict(
            [
                ("actor.weight", torch.full((2, 2), 1.0)),
                (
                    "encoders.0.model.0.weight",
                    torch.arange(12, dtype=torch.float32).reshape(3, 4),
                ),
                ("encoders.0.model.0.bias", torch.arange(3, dtype=torch.float32)),
                ("encoders.0.model.2.weight", torch.full((2, 3), 2.0)),
                ("encoders.0.model.2.bias", torch.full((2,), 3.0)),
                (
                    "critic_encoders.0.model.0.weight",
                    torch.arange(12, 24, dtype=torch.float32).reshape(3, 4),
                ),
                (
                    "critic_encoders.0.model.0.bias",
                    torch.arange(3, 6, dtype=torch.float32),
                ),
                ("critic_encoders.0.model.2.weight", torch.full((2, 3), 4.0)),
                ("critic_encoders.0.model.2.bias", torch.full((2,), 5.0)),
                ("memory_a.rnn.weight", torch.full((2, 2), 6.0)),
            ]
        )
        target_model = OrderedDict(
            (
                key,
                (
                    torch.full((3, 12), 99.0)
                    if key.endswith("model.0.weight")
                    else torch.full_like(value, 99.0)
                ),
            )
            for key, value in source_model.items()
        )
        source = dict(
            model_state_dict=source_model,
            optimizer_state_dict={"old": True},
            iter=9700,
            infos={"source": "walking"},
        )
        target = dict(model_state_dict=target_model)
        return source, target

    def test_reset_optimizer_keeps_model_and_discards_training_state(self):
        source, _ = self.make_states()
        target_model = OrderedDict(
            (key, torch.zeros_like(value))
            for key, value in source["model_state_dict"].items()
        )
        migrated = self.module.reset_optimizer_state(
            source,
            dict(model_state_dict=target_model),
        )

        for key, source_value in source["model_state_dict"].items():
            self.assertTrue(
                torch.equal(migrated["model_state_dict"][key], source_value)
            )
        self.assertNotIn("optimizer_state_dict", migrated)
        self.assertNotIn("lr_scheduler_state_dict", migrated)
        self.assertEqual(migrated["iter"], source["iter"])
        self.assertEqual(migrated["infos"], source["infos"])

    def test_reset_optimizer_rejects_model_mismatches(self):
        source, _ = self.make_states()
        target_model = OrderedDict(
            (key, torch.zeros_like(value))
            for key, value in source["model_state_dict"].items()
        )
        target_model["actor.weight"] = torch.zeros(3, 2)
        with self.assertRaises(ValueError):
            self.module.reset_optimizer_state(
                source,
                dict(model_state_dict=target_model),
            )

        del target_model["actor.weight"]
        with self.assertRaises(KeyError):
            self.module.reset_optimizer_state(
                source,
                dict(model_state_dict=target_model),
            )

    def test_reset_critic_keeps_actor_side_and_discards_training_state(self):
        names = (
            "std",
            "actor.0.weight",
            "memory_a.rnn.weight",
            "encoders.0.weight",
            "memory_s.rnn.weight",
            "state_estimator.model.0.weight",
            "critic.0.weight",
            "memory_c.rnn.weight",
            "critic_encoders.0.weight",
        )
        source_model = OrderedDict(
            (name, torch.full((2, 2), float(index + 1)))
            for index, name in enumerate(names)
        )
        target_model = OrderedDict(
            (name, torch.full((2, 2), float(index + 101)))
            for index, name in enumerate(names)
        )
        source = dict(
            model_state_dict=source_model,
            optimizer_state_dict={"old": True},
            lr_scheduler_state_dict={"old": True},
            iter=11700,
            infos={"source": "v5"},
        )

        migrated = self.module.reset_critic_and_optimizer(
            source,
            dict(model_state_dict=target_model),
        )

        critic_prefixes = ("critic.", "memory_c.", "critic_encoders.")
        for key in names:
            expected = (
                target_model[key]
                if key.startswith(critic_prefixes)
                else source_model[key]
            )
            self.assertTrue(
                torch.equal(migrated["model_state_dict"][key], expected)
            )
        self.assertNotIn("optimizer_state_dict", migrated)
        self.assertNotIn("lr_scheduler_state_dict", migrated)
        self.assertEqual(migrated["iter"], 11700)
        self.assertEqual(migrated["infos"], {"source": "v5"})

    def test_reset_critic_rejects_key_and_preserved_shape_mismatches(self):
        names = (
            "actor.0.weight",
            "critic.0.weight",
            "memory_c.rnn.weight",
            "critic_encoders.0.weight",
        )
        source_model = OrderedDict(
            (name, torch.zeros(2, 2)) for name in names
        )
        target_model = OrderedDict(
            (name, torch.ones(2, 2)) for name in names
        )
        source = dict(model_state_dict=source_model, iter=11700)

        target_model["actor.0.weight"] = torch.ones(3, 2)
        with self.assertRaises(ValueError):
            self.module.reset_critic_and_optimizer(
                source,
                dict(model_state_dict=target_model),
            )

        target_model["actor.0.weight"] = torch.ones(2, 2)
        del target_model["memory_c.rnn.weight"]
        with self.assertRaises(KeyError):
            self.module.reset_critic_and_optimizer(
                source,
                dict(model_state_dict=target_model),
            )

    def test_reinitializes_both_encoders_and_keeps_other_model_weights(self):
        source, target = self.make_states()
        migrated = self.module.reinitialize_height_encoders(source, target)

        self.assertTrue(
            torch.equal(
                migrated["model_state_dict"]["actor.weight"],
                source["model_state_dict"]["actor.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                migrated["model_state_dict"]["memory_a.rnn.weight"],
                source["model_state_dict"]["memory_a.rnn.weight"],
            )
        )
        for key in ("encoders.0.weight", "critic_encoders.0.weight"):
            self.assertTrue(
                torch.equal(
                    migrated["model_state_dict"][key],
                    target["model_state_dict"][key],
                )
            )
        self.assertNotIn("optimizer_state_dict", migrated)
        self.assertEqual(migrated["iter"], 8400)

    def test_rejects_non_encoder_key_or_shape_mismatches(self):
        source, target = self.make_states()
        source["model_state_dict"]["actor.weight"] = torch.zeros(3, 2)
        with self.assertRaises(ValueError):
            self.module.reinitialize_height_encoders(source, target)

        source, target = self.make_states()
        source["model_state_dict"]["unexpected.weight"] = torch.zeros(1)
        with self.assertRaises(KeyError):
            self.module.reinitialize_height_encoders(source, target)

    def test_expands_aligned_encoder_inputs_and_copies_every_other_parameter(self):
        source, target = self.make_expand_states()
        migrated = self.module.expand_height_encoder_inputs(
            source,
            target,
            source_grid_shape=(2, 2),
            target_grid_shape=(3, 4),
        )

        input_keys = (
            "encoders.0.model.0.weight",
            "critic_encoders.0.model.0.weight",
        )
        for key in input_keys:
            source_grid = source["model_state_dict"][key].reshape(3, 2, 2)
            migrated_grid = migrated["model_state_dict"][key].reshape(3, 3, 4)
            self.assertTrue(torch.equal(migrated_grid[:, :2, 1:3], source_grid))

            extra_mask = torch.ones(3, 4, dtype=torch.bool)
            extra_mask[:2, 1:3] = False
            self.assertTrue((migrated_grid[:, extra_mask] == 0.0).all())

        for key, source_value in source["model_state_dict"].items():
            if key not in input_keys:
                self.assertTrue(
                    torch.equal(migrated["model_state_dict"][key], source_value)
                )
        self.assertNotIn("optimizer_state_dict", migrated)
        self.assertEqual(migrated["iter"], 9700)

    def test_expanded_encoder_preserves_output_for_the_source_grid(self):
        source, target = self.make_expand_states()
        migrated = self.module.expand_height_encoder_inputs(
            source,
            target,
            source_grid_shape=(2, 2),
            target_grid_shape=(3, 4),
        )
        source_observations = torch.randn(8, 2, 2)
        target_observations = torch.randn(8, 3, 4)
        target_observations[:, :2, 1:3] = source_observations

        for prefix in ("encoders.0.", "critic_encoders.0."):
            input_key = prefix + "model.0.weight"
            bias_key = prefix + "model.0.bias"
            source_output = torch.nn.functional.linear(
                source_observations.flatten(1),
                source["model_state_dict"][input_key],
                source["model_state_dict"][bias_key],
            )
            migrated_output = torch.nn.functional.linear(
                target_observations.flatten(1),
                migrated["model_state_dict"][input_key],
                migrated["model_state_dict"][bias_key],
            )
            torch.testing.assert_close(migrated_output, source_output)

    def test_expand_rejects_unexpected_grid_or_parameter_shapes(self):
        source, target = self.make_expand_states()
        source["model_state_dict"]["encoders.0.model.0.weight"] = torch.zeros(
            3, 5
        )
        with self.assertRaises(ValueError):
            self.module.expand_height_encoder_inputs(
                source,
                target,
                source_grid_shape=(2, 2),
                target_grid_shape=(3, 4),
            )

        source, target = self.make_expand_states()
        target["model_state_dict"]["actor.weight"] = torch.zeros(2, 3)
        with self.assertRaises(ValueError):
            self.module.expand_height_encoder_inputs(
                source,
                target,
                source_grid_shape=(2, 2),
                target_grid_shape=(3, 4),
            )

        source, target = self.make_expand_states()
        with self.assertRaises(ValueError):
            self.module.expand_height_encoder_inputs(
                source,
                target,
                source_grid_shape=(2, 2),
                target_grid_shape=(3, 3),
            )


@unittest.skipUnless(
    TORCH_AVAILABLE and ISAAC_GYM_AVAILABLE,
    "Reward tests require PyTorch and Isaac Gym.",
)
class BoxRewardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from legged_gym.envs.base.legged_robot import LeggedRobot
        from legged_gym.envs.base.legged_robot_box import LeggedRobotBox
        from legged_gym.envs.go2.go2_box_parkour_config import (
            Go2BoxParkourCfg,
            Go2BoxParkourCfgPPO,
        )
        from legged_gym.envs.go2.go2_config import Go2RoughCfg
        from legged_gym.envs.go2.debug_go2_box_config import DebugGo2BoxCfg
        from legged_gym.utils.helpers import update_cfg_from_args

        cls.LeggedRobot = LeggedRobot
        cls.LeggedRobotBox = LeggedRobotBox
        cls.env_cfg = Go2BoxParkourCfg
        cls.train_cfg = Go2BoxParkourCfgPPO
        cls.debug_cfg = DebugGo2BoxCfg
        cls.walk_cfg = Go2RoughCfg
        cls.update_cfg_from_args = staticmethod(update_cfg_from_args)

    def test_event_scale_values_after_control_dt(self):
        scales = self.env_cfg.rewards.scales
        expected_scales = {
            "tracking_ang_vel": 0.2,
            "speed_error_square": -1.0,
            "overspeed": -1.5,
            "lin_pos_y": -0.1,
            "yaw_abs": -0.1,
            "energy_substeps": -2e-7,
            "torques": -1e-7,
            "dof_error_named": -1.0,
            "dof_error": -0.005,
            "collision": -0.05,
            "exceed_dof_pos_limits": -0.1,
            "exceed_torque_limits_l1norm": -0.1,
        }
        for name, expected in expected_scales.items():
            self.assertEqual(getattr(scales, name), expected)
        self.assertFalse(hasattr(scales, "tracking_lin_vel"))
        self.assertFalse(hasattr(scales, "lin_vel_x"))
        dt = 0.02
        self.assertAlmostEqual(scales.box_first_foot_contact * dt, 0.1)
        self.assertAlmostEqual(scales.box_second_foot_contact * dt, 0.2)
        self.assertAlmostEqual(scales.box_passed * dt, 0.5)
        self.assertAlmostEqual(scales.success * dt, 2.0)
        self.assertAlmostEqual(scales.termination * dt, -2.0)
        self.assertAlmostEqual(scales.episode_timeout * dt, -3.0)
        self.assertFalse(self.env_cfg.rewards.only_positive_rewards)
        self.assertFalse(hasattr(scales, "lazy_stop"))

    def test_event_buffers_and_success_timeout_semantics(self):
        env = SimpleNamespace(
            first_foot_contact_buf=torch.tensor([True, False, False]),
            second_foot_contact_buf=torch.tensor([False, True, False]),
            box_passed_buf=torch.tensor([True, False, False]),
            success_buf=torch.tensor([True, False, False]),
            episode_timeout_buf=torch.tensor([False, True, False]),
            reset_buf=torch.tensor([True, True, True]),
            time_out_buf=torch.tensor([False, True, False]),
            box_progress=SimpleNamespace(
                failure_buf=torch.tensor([False, False, True])
            ),
        )
        first_foot = self.LeggedRobotBox._reward_box_first_foot_contact(env)
        second_foot = self.LeggedRobotBox._reward_box_second_foot_contact(env)
        passed = self.LeggedRobotBox._reward_box_passed(env)
        success = self.LeggedRobotBox._reward_success(env)
        timeout = self.LeggedRobotBox._reward_episode_timeout(env)
        termination = self.LeggedRobotBox._reward_termination(env)

        self.assertEqual(first_foot.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(second_foot.tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(passed.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(success.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(timeout.tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(termination.tolist(), [0, 0, 1])

    def make_speed_reward_env(self):
        rewards = SimpleNamespace(
            box_approach_distance=0.5,
            box_exit_distance=0.2,
            box_lateral_margin=0.2,
            box_speed_allowance=0.5,
            flat_speed_limit=0.7,
            flat_severe_speed_limit=0.8,
            box_speed_limit=1.2,
        )
        single_bounds = torch.tensor(
            [
                [2.0, 3.0, -0.5, 0.5, 0.4],
                [4.0, 5.0, -0.5, 0.5, 0.4],
            ]
        )
        bounds = single_bounds.unsqueeze(0).repeat(6, 1, 1)
        root_states = torch.zeros(6, 13)
        root_states[:, 0] = torch.tensor([1.5, 3.2, 3.21, 3.8, 2.5, 2.5])
        root_states[4, 1] = 1.0
        env = object.__new__(self.LeggedRobotBox)
        env.cfg = SimpleNamespace(rewards=rewards)
        env.root_states = root_states
        env.env_box_bounds = bounds
        env.next_box_idx = torch.tensor([0, 0, 0, 0, 0, 2])
        env.base_lin_vel = torch.tensor(
            [[1.0, 0.0, 0.0]]
        ).repeat(6, 1)
        env.commands = torch.tensor(
            [[0.5, 0.0, 0.0]]
        ).repeat(6, 1)
        return env

    def test_only_current_box_opens_speed_window(self):
        env = self.make_speed_reward_env()
        near_box = self.LeggedRobotBox._near_box_for_speed_control(env)

        self.assertEqual(
            near_box.tolist(),
            [True, True, False, False, False, False],
        )

    def test_speed_error_is_square_and_allows_only_near_box_acceleration(self):
        env = self.make_speed_reward_env()
        reward = self.LeggedRobotBox._reward_speed_error_square(env)

        torch.testing.assert_close(
            reward,
            torch.tensor([0.0, 0.0, 0.25, 0.25, 0.25, 0.25]),
        )

        env.base_lin_vel[:, 0] = 0.2
        reward = self.LeggedRobotBox._reward_speed_error_square(env)
        torch.testing.assert_close(reward, torch.full((6,), 0.09))

    def test_overspeed_is_square_with_flat_and_box_margins(self):
        env = self.make_speed_reward_env()
        reward = self.LeggedRobotBox._reward_overspeed(env)

        torch.testing.assert_close(
            reward,
            torch.tensor([0.0, 0.0, 0.09, 0.09, 0.09, 0.09]),
        )

    def test_target_speed_trajectory_beats_waiting_and_flat_sprint(self):
        scales = self.env_cfg.rewards.scales
        dt = 0.02
        command = 0.5
        event_total = 5 * (0.1 + 0.2 + 0.5) + 2.0

        target_return = event_total
        waiting_steps = int(self.env_cfg.env.episode_length_s / dt)
        waiting_return = (
            waiting_steps
            * scales.speed_error_square
            * dt
            * (0.0 - command) ** 2
            + scales.episode_timeout * dt
        )
        sprint_speed = 2.0
        sprint_steps = int(14.0 / sprint_speed / dt)
        sprint_return = event_total + sprint_steps * dt * (
            scales.speed_error_square * (sprint_speed - command) ** 2
            + scales.overspeed
            * (sprint_speed - self.env_cfg.rewards.flat_speed_limit) ** 2
        )

        self.assertGreater(target_return, waiting_return)
        self.assertGreater(target_return, sprint_return)

    def test_speed_statistics_are_finite_and_reset_to_zero(self):
        env = self.make_speed_reward_env()
        num_envs = env.root_states.shape[0]
        float_names = (
            "forward_speed_sum",
            "max_forward_speed",
            "flat_forward_speed_sum",
            "flat_forward_speed_max",
            "box_forward_speed_sum",
            "box_forward_speed_max",
        )
        count_names = (
            "flat_forward_speed_count",
            "box_forward_speed_count",
            "flat_overspeed_count",
            "flat_severe_overspeed_count",
            "box_overspeed_count",
        )
        for name in float_names:
            setattr(env, name, torch.zeros(num_envs))
        for name in count_names:
            setattr(env, name, torch.zeros(num_envs, dtype=torch.long))
        env.episode_length_buf = torch.ones(num_envs, dtype=torch.long)
        env_ids = torch.arange(num_envs)

        near_box = self.LeggedRobotBox._near_box_for_speed_control(env)
        self.LeggedRobotBox._update_speed_statistics(
            env, env.base_lin_vel[:, 0], near_box
        )
        stats = self.LeggedRobotBox._get_speed_statistics(env, env_ids)

        self.assertTrue(all(torch.isfinite(value) for value in stats.values()))
        self.assertGreater(stats["flat_speed_sample_count"].item(), 0.0)
        self.assertGreater(stats["box_speed_sample_count"].item(), 0.0)

        self.LeggedRobotBox._reset_speed_statistics(env, env_ids)
        zero_stats = self.LeggedRobotBox._get_speed_statistics(env, env_ids)
        self.assertTrue(all(torch.isfinite(value) for value in zero_stats.values()))
        self.assertTrue(all(value.item() == 0.0 for value in zero_stats.values()))

    def test_4096_event_reward_shapes(self):
        env = SimpleNamespace(
            first_foot_contact_buf=torch.zeros(4096, dtype=torch.bool),
            second_foot_contact_buf=torch.zeros(4096, dtype=torch.bool),
            box_passed_buf=torch.zeros(4096, dtype=torch.bool),
            success_buf=torch.zeros(4096, dtype=torch.bool),
            episode_timeout_buf=torch.zeros(4096, dtype=torch.bool),
        )
        self.assertEqual(
            self.LeggedRobotBox._reward_box_first_foot_contact(env).shape,
            (4096,),
        )
        self.assertEqual(
            self.LeggedRobotBox._reward_box_second_foot_contact(env).shape,
            (4096,),
        )
        self.assertEqual(
            self.LeggedRobotBox._reward_box_passed(env).shape,
            (4096,),
        )
        self.assertEqual(
            self.LeggedRobotBox._reward_success(env).shape,
            (4096,),
        )
        self.assertEqual(
            self.LeggedRobotBox._reward_episode_timeout(env).shape,
            (4096,),
        )

    def test_commands_are_per_episode_and_in_locked_range(self):
        commands = self.env_cfg.commands
        self.assertEqual(commands.ranges.lin_vel_x, [0.45, 0.55])
        self.assertEqual(commands.ranges.lin_vel_y, [0.0, 0.0])
        self.assertEqual(commands.ranges.ang_vel_yaw, [0.0, 0.0])
        self.assertGreater(commands.resampling_time, self.env_cfg.env.episode_length_s)
        self.assertEqual(self.env_cfg.env.episode_length_s, 45)

    def test_expanded_height_grid_contains_the_original_grid_at_locked_indices(self):
        import numpy as np

        source = self.walk_cfg.terrain
        target = self.env_cfg.terrain
        self.assertEqual(
            (len(source.measured_points_x), len(source.measured_points_y)),
            (21, 11),
        )
        self.assertEqual(
            (len(target.measured_points_x), len(target.measured_points_y)),
            (36, 17),
        )
        np.testing.assert_allclose(
            target.measured_points_x[:21], source.measured_points_x, atol=1e-12
        )
        np.testing.assert_allclose(
            target.measured_points_y[3:14], source.measured_points_y, atol=1e-12
        )

    def test_4096_environments_use_many_physical_tracks(self):
        terrain = self.env_cfg.terrain
        physical_tracks = terrain.num_rows * terrain.num_cols
        self.assertEqual((terrain.num_rows, terrain.num_cols), (8, 64))
        self.assertEqual(physical_tracks, 512)
        self.assertEqual(
            terrain.RandomBoxTrack_kwargs["num_unique_layouts"],
            4,
        )
        self.assertEqual(self.env_cfg.env.num_envs // physical_tracks, 8)

    def test_geometry_debug_task_keeps_four_physical_tracks(self):
        terrain = self.debug_cfg.terrain
        self.assertEqual((terrain.num_rows, terrain.num_cols), (1, 4))
        self.assertNotIn(
            "num_unique_layouts",
            terrain.RandomBoxTrack_kwargs,
        )

    def test_checkpoint_source_is_locked(self):
        runner = self.train_cfg.runner
        algorithm = self.train_cfg.algorithm
        self.assertTrue(runner.resume)
        self.assertEqual(runner.checkpoint, 11700)
        self.assertEqual(
            runner.run_name,
            "five_box_v8_pre_curriculum_from11700",
        )
        self.assertIsNone(runner.ckpt_manipulator)
        self.assertTrue(
            runner.load_run.endswith(
                "Jul19_23-03-03_five_box_v5_stable_from11000"
            )
        )
        self.assertEqual(algorithm.schedule, "fixed")
        self.assertEqual(algorithm.learning_rate, 5e-5)
        self.assertEqual(algorithm.entropy_coef, 0.003)
        self.assertEqual(algorithm.gamma, 0.999)
        self.assertEqual(algorithm.lam, 0.95)
        self.assertEqual(runner.save_interval, 250)
        self.assertEqual(runner.log_interval, 50)

    def test_cli_checkpoint_manipulator_override_is_explicit(self):
        args = SimpleNamespace(
            seed=None,
            max_iterations=None,
            resume=False,
            experiment_name=None,
            run_name=None,
            load_run=None,
            checkpoint=None,
            ckpt_manipulator="reset_critic_and_optimizer",
        )
        cfg = SimpleNamespace(
            runner=SimpleNamespace(ckpt_manipulator=None)
        )

        _, updated = self.update_cfg_from_args(None, cfg, args)
        self.assertEqual(
            updated.runner.ckpt_manipulator,
            "reset_critic_and_optimizer",
        )

        args.ckpt_manipulator = "none"
        _, updated = self.update_cfg_from_args(None, cfg, args)
        self.assertIsNone(updated.runner.ckpt_manipulator)


if __name__ == "__main__":
    unittest.main()
