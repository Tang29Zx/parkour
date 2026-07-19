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
        from legged_gym.envs.go2.debug_go2_box_config import DebugGo2BoxCfg

        cls.LeggedRobot = LeggedRobot
        cls.LeggedRobotBox = LeggedRobotBox
        cls.env_cfg = Go2BoxParkourCfg
        cls.train_cfg = Go2BoxParkourCfgPPO
        cls.debug_cfg = DebugGo2BoxCfg

    def test_event_scale_values_after_control_dt(self):
        scales = self.env_cfg.rewards.scales
        expected_scales = {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 1.0,
            "lin_vel_x": 0.5,
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
        dt = 0.02
        self.assertAlmostEqual(scales.box_first_foot_contact * dt, 0.5)
        self.assertAlmostEqual(scales.box_second_foot_contact * dt, 0.5)
        self.assertAlmostEqual(scales.box_passed * dt, 5.0)
        self.assertAlmostEqual(scales.success * dt, 10.0)
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
            time_out_buf=torch.tensor([True, True, False]),
        )
        first_foot = self.LeggedRobotBox._reward_box_first_foot_contact(env)
        second_foot = self.LeggedRobotBox._reward_box_second_foot_contact(env)
        passed = self.LeggedRobotBox._reward_box_passed(env)
        success = self.LeggedRobotBox._reward_success(env)
        timeout = self.LeggedRobotBox._reward_episode_timeout(env)
        termination = self.LeggedRobot._reward_termination(env)

        self.assertEqual(first_foot.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(second_foot.tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(passed.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(success.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(timeout.tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(termination.tolist(), [0, 0, 1])

    def test_signed_forward_reward(self):
        env = SimpleNamespace(root_states=torch.zeros(2, 13))
        env.root_states[:, 7] = torch.tensor([0.8, -0.3])
        reward = self.LeggedRobotBox._reward_lin_vel_x(env)
        self.assertAlmostEqual(reward[0].item(), 0.8)
        self.assertAlmostEqual(reward[1].item(), -0.3)

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
        self.assertEqual(commands.ranges.lin_vel_x, [0.4, 0.8])
        self.assertEqual(commands.ranges.lin_vel_y, [0.0, 0.0])
        self.assertEqual(commands.ranges.ang_vel_yaw, [0.0, 0.0])
        self.assertGreater(commands.resampling_time, self.env_cfg.env.episode_length_s)

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
        self.assertTrue(runner.resume)
        self.assertEqual(runner.checkpoint, 9700)
        self.assertEqual(runner.run_name, "five_box_contact_v2_from9700")
        self.assertEqual(
            runner.ckpt_manipulator, "reinitialize_height_encoders"
        )
        self.assertTrue(runner.load_run.endswith("Jul19_13-30-09_hold_from_2000_to_10000"))


if __name__ == "__main__":
    unittest.main()
