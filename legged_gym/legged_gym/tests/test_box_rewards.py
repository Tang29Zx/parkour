"""Tests for five-box reward configuration and checkpoint migration."""

import importlib.util
from collections import OrderedDict
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np


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

    def test_v11_initializer_preserves_verified_warmup_checkpoint(self):
        model = OrderedDict(
            [("actor.weight", torch.ones(2, 2)), ("critic.weight", torch.ones(1, 2))]
        )
        source = {
            "model_state_dict": model,
            "optimizer_state_dict": {"state": "warm"},
            "reference_model_state_dict": {"actor.weight": torch.ones(2, 2)},
            "algorithm_state_dict": {
                "critic_warmup_until_iteration": 11800,
                "actor_finetune_active": False,
            },
            "iter": 11800,
        }
        target = {
            "model_state_dict": OrderedDict(
                (name, torch.zeros_like(value)) for name, value in model.items()
            ),
            "algorithm_state_dict": {
                "curriculum_state_version": 11,
                "quality_phase": 0,
                "speed_penalty_level": 0.1,
                "motion_quality_level": 0.0,
                "curriculum_stable_windows": 0,
                "curriculum_regression_windows": 0,
                "speed_master_windows": 0,
                "motion_master_windows": 0,
                "reference_kl_stable_window_count": 0,
                "collapse_windows": 0,
                "collapse_warning": False,
                "reward_order_warning": False,
                "reference_kl_min_coef": 0.02,
                "reference_kl_max_coef": 1.0,
                "current_reference_kl_coef": 0.2,
            },
        }

        migrated = self.module.initialize_v11_from_v10_warmup(source, target)

        for name, value in model.items():
            self.assertTrue(torch.equal(migrated["model_state_dict"][name], value))
        self.assertEqual(migrated["optimizer_state_dict"], {"state": "warm"})
        self.assertEqual(
            migrated["algorithm_state_dict"]["curriculum_state_version"], 11
        )
        self.assertEqual(
            migrated["algorithm_state_dict"]["quality_stage_start_iteration"],
            11800,
        )
        self.assertEqual(
            migrated["algorithm_state_dict"]["reference_kl_min_coef"], 0.02
        )

    def test_target_speed_finetune_preserves_checkpoint_and_lowers_kl(self):
        model = OrderedDict(
            [
                ("actor.weight", torch.ones(2, 2)),
                ("critic.weight", torch.full((1, 2), 2.0)),
            ]
        )
        reference = {"actor.weight": torch.full((2, 2), 3.0)}
        source = {
            "model_state_dict": model,
            "optimizer_state_dict": {"state": "trained"},
            "reference_model_state_dict": reference,
            "algorithm_state_dict": {
                "curriculum_state_version": 11,
                "actor_finetune_active": True,
                "speed_penalty_level": 1.0,
                "motion_quality_level": 0.0,
                "reference_kl_min_coef": 0.05,
                "reference_kl_max_coef": 1.0,
                "current_reference_kl_coef": 0.05,
                "reference_kl_stable_window_count": 1,
            },
            "iter": 14100,
        }
        target = {
            "model_state_dict": OrderedDict(
                (name, torch.zeros_like(value)) for name, value in model.items()
            ),
            "algorithm_state_dict": {
                "reference_kl_min_coef": 0.02,
                "reference_kl_max_coef": 1.0,
            },
        }

        migrated = self.module.enable_v11_target_speed_finetune(source, target)

        for name, value in model.items():
            self.assertTrue(torch.equal(migrated["model_state_dict"][name], value))
        self.assertTrue(
            torch.equal(
                migrated["reference_model_state_dict"]["actor.weight"],
                reference["actor.weight"],
            )
        )
        self.assertEqual(migrated["optimizer_state_dict"], {"state": "trained"})
        state = migrated["algorithm_state_dict"]
        self.assertEqual(state["speed_penalty_level"], 1.0)
        self.assertEqual(state["motion_quality_level"], 0.0)
        self.assertEqual(state["reference_kl_min_coef"], 0.02)
        self.assertEqual(state["current_reference_kl_coef"], 0.02)
        self.assertEqual(state["reference_kl_stable_window_count"], 0)
        self.assertEqual(
            source["algorithm_state_dict"]["reference_kl_min_coef"], 0.05
        )

    def test_v14_speed_priority_preserves_checkpoint_and_sets_kl_floor(self):
        model = OrderedDict(
            [
                ("actor.weight", torch.ones(2, 2)),
                ("critic.weight", torch.full((1, 2), 2.0)),
            ]
        )
        reference = {"actor.weight": torch.full((2, 2), 3.0)}
        source = {
            "model_state_dict": model,
            "optimizer_state_dict": {"state": "trained"},
            "reference_model_state_dict": reference,
            "algorithm_state_dict": {
                "curriculum_state_version": 11,
                "actor_finetune_active": True,
                "speed_penalty_level": 1.0,
                "motion_quality_level": 0.0,
                "reference_kl_min_coef": 0.02,
                "reference_kl_max_coef": 1.0,
                "current_reference_kl_coef": 0.02,
                "reference_kl_stable_window_count": 1,
            },
            "iter": 14600,
        }
        target = {
            "model_state_dict": OrderedDict(
                (name, torch.zeros_like(value))
                for name, value in model.items()
            ),
            "algorithm_state_dict": {
                "reference_kl_min_coef": 0.002,
                "reference_kl_max_coef": 1.0,
            },
        }

        migrated = self.module.enable_v14_speed_priority_finetune(
            source,
            target,
        )

        for name, value in model.items():
            self.assertTrue(
                torch.equal(migrated["model_state_dict"][name], value)
            )
        self.assertTrue(
            torch.equal(
                migrated["reference_model_state_dict"]["actor.weight"],
                reference["actor.weight"],
            )
        )
        self.assertEqual(
            migrated["optimizer_state_dict"], {"state": "trained"}
        )
        state = migrated["algorithm_state_dict"]
        self.assertEqual(state["speed_penalty_level"], 1.0)
        self.assertEqual(state["motion_quality_level"], 0.0)
        self.assertEqual(state["reference_kl_min_coef"], 0.002)
        self.assertEqual(state["current_reference_kl_coef"], 0.002)
        self.assertEqual(state["reference_kl_stable_window_count"], 0)
        self.assertEqual(
            source["algorithm_state_dict"]["reference_kl_min_coef"], 0.02
        )

    def test_target_speed_finetune_rejects_non_v11_checkpoint(self):
        source = {
            "model_state_dict": OrderedDict([("actor.weight", torch.ones(1))]),
            "reference_model_state_dict": {"actor.weight": torch.ones(1)},
            "algorithm_state_dict": {
                "curriculum_state_version": 10,
                "actor_finetune_active": True,
            },
        }
        target = {
            "model_state_dict": OrderedDict([("actor.weight", torch.ones(1))]),
            "algorithm_state_dict": {
                "reference_kl_min_coef": 0.02,
                "reference_kl_max_coef": 1.0,
            },
        }
        with self.assertRaises(ValueError):
            self.module.enable_v11_target_speed_finetune(source, target)

    def test_v11_initializer_rejects_non_warmup_checkpoint(self):
        source = {
            "model_state_dict": OrderedDict([("actor.weight", torch.ones(1))]),
            "reference_model_state_dict": {"actor.weight": torch.ones(1)},
            "algorithm_state_dict": {
                "critic_warmup_until_iteration": 11800,
                "actor_finetune_active": False,
            },
            "iter": 12100,
        }
        target = {
            "model_state_dict": OrderedDict([("actor.weight", torch.ones(1))]),
            "algorithm_state_dict": {},
        }
        with self.assertRaises(ValueError):
            self.module.initialize_v11_from_v10_warmup(source, target)

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
            Go2BoxParkour1BoxCfg,
            Go2BoxParkour1BoxCfgPPO,
            Go2BoxParkour3BoxCfg,
            Go2BoxParkour3BoxCfgPPO,
        )
        from legged_gym.envs.go2.go2_config import Go2RoughCfg
        from legged_gym.envs.go2.debug_go2_box_config import DebugGo2BoxCfg
        from legged_gym.utils.helpers import update_cfg_from_args
        from legged_gym.utils.task_registry import task_registry

        cls.LeggedRobot = LeggedRobot
        cls.LeggedRobotBox = LeggedRobotBox
        cls.env_cfg = Go2BoxParkourCfg
        cls.train_cfg = Go2BoxParkourCfgPPO
        cls.one_box_cfg = Go2BoxParkour1BoxCfg
        cls.one_box_train_cfg = Go2BoxParkour1BoxCfgPPO
        cls.three_box_cfg = Go2BoxParkour3BoxCfg
        cls.three_box_train_cfg = Go2BoxParkour3BoxCfgPPO
        cls.debug_cfg = DebugGo2BoxCfg
        cls.walk_cfg = Go2RoughCfg
        cls.update_cfg_from_args = staticmethod(update_cfg_from_args)
        cls.task_registry = task_registry

    def test_event_scale_values_after_control_dt(self):
        scales = self.env_cfg.rewards.scales
        expected_scales = {
            "tracking_ang_vel": 0.2,
            "forward_speed_tracking": 0.5,
            "speed_error_square": 0.0,
            "overspeed": -0.5,
            "action_rate": -0.02,
            "flat_orientation": -0.5,
            "flat_base_height": -10.0,
            "dof_vel": -2e-4,
            "lin_pos_y": -0.1,
            "yaw_abs": -0.1,
            "energy_substeps": -2e-7,
            "torques": -1e-7,
            "dof_error_named": -1.0,
            "dof_error": -0.005,
            "body_collision": -5.0,
            "thigh_collision": -0.5,
            "calf_collision": -0.5,
            "rear_support_missing": -0.5,
            "exceed_dof_pos_limits": -0.1,
            "exceed_torque_limits_l1norm": -0.1,
        }
        for name, expected in expected_scales.items():
            self.assertEqual(getattr(scales, name), expected)
        self.assertFalse(hasattr(scales, "tracking_lin_vel"))
        self.assertFalse(hasattr(scales, "lin_vel_x"))
        dt = 0.02
        self.assertAlmostEqual(scales.box_front_foot_contact * dt, 0.1)
        self.assertAlmostEqual(scales.box_rear_foot_contact * dt, 0.2)
        self.assertAlmostEqual(scales.box_passed * dt, 0.5)
        self.assertAlmostEqual(scales.success * dt, 10.0)
        self.assertAlmostEqual(scales.course_progress * dt, 20.0)
        self.assertAlmostEqual(scales.termination * dt, -40.0)
        self.assertAlmostEqual(scales.incomplete * dt, -40.0)
        self.assertEqual(
            self.env_cfg.rewards.forward_speed_tracking_sigma, 0.25
        )
        self.assertEqual(self.env_cfg.rewards.body_collision_floor, 1.0)
        self.assertEqual(self.env_cfg.rewards.leg_collision_floor, 0.25)
        self.assertEqual(self.env_cfg.rewards.flat_orientation_floor, 0.50)
        self.assertEqual(self.env_cfg.rewards.flat_base_height_floor, 1.0)
        self.assertEqual(self.env_cfg.rewards.action_rate_floor, 0.50)
        self.assertEqual(self.env_cfg.rewards.dof_error_floor, 0.25)
        self.assertEqual(self.env_cfg.rewards.dof_vel_floor, 0.25)
        self.assertEqual(self.env_cfg.rewards.overspeed_floor, 1.0)
        self.assertEqual(self.env_cfg.rewards.flat_base_height_target, 0.34)
        self.assertEqual(self.env_cfg.rewards.rear_support_window_steps, 25)
        self.assertEqual(self.env_cfg.rewards.rear_support_missing_steps, 8)
        progress = self.env_cfg.box_progress
        self.assertEqual(progress.front_contact_required_steps, 2)
        self.assertEqual(progress.rear_contact_required_steps, 2)
        self.assertEqual(progress.body_contact_window_steps, 25)
        self.assertEqual(progress.body_contact_failure_steps, 8)
        self.assertEqual(progress.severe_body_impact_force, 80.0)
        self.assertEqual(progress.landing_min_current_feet, 2)
        self.assertTrue(progress.landing_require_rear_foot)
        self.assertEqual(progress.landing_roll_threshold, 0.35)
        self.assertEqual(progress.landing_pitch_threshold, 0.45)
        self.assertEqual(progress.landing_base_height_threshold, 0.22)
        self.assertEqual(progress.landing_vertical_speed_threshold, 0.5)
        self.assertEqual(progress.flat_low_base_height_threshold, 0.20)
        self.assertEqual(progress.flat_low_base_height_steps, 25)
        self.assertFalse(self.env_cfg.rewards.only_positive_rewards)
        self.assertFalse(hasattr(scales, "lazy_stop"))

    def test_event_buffers_and_success_timeout_semantics(self):
        env = SimpleNamespace(
            front_foot_contact_buf=torch.tensor([True, False, False]),
            rear_foot_contact_buf=torch.tensor([False, True, False]),
            box_passed_buf=torch.tensor([True, False, False]),
            success_buf=torch.tensor([True, False, False]),
            incomplete_buf=torch.tensor([False, True, False]),
            reset_buf=torch.tensor([True, True, True]),
            time_out_buf=torch.tensor([False, True, False]),
            box_progress=SimpleNamespace(
                failure_buf=torch.tensor([False, False, True])
            ),
            task_progress_buf=torch.tensor([1.0, 0.5, 0.25]),
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(failure_progress_floor=1.0)
            ),
        )
        front_foot = self.LeggedRobotBox._reward_box_front_foot_contact(env)
        rear_foot = self.LeggedRobotBox._reward_box_rear_foot_contact(env)
        passed = self.LeggedRobotBox._reward_box_passed(env)
        success = self.LeggedRobotBox._reward_success(env)
        incomplete = self.LeggedRobotBox._reward_incomplete(env)
        termination = self.LeggedRobotBox._reward_termination(env)

        self.assertEqual(front_foot.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(rear_foot.tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(passed.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(success.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(incomplete.tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(termination.tolist(), [0.0, 0.0, 1.0])

    def test_failure_cost_is_constant_across_progress(self):
        env = SimpleNamespace(
            box_progress=SimpleNamespace(
                failure_buf=torch.tensor([True, True, True])
            ),
            task_progress_buf=torch.tensor([0.0, 0.0, 0.8]),
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(failure_progress_floor=1.0)
            ),
        )
        multiplier = self.LeggedRobotBox._reward_termination(env)
        actual = multiplier * self.env_cfg.rewards.scales.termination * 0.02

        torch.testing.assert_close(
            actual, torch.tensor([-40.0, -40.0, -40.0])
        )
        self.assertEqual(actual[0].item(), actual[1].item())
        self.assertEqual(actual[0].item(), actual[2].item())

    def make_speed_reward_env(self):
        rewards = SimpleNamespace(
            box_speed_ramp_up_distance=0.5,
            box_speed_ramp_down_distance=0.5,
            box_lateral_margin=0.2,
            box_speed_allowance=0.5,
            flat_speed_limit=1.2,
            flat_severe_speed_limit=1.5,
            box_speed_limit=1.8,
            forward_speed_tracking_sigma=0.25,
            body_collision_floor=1.0,
            leg_collision_floor=0.25,
            flat_orientation_floor=0.50,
            flat_base_height_floor=1.0,
            action_rate_floor=0.50,
            dof_error_floor=0.25,
            dof_vel_floor=0.25,
            speed_error_floor=0.0,
            overspeed_floor=1.0,
            rear_support_window_steps=25,
            rear_support_missing_steps=8,
            flat_base_height_target=0.34,
        )
        single_bounds = torch.tensor(
            [
                [2.0, 3.0, -0.5, 0.5, 0.4],
                [4.0, 5.0, -0.5, 0.5, 0.4],
            ]
        )
        bounds = single_bounds.unsqueeze(0).repeat(6, 1, 1)
        root_states = torch.zeros(6, 13)
        root_states[:, 0] = torch.tensor([1.5, 3.2, 3.21, 3.8, 2.5, 5.25])
        root_states[4, 1] = 1.0
        env = object.__new__(self.LeggedRobotBox)
        env.cfg = SimpleNamespace(rewards=rewards)
        env.root_states = root_states
        env.env_box_bounds = bounds
        env.box_progress = SimpleNamespace(required_boxes=2)
        env.next_box_idx = torch.tensor([0, 0, 0, 0, 0, 2])
        env.base_lin_vel = torch.tensor(
            [[1.0, 0.0, 0.0]]
        ).repeat(6, 1)
        env.commands = torch.tensor(
            [[0.5, 0.0, 0.0]]
        ).repeat(6, 1)
        env.speed_penalty_level = 1.0
        env.motion_quality_level = 1.0
        return env

    def test_only_current_box_opens_speed_window(self):
        env = self.make_speed_reward_env()
        near_box = self.LeggedRobotBox._near_box_for_speed_control(env)

        self.assertEqual(
            near_box.tolist(),
            [False, True, True, False, False, True],
        )

    def test_speed_limit_blend_is_smooth_and_uses_only_current_box(self):
        env = self.make_speed_reward_env()
        blend = self.LeggedRobotBox._box_speed_blend(env)
        self.assertEqual(blend[0].item(), 0.0)
        self.assertGreater(blend[1].item(), 0.0)
        self.assertLess(blend[1].item(), 1.0)
        self.assertGreater(blend[2].item(), 0.0)
        self.assertEqual(blend[3].item(), 0.0)
        self.assertEqual(blend[4].item(), 0.0)
        self.assertGreater(blend[5].item(), 0.0)
        self.assertLess(blend[5].item(), 1.0)

    def test_speed_error_is_square_and_allows_only_near_box_acceleration(self):
        env = self.make_speed_reward_env()
        reward = self.LeggedRobotBox._reward_speed_error_square(env)

        blend = self.LeggedRobotBox._box_speed_blend(env)
        expected = torch.square(0.5 - 0.5 * blend)
        torch.testing.assert_close(reward, expected)

        env.base_lin_vel[:, 0] = 0.2
        reward = self.LeggedRobotBox._reward_speed_error_square(env)
        torch.testing.assert_close(reward, torch.full((6,), 0.09))

    def test_overspeed_is_square_with_flat_and_box_margins(self):
        env = self.make_speed_reward_env()
        env.speed_penalty_level = 0.0
        env.base_lin_vel[:, 0] = 2.0
        reward = self.LeggedRobotBox._reward_overspeed(env)

        blend = self.LeggedRobotBox._box_speed_blend(env)
        limits = 1.2 + blend * 0.6
        torch.testing.assert_close(reward, torch.square(torch.relu(2.0 - limits)))

    def test_speed_and_motion_levels_scale_separate_penalties(self):
        env = self.make_speed_reward_env()
        env.speed_penalty_level = 0.0
        env.motion_quality_level = 0.0
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_speed_error_square(env),
            torch.zeros(6),
        )
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_overspeed(env),
            torch.square(
                torch.relu(
                    1.0
                    - (
                        1.2
                        + 0.6 * self.LeggedRobotBox._box_speed_blend(env)
                    )
                )
            ),
        )

        env.speed_penalty_level = 0.5
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_speed_error_square(env),
            0.5 * torch.square(
                0.5 - 0.5 * self.LeggedRobotBox._box_speed_blend(env)
            ),
        )
        tracking = self.LeggedRobotBox._reward_forward_speed_tracking(env)
        env.speed_penalty_level = 1.0
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_forward_speed_tracking(env), tracking
        )

        env.actions = torch.ones(6, 2)
        env.last_actions = torch.zeros(6, 2)
        env.episode_length_buf = torch.full((6,), 2)
        env.motion_quality_level = 0.0
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_action_rate(env),
            torch.full((6,), 1.0),
        )
        env.motion_quality_level = 0.5
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_action_rate(env),
            torch.full((6,), 1.5),
        )

    def test_motion_quality_floors_remain_enabled_at_level_zero(self):
        env = self.make_speed_reward_env()
        env.motion_quality_level = 0.0
        expected = {
            "body_collision_floor": 1.0,
            "leg_collision_floor": 0.25,
            "flat_orientation_floor": 0.50,
            "flat_base_height_floor": 1.0,
            "action_rate_floor": 0.50,
            "dof_error_floor": 0.25,
            "dof_vel_floor": 0.25,
        }
        for floor_name, expected_level in expected.items():
            self.assertAlmostEqual(
                self.LeggedRobotBox._effective_motion_quality_level(
                    env, floor_name
                ),
                expected_level,
            )

        env.motion_quality_level = 1.0
        for floor_name in expected:
            self.assertEqual(
                self.LeggedRobotBox._effective_motion_quality_level(
                    env, floor_name
                ),
                1.0,
            )

    def test_rear_support_penalty_ignores_box_maneuvers_and_flight(self):
        env = object.__new__(self.LeggedRobotBox)
        env.contact_forces = torch.zeros(3, 4, 3)
        env.feet_indices = torch.arange(4)
        env.front_foot_local_indices = torch.tensor([0, 1])
        env.rear_foot_local_indices = torch.tensor([2, 3])
        env.rear_support_missing_counter = torch.zeros(3, dtype=torch.long)
        env.rear_support_missing_count = torch.zeros(3, dtype=torch.long)
        env.max_rear_support_missing_steps = torch.zeros(3, dtype=torch.long)
        env.rear_support_missing_history = torch.zeros(
            3, 25, dtype=torch.bool
        )
        env.rear_support_history_index = 0
        env.cfg = SimpleNamespace(
            box_progress=SimpleNamespace(contact_force_threshold=1.0),
            rewards=SimpleNamespace(
                rear_support_window_steps=25,
                rear_support_missing_steps=8,
            ),
        )
        # Environment 0 is front-only, environment 1 is inside a box window,
        # and environment 2 is airborne with no supporting feet.
        env.contact_forces[:2, 0, 2] = 2.0
        box_mask = torch.tensor([False, True, False])
        for _ in range(8):
            self.LeggedRobotBox._update_rear_support_state(env, box_mask)

        penalty = self.LeggedRobotBox._reward_rear_support_missing(env)
        self.assertEqual(penalty.tolist(), [1.0, 0.0, 0.0])

        env.contact_forces[0, 2, 2] = 2.0
        self.LeggedRobotBox._update_rear_support_state(env, box_mask)
        self.assertEqual(env.rear_support_missing_counter[0].item(), 8)
        # One rear-foot tap no longer erases the recent front-only history.
        self.assertEqual(
            self.LeggedRobotBox._reward_rear_support_missing(env)[0].item(),
            1.0,
        )

    def test_course_progress_reward_is_monotonic_and_non_repeatable(self):
        env = SimpleNamespace(
            task_progress_buf=torch.tensor([0.2]),
            progress_reward_start=torch.zeros(1),
            rewarded_progress_ratio=torch.zeros(1),
            course_progress_delta_buf=torch.zeros(1),
            progress_reward_initialized=torch.zeros(1, dtype=torch.bool),
        )
        method = self.LeggedRobotBox._update_course_progress_reward

        method(env)
        self.assertEqual(env.course_progress_delta_buf.item(), 0.0)
        env.task_progress_buf[:] = 0.5
        method(env)
        first_delta = env.course_progress_delta_buf.item()
        env.task_progress_buf[:] = 0.3
        method(env)
        self.assertEqual(env.course_progress_delta_buf.item(), 0.0)
        env.task_progress_buf[:] = 0.5
        method(env)
        self.assertEqual(env.course_progress_delta_buf.item(), 0.0)
        env.task_progress_buf[:] = 1.0
        method(env)

        self.assertAlmostEqual(
            first_delta + env.course_progress_delta_buf.item(), 1.0
        )
        actual_total = (
            env.rewarded_progress_ratio.item()
            * self.env_cfg.rewards.scales.course_progress
            * 0.02
        )
        self.assertAlmostEqual(actual_total, 20.0)

    def test_forward_speed_tracking_peaks_only_at_the_command(self):
        env = self.make_speed_reward_env()
        env.base_lin_vel[:, 0] = torch.tensor(
            [0.5, 1.0, 0.0, 0.4, 0.6, 2.0]
        )
        reward = self.LeggedRobotBox._reward_forward_speed_tracking(env)

        self.assertAlmostEqual(reward[0].item(), 1.0)
        self.assertGreater(reward[3].item(), 0.4)
        self.assertGreater(reward[4].item(), 0.6)
        self.assertGreater(reward[1].item(), 0.3)
        self.assertLess(reward[1].item(), 0.5)
        self.assertEqual(reward[2].item(), 0.0)
        self.assertLess(reward[5].item(), 1e-3)

    def test_zero_yaw_error_has_zero_reward_and_deviation_is_negative(self):
        env = self.make_speed_reward_env()
        env.base_ang_vel = torch.zeros(6, 3)
        env.base_ang_vel[:, 2] = torch.tensor([0.0, 0.1, -0.1, 0.5, -0.5, 1.0])
        env.cfg.rewards.tracking_sigma = 0.25
        reward = self.LeggedRobotBox._reward_tracking_ang_vel(env)
        self.assertEqual(reward[0].item(), 0.0)
        self.assertTrue((reward[1:] < 0.0).all())

    def test_flat_orientation_is_disabled_only_in_current_box_window(self):
        env = self.make_speed_reward_env()
        env.projected_gravity = torch.tensor(
            [[0.3, 0.4, -0.866]]
        ).repeat(6, 1)
        reward = self.LeggedRobotBox._reward_flat_orientation(env)

        blend = self.LeggedRobotBox._box_speed_blend(env)
        torch.testing.assert_close(reward, 0.25 * (1.0 - blend))

    def test_flat_base_height_target_is_disabled_in_box_window(self):
        env = self.make_speed_reward_env()
        env.root_states[:, 2] = torch.tensor(
            [0.20, 0.25, 0.30, 0.34, 0.40, 0.50]
        )
        env.terrain = SimpleNamespace(
            get_terrain_heights=lambda points: torch.zeros(points.shape[0])
        )

        reward = self.LeggedRobotBox._reward_flat_base_height(env)
        blend = self.LeggedRobotBox._box_speed_blend(env)
        expected = torch.square(env.root_states[:, 2] - 0.34) * (
            1.0 - blend
        )
        torch.testing.assert_close(reward, expected)

    def test_action_rate_is_zero_for_unchanged_actions(self):
        env = SimpleNamespace(
            actions=torch.tensor([[1.0, -1.0], [0.5, 0.5]]),
            last_actions=torch.tensor([[1.0, -1.0], [0.0, 0.0]]),
            episode_length_buf=torch.tensor([2, 2]),
        )
        reward = self.LeggedRobotBox._reward_action_rate(env)

        torch.testing.assert_close(reward, torch.tensor([0.0, 0.5]))

        env.episode_length_buf[:] = 1
        reward = self.LeggedRobotBox._reward_action_rate(env)
        torch.testing.assert_close(reward, torch.zeros(2))

    def test_v16_valid_completion_beats_waiting_and_immediate_failure(self):
        scales = self.env_cfg.rewards.scales
        dt = 0.02
        command = 0.5
        course_distance = 3.0
        event_total = 5 * (0.1 + 0.2 + 0.5) + 10.0 + 20.0
        target_steps = int(course_distance / command / dt)
        target_return = event_total + target_steps * dt * (
            scales.forward_speed_tracking
        )
        waiting_steps = int(self.env_cfg.env.episode_length_s / dt)
        waiting_tracking = np.exp(
            -(0.0 - command) ** 2
            / self.env_cfg.rewards.forward_speed_tracking_sigma
        ) * 0.0
        waiting_return = (
            waiting_steps
            * dt
            * scales.forward_speed_tracking
            * waiting_tracking
            + scales.incomplete * dt
        )
        immediate_failure_return = scales.termination * dt

        self.assertGreater(target_return, waiting_return)
        self.assertGreater(target_return, immediate_failure_return)
        self.assertLessEqual(waiting_return, immediate_failure_return)
        self.assertEqual(scales.speed_error_square, 0.0)
        self.assertEqual(scales.overspeed, -0.5)

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

    def test_motion_quality_statistics_are_finite_and_reset_to_zero(self):
        env = object.__new__(self.LeggedRobotBox)
        num_envs = 3
        env.dt = 0.02
        env.episode_length_buf = torch.tensor([1, 50, 100])
        env.box_progress = SimpleNamespace(
            failure_buf=torch.tensor([False, True, True]),
            max_body_contact_window_count=torch.zeros(
                num_envs, dtype=torch.long
            ),
        )
        env.rear_support_missing_history = torch.zeros(
            num_envs, 25, dtype=torch.bool
        )
        for name in (
            "abs_roll_sum",
            "max_abs_roll",
            "abs_pitch_sum",
            "max_abs_pitch",
            "action_rate_l2_sum",
            "max_action_rate_l2",
            "named_dof_error_sum",
            "max_named_dof_error",
            "action_abs_sum",
            "min_lr_foot_lateral_distance",
        ):
            setattr(env, name, torch.zeros(num_envs))
        env.named_joint_error_sum = torch.zeros(num_envs, 2)
        env.named_joint_error_max = torch.zeros(num_envs, 2)
        for name in (
            "dof_near_limit_count",
            "action_saturation_count",
            "left_foot_crossing_count",
            "right_foot_crossing_count",
            "thigh_collision_count",
            "calf_collision_count",
            "body_collision_count",
            "rear_support_missing_counter",
            "rear_support_missing_count",
            "max_rear_support_missing_steps",
        ):
            setattr(env, name, torch.zeros(num_envs, dtype=torch.long))
        env.dof_error_named_indices = torch.tensor([0, 1])
        env.dof_names = ["FL_hip_joint", "FR_hip_joint", "joint_2", "joint_3"]
        env.num_dof = 4
        env.num_actions = 2
        env.left_foot_local_indices = torch.tensor([0, 1])
        env.right_foot_local_indices = torch.tensor([2, 3])
        env.abs_roll_sum[:] = torch.tensor([0.1, 1.0, 2.0])
        env.max_abs_roll[:] = torch.tensor([0.1, 0.2, 0.3])
        env.abs_pitch_sum[:] = torch.tensor([0.2, 2.0, 3.0])
        env.max_abs_pitch[:] = torch.tensor([0.2, 0.3, 0.4])
        env.action_rate_l2_sum[:] = torch.tensor([0.3, 3.0, 4.0])
        env.max_action_rate_l2[:] = torch.tensor([0.3, 0.4, 0.5])
        env_ids = torch.arange(num_envs)

        stats = self.LeggedRobotBox._get_motion_quality_statistics(
            env, env_ids
        )
        self.assertTrue(all(torch.isfinite(value) for value in stats.values()))
        self.assertAlmostEqual(stats["early_failure_rate"].item(), 1.0 / 3.0)
        self.assertAlmostEqual(stats["mean_failure_time_s"].item(), 1.5)

        self.LeggedRobotBox._reset_motion_quality_statistics(env, env_ids)
        reset_stats = self.LeggedRobotBox._get_motion_quality_statistics(
            env, env_ids
        )
        for name in (
            "mean_abs_roll_rad",
            "max_abs_roll_rad",
            "mean_abs_pitch_rad",
            "max_abs_pitch_rad",
            "mean_action_rate_l2",
            "max_action_rate_l2",
        ):
            self.assertEqual(reset_stats[name].item(), 0.0)

    def test_flat_gait_statistics_detect_persistent_low_body(self):
        env = object.__new__(self.LeggedRobotBox)
        num_envs = 2
        num_feet = 4
        env.device = "cpu"
        env.dt = 0.02
        env.cfg = SimpleNamespace(
            rewards=SimpleNamespace(flat_base_height_target=0.34),
            box_progress=SimpleNamespace(
                contact_force_threshold=1.0,
                flat_low_base_height_threshold=0.22,
                flat_low_base_height_steps=3,
            ),
        )
        env.root_states = torch.zeros(num_envs, 13)
        env.root_states[:, 2] = torch.tensor([0.20, 0.34])
        env.terrain = SimpleNamespace(
            get_terrain_heights=lambda points: torch.zeros(points.shape[0])
        )
        env.feet_indices = torch.arange(num_feet)
        env.feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        env.front_foot_local_indices = torch.tensor([0, 1])
        env.rear_foot_local_indices = torch.tensor([2, 3])
        env.contact_forces = torch.zeros(num_envs, num_feet, 3)
        env.last_contact_forces = torch.zeros_like(env.contact_forces)
        env.contact_forces[:, :, 2] = 2.0
        env.last_contact_forces[:, :, 2] = 2.0
        env.flat_base_height_sum = torch.zeros(num_envs)
        env.flat_base_height_count = torch.zeros(num_envs, dtype=torch.long)
        env.min_flat_base_height = torch.full((num_envs,), 0.34)
        env.flat_low_base_height_count = torch.zeros(
            num_envs, dtype=torch.long
        )
        env.flat_low_base_height_counter = torch.zeros(
            num_envs, dtype=torch.long
        )
        env.max_flat_low_base_height_steps = torch.zeros(
            num_envs, dtype=torch.long
        )
        env.flat_low_base_height_failure_buf = torch.zeros(
            num_envs, dtype=torch.bool
        )
        env.flat_foot_contact_count = torch.zeros(
            num_envs, num_feet, dtype=torch.long
        )
        env.flat_foot_contact_transition_count = torch.zeros_like(
            env.flat_foot_contact_count
        )
        env.previous_flat_foot_contact = torch.zeros(
            num_envs, num_feet, dtype=torch.bool
        )
        env.previous_flat_contact_valid = torch.zeros(
            num_envs, dtype=torch.bool
        )

        flat_mask = torch.ones(num_envs, dtype=torch.bool)
        for _ in range(3):
            self.LeggedRobotBox._update_flat_gait_statistics(env, flat_mask)

        self.assertEqual(
            env.flat_low_base_height_failure_buf.tolist(), [True, False]
        )
        stats = self.LeggedRobotBox._get_flat_gait_statistics(
            env, torch.arange(num_envs)
        )
        self.assertTrue(all(torch.isfinite(value) for value in stats.values()))
        self.assertAlmostEqual(stats["flat_base_height_mean_m"].item(), 0.27)
        self.assertAlmostEqual(stats["flat_base_height_min_m"].item(), 0.20)
        self.assertAlmostEqual(
            stats["flat_low_base_height_ratio"].item(), 0.5
        )
        self.assertAlmostEqual(
            stats["flat_low_base_height_failure_rate"].item(), 0.5
        )
        self.assertAlmostEqual(
            stats["flat_rear_contact_duty_ratio"].item(), 1.0
        )

        self.LeggedRobotBox._reset_flat_gait_statistics(
            env, torch.arange(num_envs)
        )
        reset_stats = self.LeggedRobotBox._get_flat_gait_statistics(
            env, torch.arange(num_envs)
        )
        self.assertTrue(
            all(torch.isfinite(value) for value in reset_stats.values())
        )
        self.assertTrue(
            all(value.item() == 0.0 for value in reset_stats.values())
        )

    def test_4096_event_reward_shapes(self):
        env = SimpleNamespace(
            front_foot_contact_buf=torch.zeros(4096, dtype=torch.bool),
            rear_foot_contact_buf=torch.zeros(4096, dtype=torch.bool),
            box_passed_buf=torch.zeros(4096, dtype=torch.bool),
            success_buf=torch.zeros(4096, dtype=torch.bool),
            incomplete_buf=torch.zeros(4096, dtype=torch.bool),
            task_progress_buf=torch.zeros(4096),
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(failure_progress_floor=1.0)
            ),
        )
        self.assertEqual(
            self.LeggedRobotBox._reward_box_front_foot_contact(env).shape,
            (4096,),
        )
        self.assertEqual(
            self.LeggedRobotBox._reward_box_rear_foot_contact(env).shape,
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
            self.LeggedRobotBox._reward_incomplete(env).shape,
            (4096,),
        )

    def test_commands_are_per_episode_and_in_locked_range(self):
        commands = self.env_cfg.commands
        self.assertEqual(commands.ranges.lin_vel_x, [0.45, 0.55])
        self.assertEqual(commands.ranges.lin_vel_y, [0.0, 0.0])
        self.assertEqual(commands.ranges.ang_vel_yaw, [0.0, 0.0])
        self.assertGreater(commands.resampling_time, self.env_cfg.env.episode_length_s)
        self.assertEqual(self.env_cfg.env.episode_length_s, 45)

    def test_curriculum_stage_configs_keep_the_five_box_geometry(self):
        stages = (
            (self.one_box_cfg, self.one_box_train_cfg, 1, 15, "one_box"),
            (
                self.three_box_cfg,
                self.three_box_train_cfg,
                3,
                30,
                "three_box",
            ),
            (self.env_cfg, self.train_cfg, 5, 45, "five_box"),
        )
        reference_boxes = self.env_cfg.terrain.RandomBoxTrack_kwargs["boxes"]
        for env_cfg, train_cfg, required_boxes, episode_length, run_prefix in stages:
            self.assertEqual(env_cfg.box_progress.required_boxes, required_boxes)
            self.assertEqual(env_cfg.env.episode_length_s, episode_length)
            self.assertEqual(
                env_cfg.box_progress.min_landing_zone_length, 0.85
            )
            self.assertEqual(
                env_cfg.terrain.RandomBoxTrack_kwargs["boxes"], reference_boxes
            )
            self.assertEqual(len(reference_boxes), 5)
            self.assertTrue(train_cfg.runner.run_name.startswith(run_prefix))

        self.assertFalse(self.one_box_train_cfg.runner.resume)
        self.assertFalse(self.three_box_train_cfg.runner.resume)
        self.assertIsNone(self.one_box_train_cfg.runner.ckpt_manipulator)
        self.assertIsNone(self.three_box_train_cfg.runner.ckpt_manipulator)
        self.assertEqual(self.one_box_train_cfg.runner.max_iterations, 1000)
        self.assertEqual(self.three_box_train_cfg.runner.max_iterations, 1000)

    def test_curriculum_tasks_are_registered_with_the_box_environment(self):
        expected = {
            "go2_box_parkour_1box": 1,
            "go2_box_parkour_3box": 3,
            "go2_box_parkour": 5,
        }
        for task_name, required_boxes in expected.items():
            self.assertIs(
                self.task_registry.task_classes[task_name],
                self.LeggedRobotBox,
            )
            self.assertEqual(
                self.task_registry.env_cfgs[
                    task_name
                ].box_progress.required_boxes,
                required_boxes,
            )

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
        self.assertFalse(runner.init_at_random_ep_len)
        self.assertTrue(runner.resume)
        self.assertEqual(runner.checkpoint, 11300)
        self.assertEqual(
            runner.run_name,
            "five_box_v16_flat_gait_repair_from11300",
        )
        self.assertIsNone(runner.ckpt_manipulator)
        self.assertTrue(
            runner.load_run.endswith(
                "Jul19_22-22-00_five_box_v4_from11200"
            )
        )
        self.assertEqual(algorithm.schedule, "fixed")
        self.assertEqual(algorithm.learning_rate, 5e-5)
        self.assertEqual(algorithm.entropy_coef, 0.003)
        self.assertEqual(algorithm.gamma, 0.999)
        self.assertEqual(algorithm.lam, 0.95)
        self.assertEqual(algorithm.critic_warmup_iterations, 100)
        self.assertEqual(algorithm.actor_finetune_learning_rate, 2e-5)
        self.assertEqual(algorithm.actor_finetune_clip_param, 0.1)
        self.assertEqual(algorithm.actor_finetune_entropy_coef, 0.001)
        self.assertEqual(algorithm.reference_kl_min_coef, 0.0)
        self.assertEqual(algorithm.reference_kl_max_coef, 0.0)
        self.assertEqual(algorithm.reference_kl_start_coef, 0.0)
        self.assertEqual(algorithm.speed_penalty_initial_level, 0.0)
        self.assertEqual(algorithm.motion_quality_initial_level, 0.0)
        self.assertEqual(algorithm.curriculum_stage_min_iterations, 200)
        self.assertEqual(
            algorithm.actor_parameter_equivalence_tolerance, 0.0
        )
        self.assertEqual(
            algorithm.actor_output_equivalence_tolerance, 1e-5
        )
        self.assertEqual(algorithm.actor_std_equivalence_tolerance, 1e-7)
        self.assertEqual(runner.max_iterations, 300)
        self.assertEqual(runner.save_interval, 100)
        self.assertEqual(runner.log_interval, 10)

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
