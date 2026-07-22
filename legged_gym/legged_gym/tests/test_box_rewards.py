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

    def test_one_box_lift_initializer_preserves_warmup_and_caps_kl(self):
        model = OrderedDict(actor=torch.tensor([1.0]))
        source = {
            "model_state_dict": model,
            "optimizer_state_dict": {"warmup": True},
            "reference_model_state_dict": OrderedDict(
                actor=torch.tensor([1.0])
            ),
            "algorithm_state_dict": {
                "critic_warmup_until_iteration": 2100,
                "actor_finetune_active": False,
                "reference_kl_min_coef": 0.01,
                "reference_kl_max_coef": 0.20,
                "current_reference_kl_coef": 0.05,
                "reference_kl_stable_window_count": 3,
                "curriculum_stable_windows": 2,
                "curriculum_regression_windows": 2,
                "collapse_windows": 1,
                "collapse_warning": True,
                "quality_stage_start_iteration": 2000,
            },
            "iter": 2100,
        }
        target = {
            "model_state_dict": OrderedDict(actor=torch.tensor([9.0])),
            "algorithm_state_dict": {
                "reference_kl_min_coef": 0.0,
                "reference_kl_max_coef": 0.02,
                "current_reference_kl_coef": 0.02,
            },
        }

        migrated = self.module.initialize_one_box_lift_from_warmup2100(
            source, target
        )

        torch.testing.assert_close(
            migrated["model_state_dict"]["actor"], torch.tensor([1.0])
        )
        self.assertEqual(
            migrated["optimizer_state_dict"], {"warmup": True}
        )
        state = migrated["algorithm_state_dict"]
        self.assertEqual(state["reference_kl_min_coef"], 0.0)
        self.assertEqual(state["reference_kl_max_coef"], 0.02)
        self.assertEqual(state["current_reference_kl_coef"], 0.02)
        self.assertEqual(state["collapse_windows"], 0)
        self.assertFalse(state["collapse_warning"])

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

    def test_one_box_critic_reset_preserves_task_and_reference_state(self):
        names = (
            "std",
            "actor.0.weight",
            "memory_a.rnn.weight",
            "encoders.0.weight",
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
        task_state = {
            "version": 3,
            "stage": 3,
            "height_level": 0,
            "landing_blend": 0.4,
        }
        reference_state = {"actor.0.weight": torch.full((2, 2), 9.0)}
        algorithm_state = {"current_reference_kl_coef": 0.02}
        source = {
            "model_state_dict": source_model,
            "optimizer_state_dict": {"old": True},
            "lr_scheduler_state_dict": {"old": True},
            "algorithm_state_dict": algorithm_state,
            "reference_model_state_dict": reference_state,
            "task_curriculum_state_dict": task_state,
            "iter": 4000,
            "infos": {"source": "one_box_v187"},
        }

        migrated = self.module.reset_one_box_critic_from4000(
            source, {"model_state_dict": target_model}
        )

        critic_prefixes = ("critic.", "memory_c.", "critic_encoders.")
        for key in names:
            expected = (
                target_model[key]
                if key.startswith(critic_prefixes)
                else source_model[key]
            )
            torch.testing.assert_close(
                migrated["model_state_dict"][key], expected
            )
        self.assertEqual(migrated["task_curriculum_state_dict"], task_state)
        self.assertEqual(
            migrated["algorithm_state_dict"], algorithm_state
        )
        torch.testing.assert_close(
            migrated["reference_model_state_dict"]["actor.0.weight"],
            reference_state["actor.0.weight"],
        )
        self.assertNotIn("optimizer_state_dict", migrated)
        self.assertNotIn("lr_scheduler_state_dict", migrated)

        bad_source = dict(source, iter=3999)
        with self.assertRaises(ValueError):
            self.module.reset_one_box_critic_from4000(
                bad_source, {"model_state_dict": target_model}
            )

    def test_one_box_initializer_expands_actor_scan_and_resets_critic(self):
        source_model = OrderedDict(
            [
                ("std", torch.full((2,), 1.0)),
                ("actor.weight", torch.full((2, 2), 2.0)),
                (
                    "encoders.0.model.0.weight",
                    torch.arange(12, dtype=torch.float32).reshape(3, 4),
                ),
                ("encoders.0.model.0.bias", torch.full((3,), 3.0)),
                ("memory_a.rnn.weight", torch.full((2, 2), 4.0)),
                ("memory_s.rnn.weight", torch.full((2, 2), 5.0)),
                ("state_estimator.model.0.weight", torch.full((2, 2), 6.0)),
                ("critic.weight", torch.full((2, 2), 7.0)),
                ("memory_c.rnn.weight", torch.full((2, 2), 8.0)),
                (
                    "critic_encoders.0.model.0.weight",
                    torch.full((3, 4), 9.0),
                ),
            ]
        )
        target_model = OrderedDict(
            (
                name,
                (
                    torch.full((3, 12), 99.0)
                    if name.endswith("encoders.0.model.0.weight")
                    else torch.full_like(value, 99.0)
                ),
            )
            for name, value in source_model.items()
        )
        source = dict(
            model_state_dict=source_model,
            optimizer_state_dict={"old": True},
            lr_scheduler_state_dict={"old": True},
            iter=2000,
            infos={"source": "rough"},
        )

        migrated = self.module.initialize_one_box_from_rough2000(
            source,
            dict(model_state_dict=target_model),
            source_grid_shape=(2, 2),
            target_grid_shape=(3, 4),
        )

        actor_grid = migrated["model_state_dict"][
            "encoders.0.model.0.weight"
        ].reshape(3, 3, 4)
        source_grid = source_model[
            "encoders.0.model.0.weight"
        ].reshape(3, 2, 2)
        self.assertTrue(torch.equal(actor_grid[:, :2, 1:3], source_grid))
        extra_mask = torch.ones(3, 4, dtype=torch.bool)
        extra_mask[:2, 1:3] = False
        self.assertTrue((actor_grid[:, extra_mask] == 0.0).all())

        for name in (
            "std",
            "actor.weight",
            "encoders.0.model.0.bias",
            "memory_a.rnn.weight",
            "memory_s.rnn.weight",
            "state_estimator.model.0.weight",
        ):
            self.assertTrue(
                torch.equal(migrated["model_state_dict"][name], source_model[name])
            )
        for name in (
            "critic.weight",
            "memory_c.rnn.weight",
            "critic_encoders.0.model.0.weight",
        ):
            self.assertTrue(
                torch.equal(migrated["model_state_dict"][name], target_model[name])
            )
        self.assertNotIn("optimizer_state_dict", migrated)
        self.assertNotIn("lr_scheduler_state_dict", migrated)
        self.assertEqual(migrated["iter"], 2000)

        source["iter"] = 1999
        with self.assertRaises(ValueError):
            self.module.initialize_one_box_from_rough2000(
                source,
                dict(model_state_dict=target_model),
                source_grid_shape=(2, 2),
                target_grid_shape=(3, 4),
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

    def test_flat_reference_kl_initializer_preserves_model_and_optimizer(self):
        source_model = OrderedDict(
            [
                ("actor.weight", torch.ones(2, 2)),
                ("memory_a.rnn.weight", torch.full((2, 2), 2.0)),
                ("critic.weight", torch.full((1, 2), 3.0)),
            ]
        )
        source = {
            "model_state_dict": source_model,
            "reference_model_state_dict": None,
            "optimizer_state_dict": {"kept": True},
            "algorithm_state_dict": {
                "reference_kl_min_coef": 0.0,
                "reference_kl_max_coef": 0.0,
                "current_reference_kl_coef": 0.0,
                "reference_kl_stable_window_count": 7,
            },
            "iter": 2500,
        }
        target = {
            "model_state_dict": OrderedDict(
                (name, torch.zeros_like(value))
                for name, value in source_model.items()
            ),
            "algorithm_state_dict": {
                "reference_kl_min_coef": 0.02,
                "reference_kl_max_coef": 0.02,
                "current_reference_kl_coef": 0.02,
            },
        }

        migrated = self.module.enable_flat_reference_kl_from_one_box2500(
            source, target
        )

        for name, value in source_model.items():
            torch.testing.assert_close(migrated["model_state_dict"][name], value)
        self.assertIsNone(migrated["reference_model_state_dict"])
        self.assertEqual(migrated["optimizer_state_dict"], {"kept": True})
        algorithm = migrated["algorithm_state_dict"]
        self.assertEqual(algorithm["reference_kl_min_coef"], 0.02)
        self.assertEqual(algorithm["reference_kl_max_coef"], 0.02)
        self.assertEqual(algorithm["current_reference_kl_coef"], 0.02)
        self.assertEqual(algorithm["reference_kl_stable_window_count"], 0)

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
        from legged_gym.envs.base.box_progress import BoxProgressTracker
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
        cls.BoxProgressTracker = BoxProgressTracker
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
            "forward_speed_tracking": 0.3,
            "speed_error_square": 0.0,
            "overspeed": -0.2,
            "landing_quality_progress": 150.0,
            "landing_hold_progress": 350.0,
            "landing_deceleration_progress": 200.0,
            "landing_alignment_progress": 250.0,
            "action_rate": -0.005,
            "flat_orientation": -0.2,
            "flat_base_height": 0.0,
            "dof_vel": -5e-5,
            "lin_pos_y": -0.1,
            "yaw_abs": -0.1,
            "dof_error_named": -0.2,
            "dof_error": -0.001,
            "body_collision": -5.0,
            "thigh_collision": -0.2,
            "calf_collision": -0.2,
            "rear_support_missing": -0.2,
            "flat_airborne": -0.1,
            "rear_upper_joint_excursion": 0.0,
            "exceed_torque_limits_l1norm": -1.0,
        }
        for name, expected in expected_scales.items():
            self.assertEqual(getattr(scales, name), expected)
        self.assertFalse(hasattr(scales, "tracking_lin_vel"))
        self.assertFalse(hasattr(scales, "lin_vel_x"))
        dt = 0.02
        self.assertAlmostEqual(scales.box_front_foot_contact * dt, 0.1)
        self.assertAlmostEqual(scales.box_rear_foot_contact * dt, 0.2)
        self.assertAlmostEqual(scales.box_passed * dt, 0.5)
        self.assertAlmostEqual(scales.success * dt, 25.0)
        self.assertAlmostEqual(scales.course_progress * dt, 20.0)
        self.assertAlmostEqual(scales.landing_quality_progress * dt, 3.0)
        self.assertAlmostEqual(scales.landing_hold_progress * dt, 7.0)
        self.assertAlmostEqual(
            scales.landing_deceleration_progress * dt, 4.0
        )
        self.assertAlmostEqual(scales.landing_alignment_progress * dt, 5.0)
        self.assertAlmostEqual(scales.termination * dt, -40.0)
        self.assertAlmostEqual(scales.landing_overrun * dt, -20.0)
        self.assertAlmostEqual(scales.landing_lateral_exit * dt, -25.0)
        self.assertAlmostEqual(scales.landing_timeout * dt, -15.0)
        self.assertAlmostEqual(scales.incomplete * dt, -40.0)
        self.assertEqual(
            self.env_cfg.rewards.forward_speed_tracking_sigma, 0.25
        )
        self.assertEqual(self.env_cfg.rewards.body_collision_floor, 1.0)
        self.assertEqual(self.env_cfg.rewards.leg_collision_floor, 0.25)
        self.assertEqual(self.env_cfg.rewards.flat_orientation_floor, 0.25)
        self.assertEqual(self.env_cfg.rewards.flat_base_height_floor, 1.0)
        self.assertEqual(self.env_cfg.rewards.action_rate_floor, 0.10)
        self.assertEqual(self.env_cfg.rewards.dof_error_floor, 0.10)
        self.assertEqual(self.env_cfg.rewards.dof_vel_floor, 0.10)
        self.assertEqual(self.env_cfg.rewards.overspeed_floor, 1.0)
        self.assertEqual(self.env_cfg.rewards.flat_height_min, 0.28)
        self.assertEqual(self.env_cfg.rewards.flat_height_max, 0.38)
        self.assertEqual(self.env_cfg.rewards.rear_support_window_steps, 25)
        self.assertEqual(self.env_cfg.rewards.rear_support_target_ratio, 0.20)
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
        self.assertEqual(progress.landing_horizontal_speed_threshold, 0.35)
        self.assertEqual(progress.landing_lateral_speed_threshold, 0.20)
        self.assertEqual(progress.landing_lateral_offset_threshold, 0.40)
        self.assertEqual(progress.landing_yaw_threshold, 0.35)
        self.assertEqual(progress.landing_deceleration_start_speed, 1.5)
        self.assertEqual(progress.landing_deadline_steps, 150)
        self.assertEqual(progress.landing_command_ramp_steps, 20)
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
            generic_failure_buf=torch.tensor([False, False, True]),
            task_progress_buf=torch.tensor([1.0, 0.5, 0.25]),
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
        torch.testing.assert_close(
            incomplete, torch.tensor([0.0, 0.875, 0.0])
        )
        torch.testing.assert_close(
            termination, torch.tensor([0.0, 0.0, 0.90625])
        )

    def test_one_box_failure_rewards_do_not_shrink_with_progress(self):
        env = SimpleNamespace(
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(failure_progress_scaling=False)
            ),
            generic_failure_buf=torch.tensor([True, True, False]),
            severe_body_impact_buf=torch.zeros(3, dtype=torch.bool),
            fall_buf=torch.tensor([True, True, False]),
            incomplete_buf=torch.tensor([False, True, True]),
            task_progress_buf=torch.tensor([0.0, 0.95, 1.0]),
        )

        termination = self.LeggedRobotBox._reward_termination(env)
        incomplete = self.LeggedRobotBox._reward_incomplete(env)

        torch.testing.assert_close(
            termination, torch.tensor([1.0, 1.0, 0.0])
        )
        torch.testing.assert_close(
            incomplete, torch.tensor([0.0, 1.0, 1.0])
        )

    def test_one_box_terminal_reward_classes_do_not_stack(self):
        env = SimpleNamespace(
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(failure_progress_scaling=False)
            ),
            generic_failure_buf=torch.tensor([True, True, False, False]),
            severe_body_impact_buf=torch.tensor(
                [True, False, False, False]
            ),
            fall_buf=torch.tensor([True, True, False, False]),
            stagnation_buf=torch.tensor([False, False, True, False]),
            incomplete_buf=torch.tensor([False, False, False, True]),
            task_progress_buf=torch.zeros(4),
        )
        dt = 0.02
        ordinary = (
            self.LeggedRobotBox._reward_termination(env)
            * self.one_box_cfg.rewards.scales.termination
            * dt
        )
        severe = (
            self.LeggedRobotBox._reward_severe_body_impact(env)
            * self.one_box_cfg.rewards.scales.severe_body_impact
            * dt
        )
        stagnation = (
            self.LeggedRobotBox._reward_stagnation(env)
            * self.one_box_cfg.rewards.scales.stagnation
            * dt
        )
        incomplete = (
            self.LeggedRobotBox._reward_incomplete(env)
            * self.one_box_cfg.rewards.scales.incomplete
            * dt
        )

        torch.testing.assert_close(
            ordinary, torch.tensor([0.0, -45.0, 0.0, 0.0])
        )
        torch.testing.assert_close(
            severe, torch.tensor([-50.0, 0.0, 0.0, 0.0])
        )
        torch.testing.assert_close(
            stagnation, torch.tensor([0.0, 0.0, -50.0, 0.0])
        )
        torch.testing.assert_close(
            incomplete, torch.tensor([0.0, 0.0, 0.0, -50.0])
        )

    def test_landing_terminal_rewards_are_exclusive_and_have_stage_a_values(self):
        env = SimpleNamespace(
            landing_timeout_buf=torch.tensor([True, False, False]),
            landing_overrun_buf=torch.tensor([False, True, False]),
            landing_lateral_exit_buf=torch.tensor([False, False, True]),
            generic_failure_buf=torch.tensor([False, False, False]),
            incomplete_buf=torch.tensor([False, False, False]),
            task_progress_buf=torch.ones(3),
        )
        dt = 0.02

        timeout = (
            self.LeggedRobotBox._reward_landing_timeout(env)
            * self.env_cfg.rewards.scales.landing_timeout
            * dt
        )
        overrun = (
            self.LeggedRobotBox._reward_landing_overrun(env)
            * self.env_cfg.rewards.scales.landing_overrun
            * dt
        )
        lateral_exit = (
            self.LeggedRobotBox._reward_landing_lateral_exit(env)
            * self.env_cfg.rewards.scales.landing_lateral_exit
            * dt
        )
        termination = self.LeggedRobotBox._reward_termination(env)
        incomplete = self.LeggedRobotBox._reward_incomplete(env)

        torch.testing.assert_close(timeout, torch.tensor([-15.0, 0.0, 0.0]))
        torch.testing.assert_close(overrun, torch.tensor([0.0, -20.0, 0.0]))
        torch.testing.assert_close(
            lateral_exit, torch.tensor([0.0, 0.0, -25.0])
        )
        torch.testing.assert_close(termination, torch.zeros(3))
        torch.testing.assert_close(incomplete, torch.zeros(3))

    def test_failure_and_incomplete_progress_multipliers(self):
        env = SimpleNamespace(
            generic_failure_buf=torch.ones(3, dtype=torch.bool),
            incomplete_buf=torch.ones(3, dtype=torch.bool),
            task_progress_buf=torch.tensor([0.0, 0.5, 1.0]),
        )
        dt = 0.02
        failure = (
            self.LeggedRobotBox._reward_termination(env)
            * self.env_cfg.rewards.scales.termination
            * dt
        )
        incomplete = (
            self.LeggedRobotBox._reward_incomplete(env)
            * self.env_cfg.rewards.scales.incomplete
            * dt
        )

        torch.testing.assert_close(
            failure, torch.tensor([-40.0, -32.5, -25.0])
        )
        torch.testing.assert_close(
            incomplete, torch.tensor([-40.0, -35.0, -30.0])
        )

    def test_landing_phase_ramps_episode_command_to_zero_in_twenty_steps(self):
        env = SimpleNamespace(
            next_box_idx=torch.tensor([4, 5, 5, 5]),
            box_progress=SimpleNamespace(required_boxes=5),
            landing_command_ramp_steps=20,
            steps_in_landing_phase=torch.tensor([0, 0, 10, 20]),
            episode_command_x=torch.tensor([0.5, 0.5, 0.5, 0.5]),
            commands=torch.tensor(
                [
                    [0.5, 0.0, 0.0],
                    [0.1, 0.1, 0.2],
                    [0.1, 0.1, 0.2],
                    [0.1, 0.1, 0.2],
                ]
            ),
        )
        self.LeggedRobotBox._apply_landing_commands(env)
        torch.testing.assert_close(
            env.commands,
            torch.tensor(
                [
                    [0.5, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.25, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        )

        env.steps_in_landing_phase[-1] = 30
        env.commands[-1, 0] = 0.5
        self.LeggedRobotBox._apply_landing_commands(env)
        self.assertEqual(env.commands[-1, 0].item(), 0.0)

    def test_one_box_landing_keeps_the_episode_forward_command(self):
        env = SimpleNamespace(
            next_box_idx=torch.tensor([0, 1]),
            box_progress=SimpleNamespace(required_boxes=1),
            stop_command_after_course=False,
            episode_command_x=torch.tensor([0.5, 0.55]),
            commands=torch.tensor(
                [[0.5, 0.0, 0.0], [0.0, 0.2, 0.3]]
            ),
        )

        self.LeggedRobotBox._apply_landing_commands(env)

        torch.testing.assert_close(
            env.commands,
            torch.tensor([[0.5, 0.0, 0.0], [0.55, 0.0, 0.0]]),
        )

    def test_episode_command_is_saved_for_reset_environments(self):
        env = object.__new__(self.LeggedRobotBox)
        env.commands = torch.tensor(
            [[0.5, 0.0, 0.0], [0.6, 0.0, 0.0], [0.7, 0.0, 0.0]]
        )
        env.episode_command_x = torch.tensor([0.1, 0.2, 0.3])

        self.LeggedRobotBox._store_episode_commands(
            env, torch.tensor([0, 2])
        )

        torch.testing.assert_close(
            env.episode_command_x, torch.tensor([0.5, 0.2, 0.7])
        )

    def test_landing_guidance_rewards_only_new_high_water_marks(self):
        env = object.__new__(self.LeggedRobotBox)
        env.next_box_idx = torch.tensor([5])
        env.box_progress = SimpleNamespace(required_boxes=5)
        env.landing_phase_entry_buf = torch.tensor([True])
        env.root_states = torch.zeros(1, 13)
        env.root_states[:, 7] = 1.5
        env.root_states[:, 8] = 0.20
        env.root_states[:, 1] = 0.8
        env.env_origins = torch.zeros(1, 3)
        env.landing_deceleration_start_speed = 1.5
        env.cfg = SimpleNamespace(
            box_progress=SimpleNamespace(
                landing_horizontal_speed_threshold=0.35,
            )
        )
        env.box_progress.landing_yaw_threshold = 0.35
        env.box_progress.landing_lateral_speed_threshold = 0.20
        env.box_progress.landing_lateral_offset_threshold = 0.40
        env.landing_entry_horizontal_speed = torch.zeros(1)
        env.landing_deceleration_best = torch.zeros(1)
        env.landing_deceleration_delta = torch.zeros(1)
        env.landing_alignment_start = torch.zeros(1)
        env.landing_alignment_best = torch.zeros(1)
        env.landing_alignment_delta = torch.zeros(1)

        self.LeggedRobotBox._update_landing_guidance(
            env, torch.tensor([0.35])
        )
        self.assertEqual(env.landing_deceleration_delta.item(), 0.0)
        self.assertEqual(env.landing_alignment_delta.item(), 0.0)

        env.landing_phase_entry_buf[:] = False
        env.root_states[:, 7] = 0.35
        env.root_states[:, 8] = 0.0
        env.root_states[:, 1] = 0.0
        self.LeggedRobotBox._update_landing_guidance(
            env, torch.tensor([0.0])
        )
        self.assertAlmostEqual(env.landing_deceleration_delta.item(), 1.0)
        self.assertAlmostEqual(env.landing_alignment_delta.item(), 1.0)

        env.root_states[:, 7] = 1.0
        env.root_states[:, 1] = 0.4
        self.LeggedRobotBox._update_landing_guidance(
            env, torch.tensor([0.3])
        )
        torch.testing.assert_close(
            env.landing_deceleration_delta, torch.zeros(1)
        )
        torch.testing.assert_close(env.landing_alignment_delta, torch.zeros(1))

    def test_landing_guidance_rewards_a_good_entry_without_bad_setup(self):
        env = object.__new__(self.LeggedRobotBox)
        env.next_box_idx = torch.tensor([5])
        env.box_progress = SimpleNamespace(required_boxes=5)
        env.landing_phase_entry_buf = torch.tensor([True])
        env.root_states = torch.zeros(1, 13)
        env.root_states[:, 7] = 0.35
        env.env_origins = torch.zeros(1, 3)
        env.landing_deceleration_start_speed = 1.5
        env.cfg = SimpleNamespace(
            box_progress=SimpleNamespace(
                landing_horizontal_speed_threshold=0.35,
            )
        )
        env.box_progress.landing_yaw_threshold = 0.35
        env.box_progress.landing_lateral_speed_threshold = 0.20
        env.box_progress.landing_lateral_offset_threshold = 0.40
        env.landing_entry_horizontal_speed = torch.zeros(1)
        env.landing_deceleration_best = torch.zeros(1)
        env.landing_deceleration_delta = torch.zeros(1)
        env.landing_alignment_start = torch.zeros(1)
        env.landing_alignment_best = torch.zeros(1)
        env.landing_alignment_delta = torch.zeros(1)

        self.LeggedRobotBox._update_landing_guidance(env, torch.zeros(1))

        self.assertAlmostEqual(env.landing_deceleration_delta.item(), 1.0)
        self.assertAlmostEqual(env.landing_alignment_delta.item(), 1.0)

    def test_normalized_torque_excess_uses_mean_not_joint_sum(self):
        env = SimpleNamespace(
            torque_limits=torch.ones(2),
            substep_torques=torch.tensor(
                [[[2.0, 2.0], [2.0, 2.0]]]
            ),
            cfg=SimpleNamespace(rewards=SimpleNamespace(soft_torque_limit=1.0)),
        )
        two_joint = self.LeggedRobotBox._reward_exceed_torque_limits_l1norm(
            env
        )
        env.torque_limits = torch.ones(4)
        env.substep_torques = torch.tensor(
            [[[2.0] * 4, [2.0] * 4]]
        )
        four_joint = self.LeggedRobotBox._reward_exceed_torque_limits_l1norm(
            env
        )
        torch.testing.assert_close(two_joint, torch.full((1,), 0.25))
        torch.testing.assert_close(four_joint, torch.full((1,), 0.25))

    def test_normalized_torque_excess_is_capped_at_quarter(self):
        env = SimpleNamespace(
            torque_limits=torch.ones(2),
            substep_torques=torch.full((1, 3, 2), 100.0),
            cfg=SimpleNamespace(rewards=SimpleNamespace(soft_torque_limit=1.0)),
        )

        raw = self.LeggedRobotBox._reward_exceed_torque_limits_l1norm(env)

        torch.testing.assert_close(raw, torch.full((1,), 0.25))

    def test_rear_excursion_is_four_joint_mean(self):
        env = SimpleNamespace(
            dof_pos=torch.ones(1, 4),
            default_dof_pos=torch.zeros(1, 4),
            rear_upper_joint_indices=torch.arange(4),
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(
                    flat_hip_allowance=0.0,
                    flat_thigh_allowance=0.0,
                    box_hip_allowance=0.0,
                    box_thigh_allowance=0.0,
                )
            ),
        )
        raw = self.LeggedRobotBox._rear_upper_joint_excursion_raw(
            env, torch.tensor([False])
        )
        torch.testing.assert_close(raw, torch.ones(1))

    def test_failure_cost_is_reduced_by_spatial_progress(self):
        env = SimpleNamespace(
            generic_failure_buf=torch.tensor([True, True, True]),
            task_progress_buf=torch.tensor([0.0, 0.0, 0.8]),
        )
        multiplier = self.LeggedRobotBox._reward_termination(env)
        actual = multiplier * self.env_cfg.rewards.scales.termination * 0.02

        torch.testing.assert_close(
            actual, torch.tensor([-40.0, -40.0, -28.0])
        )
        self.assertEqual(actual[0].item(), actual[1].item())
        self.assertGreater(actual[2].item(), actual[0].item())

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
            flat_orientation_floor=0.25,
            flat_base_height_floor=1.0,
            action_rate_floor=0.10,
            dof_error_floor=0.10,
            dof_vel_floor=0.10,
            speed_error_floor=0.0,
            overspeed_floor=1.0,
            rear_support_window_steps=25,
            rear_support_target_ratio=0.20,
            flat_airborne_free_ratio=0.40,
            flat_height_min=0.28,
            flat_height_max=0.38,
            flat_height_normalization=0.08,
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
        env.feet_indices = torch.arange(4)
        env.rear_foot_local_indices = torch.tensor([2, 3])
        env.contact_forces = torch.zeros(6, 4, 3)
        env.contact_forces[:, [0, 2], 2] = 2.0
        env.cfg.box_progress = SimpleNamespace(contact_force_threshold=1.0)
        return env

    def test_only_current_box_opens_speed_window(self):
        env = self.make_speed_reward_env()
        near_box = self.LeggedRobotBox._near_box_for_speed_control(env)

        self.assertEqual(
            near_box.tolist(),
            [False, True, True, False, False, False],
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
        self.assertEqual(blend[5].item(), 0.0)

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
            torch.full((6,), 0.2),
        )
        env.motion_quality_level = 0.5
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_action_rate(env),
            torch.full((6,), 1.1),
        )

    def test_motion_quality_floors_remain_enabled_at_level_zero(self):
        env = self.make_speed_reward_env()
        env.motion_quality_level = 0.0
        expected = {
            "body_collision_floor": 1.0,
            "leg_collision_floor": 0.25,
            "flat_orientation_floor": 0.25,
            "flat_base_height_floor": 1.0,
            "action_rate_floor": 0.10,
            "dof_error_floor": 0.10,
            "dof_vel_floor": 0.10,
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

    def test_rear_support_and_airborne_windows_close_the_flight_loophole(self):
        env = object.__new__(self.LeggedRobotBox)
        env.contact_forces = torch.zeros(3, 4, 3)
        env.feet_indices = torch.arange(4)
        env.rear_foot_local_indices = torch.tensor([2, 3])
        env.rear_support_missing_count = torch.zeros(3, dtype=torch.long)
        env.max_rear_support_missing_steps = torch.zeros(3, dtype=torch.long)
        env.rear_support_history = torch.zeros(
            3, 25, dtype=torch.bool
        )
        env.airborne_history = torch.zeros_like(env.rear_support_history)
        env.rear_support_valid_history = torch.zeros_like(
            env.rear_support_history
        )
        env.rear_support_window_ratio = torch.zeros(3)
        env.flat_airborne_window_ratio = torch.zeros(3)
        env.rear_support_history_index = 0
        env.episode_length_buf = torch.full((3,), 26, dtype=torch.long)
        env.next_box_idx = torch.zeros(3, dtype=torch.long)
        env.box_progress = SimpleNamespace(required_boxes=2)
        env.current_ground_contact_feet_sum = torch.zeros(3)
        env.current_rear_support_count = torch.zeros(3, dtype=torch.long)
        env.flat_airborne_count = torch.zeros(3, dtype=torch.long)
        env.rear_excursion_raw_sum = torch.zeros(3)
        env.rear_excursion_raw_max = torch.zeros(3)
        env._rear_upper_joint_excursion_raw = lambda box_mask: torch.zeros(3)
        env._box_speed_blend = lambda: torch.tensor([0.0, 1.0, 0.0])
        env.cfg = SimpleNamespace(
            box_progress=SimpleNamespace(contact_force_threshold=1.0),
            rewards=SimpleNamespace(
                rear_support_window_steps=25,
                rear_support_target_ratio=0.20,
                flat_airborne_free_ratio=0.40,
            ),
        )
        # Environment 0 is front-only, environment 1 is inside a box window,
        # and environment 2 is airborne with no supporting feet.
        env.contact_forces[:2, 0, 2] = 2.0
        box_mask = torch.tensor([False, True, False])
        for _ in range(25):
            self.LeggedRobotBox._update_rear_support_state(env, box_mask)

        rear_penalty = self.LeggedRobotBox._reward_rear_support_missing(env)
        airborne_penalty = self.LeggedRobotBox._reward_flat_airborne(env)
        self.assertEqual(rear_penalty.tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(airborne_penalty[0].item(), 0.0)
        self.assertEqual(airborne_penalty[1].item(), 0.0)
        self.assertEqual(airborne_penalty[2].item(), 1.0)

        env.contact_forces[0, 2, 2] = 2.0
        self.LeggedRobotBox._update_rear_support_state(env, box_mask)
        # One rear-foot tap no longer erases the recent front-only history.
        self.assertGreater(
            self.LeggedRobotBox._reward_rear_support_missing(env)[0].item(),
            0.0,
        )

    def test_stage_c_gait_repair_skips_earlier_stages_and_box_window(self):
        env = object.__new__(self.LeggedRobotBox)
        env.uses_task_curriculum = True
        env.landing_blend = 1.0
        env.episode_curriculum_stage = torch.tensor([1, 2, 2])
        env.episode_length_buf = torch.full((3,), 2, dtype=torch.long)
        env.root_states = torch.zeros(3, 13)
        env.env_origins = torch.zeros(3, 3)
        env.actions = torch.ones(3, 2)
        env.last_actions = torch.zeros(3, 2)
        env.motion_quality_level = 0.0
        env._box_speed_blend = lambda: torch.tensor([0.0, 0.0, 1.0])
        env.cfg = SimpleNamespace(
            rewards=SimpleNamespace(
                quality_repair_min_curriculum_stage=2,
                quality_repair_flat_speed_limit=1.2,
                action_rate_flat_only=True,
                action_rate_floor=1.0,
            )
        )

        env.root_states[:, 8] = 1.0
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_lateral_velocity_square(env),
            torch.tensor([0.0, 1.0, 0.0]),
        )
        env.root_states[:, 1] = 0.5
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_flat_lateral_position(env),
            torch.tensor([0.0, 0.5, 0.0]),
        )
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_action_rate(env),
            torch.tensor([0.0, 2.0, 0.0]),
        )

        env.root_states[:, 7] = 2.0
        env.root_states[:, 8] = 0.0
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_world_overspeed(env),
            torch.tensor([0.0, 0.64, 0.0]),
        )

        env.episode_curriculum_stage[:] = torch.tensor([2, 3, 3])
        env.landing_quality_delta = torch.ones(3)
        env.landing_hold_delta = torch.ones(3)
        env.landing_alignment_delta = torch.ones(3)
        env.curriculum_success_buf = torch.tensor([True, True, False])
        env.basic_recovery_buf = torch.tensor([False, True, True])
        env.success_buf = torch.zeros(3, dtype=torch.bool)
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_landing_quality_progress(env),
            torch.tensor([0.0, 1.0, 1.0]),
        )
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_landing_hold_progress(env),
            torch.tensor([0.0, 1.0, 1.0]),
        )
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_landing_alignment_progress(env),
            torch.tensor([0.0, 1.0, 1.0]),
        )
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_recovery_success(env),
            torch.tensor([1.0, 0.0, 0.0]),
        )
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_basic_recovery(env),
            torch.tensor([0.0, 1.0, 1.0]),
        )

        env.landing_blend = 0.4
        torch.testing.assert_close(
            self.LeggedRobotBox._reward_landing_hold_progress(env),
            torch.tensor([0.0, 0.4, 0.4]),
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

    def test_front_foot_lift_progress_is_local_and_non_repeatable(self):
        env = SimpleNamespace(
            num_envs=1,
            device="cpu",
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(
                    scales=SimpleNamespace(
                        front_foot_lift_progress=25.0,
                        front_foot_reach_progress=50.0,
                        rear_foot_lift_progress=25.0,
                        rear_foot_reach_progress=50.0,
                    ),
                    front_foot_lift_approach_distance=0.5,
                    front_foot_lift_clearance=0.03,
                    front_foot_lift_lateral_margin=0.2,
                    front_foot_reach_start_distance=0.25,
                    front_foot_reach_target_inset=0.125,
                    front_foot_reach_height_tolerance=0.02,
                    front_foot_reach_base_overrun=0.15,
                    foot_guidance_min_forward_speed=0.05,
                ),
                box_progress=SimpleNamespace(contact_force_threshold=1.0),
            ),
            box_progress=SimpleNamespace(
                required_boxes=1,
                front_contact_required_steps=2,
                rear_contact_required_steps=2,
                pass_margin=0.15,
                top_contact_tolerance=0.06,
            ),
            next_box_idx=torch.zeros(1, dtype=torch.long),
            front_contact_counter=torch.zeros(1, dtype=torch.long),
            rear_contact_counter=torch.zeros(1, dtype=torch.long),
            front_foot_local_indices=torch.tensor([0, 1]),
            rear_foot_local_indices=torch.tensor([2, 3]),
            env_box_bounds=torch.tensor(
                [[[1.2, 2.4, -0.6, 0.6, 0.15]]]
            ),
            env_origins=torch.zeros(1, 3),
            root_states=torch.zeros(1, 13),
            front_foot_lift_best=torch.zeros(1),
            front_foot_lift_delta=torch.zeros(1),
            front_foot_reach_best=torch.zeros(1),
            front_foot_reach_delta=torch.zeros(1),
            rear_foot_lift_best=torch.zeros(1),
            rear_foot_lift_delta=torch.zeros(1),
            rear_foot_reach_best=torch.zeros(1),
            rear_foot_reach_delta=torch.zeros(1),
            front_foot_guidance_target_idx=torch.full(
                (1,), -1, dtype=torch.long
            ),
        )
        env.root_states[:, 0] = 0.8
        env.root_states[:, 7] = 0.5
        feet_positions = torch.zeros(1, 4, 3)
        feet_positions[:, :2, 2] = 0.09
        feet_positions[:, :2, 0] = 0.96
        feet_terrain_heights = torch.zeros(1, 4)
        feet_forces = torch.zeros(1, 4, 3)
        update = self.LeggedRobotBox._update_foot_guidance_progress

        update(env, feet_positions, feet_terrain_heights, feet_forces)
        self.assertEqual(env.front_foot_lift_delta.item(), 0.0)
        feet_positions[:, 0, 2] = 0.155
        update(env, feet_positions, feet_terrain_heights, feet_forces)
        self.assertAlmostEqual(
            env.front_foot_lift_delta.item(), 0.5, places=6
        )
        update(env, feet_positions, feet_terrain_heights, feet_forces)
        self.assertEqual(env.front_foot_lift_delta.item(), 0.0)

        feet_positions[:, 0, 2] = 0.18
        update(env, feet_positions, feet_terrain_heights, feet_forces)
        self.assertAlmostEqual(
            env.front_foot_lift_delta.item(), 0.5, places=6
        )
        self.assertAlmostEqual(
            env.front_foot_lift_best.item()
            * 25.0
            * 0.02,
            0.5,
        )

        feet_positions[:, 0, 0] = 1.2625
        feet_positions[:, 0, 2] = 0.15
        feet_terrain_heights[:, 0] = 0.15
        feet_forces[:, 0, 2] = 2.0
        update(env, feet_positions, feet_terrain_heights, feet_forces)
        self.assertAlmostEqual(env.front_foot_reach_delta.item(), 0.5)
        update(env, feet_positions, feet_terrain_heights, feet_forces)
        self.assertEqual(env.front_foot_reach_delta.item(), 0.0)
        feet_positions[:, 0, 0] = 1.325
        update(env, feet_positions, feet_terrain_heights, feet_forces)
        self.assertAlmostEqual(env.front_foot_reach_delta.item(), 0.5)
        self.assertAlmostEqual(
            env.front_foot_reach_best.item()
            * 50.0
            * 0.02,
            1.0,
        )

        # Forward placement cannot be rewarded while the foot is only hovering.
        env.front_foot_reach_best.zero_()
        feet_forces.zero_()
        update(env, feet_positions, feet_terrain_heights, feet_forces)
        self.assertEqual(env.front_foot_reach_delta.item(), 0.0)
        feet_forces[:, 0, 2] = 2.0
        feet_positions[:, 0, 1] = 0.7
        update(env, feet_positions, feet_terrain_heights, feet_forces)
        self.assertEqual(env.front_foot_reach_delta.item(), 0.0)
        feet_positions[:, 0, 1] = 0.0

        # Front guidance closes after front contact; only rear guidance opens.
        env.front_foot_lift_best.zero_()
        env.front_foot_reach_best.zero_()
        env.front_contact_counter[:] = 2
        feet_forces.zero_()
        feet_terrain_heights.zero_()
        feet_positions[:, 2, 0] = 0.96
        feet_positions[:, 2, 2] = 0.155
        update(env, feet_positions, feet_terrain_heights, feet_forces)
        self.assertEqual(env.front_foot_lift_delta.item(), 0.0)
        self.assertAlmostEqual(
            env.rear_foot_lift_delta.item(), 0.5, places=6
        )
        feet_positions[:, 2, 0] = 1.325
        feet_positions[:, 2, 2] = 0.15
        feet_terrain_heights[:, 2] = 0.15
        feet_forces[:, 2, 2] = 2.0
        update(env, feet_positions, feet_terrain_heights, feet_forces)
        self.assertAlmostEqual(env.rear_foot_reach_delta.item(), 1.0)

        # Merely standing in the window is not an approach.
        env.rear_foot_lift_best.zero_()
        env.root_states[:, 7] = 0.0
        feet_forces.zero_()
        update(env, feet_positions, feet_terrain_heights, feet_forces)
        self.assertEqual(env.rear_foot_lift_delta.item(), 0.0)

    def test_one_box_clearance_requires_edge_crossing_and_is_non_repeatable(self):
        tracker = self.BoxProgressTracker(
            num_envs=1,
            num_feet=4,
            num_boxes=1,
            required_boxes=1,
            device="cpu",
        )
        env = object.__new__(self.LeggedRobotBox)
        env.num_envs = 1
        env.device = "cpu"
        env.cfg = SimpleNamespace(
            rewards=SimpleNamespace(
                foot_clearance_start_distance=0.30,
                foot_clearance_target_inset=0.12,
                foot_clearance_height=0.03,
                post_front_base_target_fraction=0.65,
            ),
            box_progress=SimpleNamespace(
                contact_force_threshold=1.0,
                stagnation_region_distance=0.6,
                stagnation_steps=125,
            ),
        )
        env.box_progress = tracker
        env.next_box_idx = tracker.next_box_idx
        env.front_contact_counter = tracker.front_contact_counter
        env.rear_contact_counter = tracker.rear_contact_counter
        env.front_foot_local_indices = torch.tensor([0, 1])
        env.rear_foot_local_indices = torch.tensor([2, 3])
        env.env_box_bounds = torch.tensor(
            [[[1.2, 2.8, -0.8, 0.8, 0.10]]]
        )
        env.env_origins = torch.zeros(1, 3)
        env.root_states = torch.zeros(1, 13)
        env.root_states[:, 0] = 0.7
        for name in (
            "box_approach",
            "front_foot_clearance",
            "post_front_base",
            "rear_foot_clearance",
            "box_exit",
        ):
            setattr(env, f"{name}_best", torch.zeros(1))
            setattr(env, f"{name}_delta", torch.zeros(1))
        env.current_box_top_contact_mask = torch.zeros(
            1, 4, dtype=torch.bool
        )
        env.stagnation_counter = torch.zeros(1, dtype=torch.long)
        env.max_stagnation_steps = torch.zeros(1, dtype=torch.long)
        env.stagnation_candidate_buf = torch.zeros(1, dtype=torch.bool)

        feet_positions = torch.zeros(1, 4, 3)
        feet_positions[:, 0] = torch.tensor([0.7, 0.0, 0.30])
        terrain_heights = torch.zeros(1, 4)
        contact_forces = torch.zeros(1, 4, 3)
        update = self.LeggedRobotBox._update_one_box_stage_progress

        update(env, feet_positions, terrain_heights, contact_forces)
        self.assertEqual(env.front_foot_clearance_best.item(), 0.0)

        feet_positions[:, 0] = torch.tensor([1.32, 0.0, 0.14])
        update(env, feet_positions, terrain_heights, contact_forces)
        self.assertEqual(env.front_foot_clearance_best.item(), 1.0)
        self.assertEqual(env.front_foot_clearance_delta.item(), 1.0)
        update(env, feet_positions, terrain_heights, contact_forces)
        self.assertEqual(env.front_foot_clearance_delta.item(), 0.0)

        # Once both groups have qualified, progress to the rear edge is
        # rewarded once and cannot be replayed by stepping backward.
        env.front_contact_counter[:] = 2
        env.rear_contact_counter[:] = 2
        feet_positions[:] = torch.tensor([1.5, 0.0, 0.10])
        terrain_heights[:] = 0.10
        contact_forces[:, :, 2] = 2.0
        env.root_states[:, 0] = 2.595
        update(env, feet_positions, terrain_heights, contact_forces)
        self.assertAlmostEqual(env.box_exit_delta.item(), 0.5, places=5)
        self.assertTrue(env.current_box_top_contact_mask.all().item())
        env.root_states[:, 0] = 2.30
        update(env, feet_positions, terrain_heights, contact_forces)
        self.assertEqual(env.box_exit_delta.item(), 0.0)
        env.root_states[:, 0] = 2.95
        update(env, feet_positions, terrain_heights, contact_forces)
        self.assertAlmostEqual(env.box_exit_delta.item(), 0.5, places=5)
        self.assertAlmostEqual(
            env.box_exit_best.item()
            * self.one_box_cfg.rewards.scales.box_exit_progress
            * 0.02,
            3.0,
        )

    def test_one_box_stagnation_triggers_after_125_no_progress_steps(self):
        tracker = self.BoxProgressTracker(
            num_envs=1,
            num_feet=4,
            num_boxes=1,
            required_boxes=1,
            device="cpu",
        )
        env = object.__new__(self.LeggedRobotBox)
        env.num_envs = 1
        env.device = "cpu"
        env.cfg = SimpleNamespace(
            rewards=SimpleNamespace(
                foot_clearance_start_distance=0.30,
                foot_clearance_target_inset=0.12,
                foot_clearance_height=0.03,
                post_front_base_target_fraction=0.65,
            ),
            box_progress=SimpleNamespace(
                contact_force_threshold=1.0,
                stagnation_region_distance=0.6,
                stagnation_steps=125,
            ),
        )
        env.box_progress = tracker
        env.next_box_idx = tracker.next_box_idx
        env.front_contact_counter = tracker.front_contact_counter
        env.rear_contact_counter = tracker.rear_contact_counter
        env.front_foot_local_indices = torch.tensor([0, 1])
        env.rear_foot_local_indices = torch.tensor([2, 3])
        env.env_box_bounds = torch.tensor(
            [[[1.2, 2.8, -0.8, 0.8, 0.10]]]
        )
        env.env_origins = torch.zeros(1, 3)
        env.root_states = torch.zeros(1, 13)
        env.root_states[:, 0] = 0.7
        for name in (
            "box_approach",
            "front_foot_clearance",
            "post_front_base",
            "rear_foot_clearance",
            "box_exit",
        ):
            setattr(env, f"{name}_best", torch.ones(1))
            setattr(env, f"{name}_delta", torch.zeros(1))
        env.post_front_base_best.zero_()
        env.rear_foot_clearance_best.zero_()
        env.stagnation_counter = torch.zeros(1, dtype=torch.long)
        env.max_stagnation_steps = torch.zeros(1, dtype=torch.long)
        env.stagnation_candidate_buf = torch.zeros(1, dtype=torch.bool)
        env.current_box_top_contact_mask = torch.zeros(
            1, 4, dtype=torch.bool
        )
        feet_positions = torch.zeros(1, 4, 3)
        terrain_heights = torch.zeros(1, 4)
        contact_forces = torch.zeros(1, 4, 3)

        for _ in range(124):
            self.LeggedRobotBox._update_one_box_stage_progress(
                env, feet_positions, terrain_heights, contact_forces
            )
        self.assertFalse(env.stagnation_candidate_buf.item())
        self.LeggedRobotBox._update_one_box_stage_progress(
            env, feet_positions, terrain_heights, contact_forces
        )
        self.assertTrue(env.stagnation_candidate_buf.item())

    def test_one_box_curriculum_smoothly_blends_and_regresses_landing(self):
        env = object.__new__(self.LeggedRobotBox)
        env.uses_task_curriculum = True
        env.box_progress = self.BoxProgressTracker(
            1,
            4,
            1,
            "cpu",
            required_boxes=1,
            recovery_steps=3,
            recovery_min_forward_distance=0.25,
            landing_steps=10,
            landing_min_forward_distance=0.6,
            landing_horizontal_speed_threshold=1.2,
            landing_lateral_speed_threshold=0.35,
            landing_lateral_offset_threshold=0.4,
            landing_yaw_threshold=0.35,
        )
        env.cfg = SimpleNamespace(
            one_box_curriculum=SimpleNamespace(
                state_version=3,
                stage_names=(
                    "front_contact",
                    "rear_contact",
                    "traversal_recovery",
                    "stable_landing",
                ),
                minimum_episodes=256,
                minimum_stage_iterations=100,
                promotion_success_rate=0.65,
                required_stable_windows=2,
                full_height_layouts=(2, 3, 4, 5, 6),
                landing_blend_step=0.1,
                landing_blend_minimum_iterations=100,
                landing_blend_required_stable_windows=2,
                landing_blend_required_regression_windows=2,
                landing_blend_start_steps=3,
                landing_blend_start_min_forward_distance=0.25,
                landing_blend_start_horizontal_speed_threshold=2.5,
                landing_blend_start_lateral_speed_threshold=2.0,
                landing_blend_start_lateral_offset_threshold=0.8,
                landing_blend_start_yaw_threshold=np.pi,
                landing_blend_success_up=0.65,
                landing_blend_box_pass_up=0.90,
                landing_blend_recovery_up=0.80,
                landing_blend_fall_up=0.10,
                landing_blend_stagnation_up=0.08,
                landing_blend_box_pass_down=0.85,
                landing_blend_recovery_down=0.70,
                landing_blend_fall_down=0.15,
                landing_blend_stagnation_down=0.12,
            )
        )
        env.initialize_task_curriculum(2000)
        summary = {
            "one_box_stage_episode_count": torch.tensor(300.0),
            "one_box_stage_success_rate": torch.tensor(0.70),
            "box_1_pass_rate": torch.tensor(0.95),
            "basic_recovery_rate": torch.tensor(0.90),
            "fall_rate": torch.tensor(0.05),
            "stagnation_rate": torch.tensor(0.03),
        }

        self.assertFalse(env.update_task_curriculum(summary, 2100))
        self.assertTrue(env.update_task_curriculum(summary, 2150))
        self.assertEqual(env.one_box_curriculum_stage, 1)
        self.assertFalse(env.update_task_curriculum(summary, 2250))
        self.assertTrue(env.update_task_curriculum(summary, 2300))
        self.assertEqual(env.one_box_curriculum_stage, 2)
        self.assertEqual(env.one_box_height_level, 0)
        self.assertFalse(env.update_task_curriculum(summary, 2400))
        self.assertTrue(env.update_task_curriculum(summary, 2450))
        self.assertEqual(env.one_box_curriculum_stage, 3)
        self.assertEqual(env.one_box_height_level, 0)
        self.assertEqual(env.landing_blend, 0.0)
        self.assertEqual(env.box_progress.landing_steps, 3)
        self.assertFalse(env.update_task_curriculum(summary, 2550))
        self.assertTrue(env.update_task_curriculum(summary, 2600))
        self.assertEqual(env.landing_blend, 0.1)
        self.assertEqual(env.one_box_height_level, 0)

        weak_summary = dict(summary)
        weak_summary.update(
            **{
                "one_box_stage_success_rate": torch.tensor(0.20),
                "box_1_pass_rate": torch.tensor(0.70),
                "basic_recovery_rate": torch.tensor(0.60),
                "stagnation_rate": torch.tensor(0.20),
            }
        )
        self.assertFalse(env.update_task_curriculum(weak_summary, 2700))
        self.assertTrue(env.update_task_curriculum(weak_summary, 2750))
        self.assertEqual(env.landing_blend, 0.0)
        self.assertTrue(env.landing_blend_regressed)

        state = env.get_task_curriculum_state()
        restored = object.__new__(self.LeggedRobotBox)
        restored.uses_task_curriculum = True
        restored.cfg = env.cfg
        restored.box_progress = self.BoxProgressTracker(
            1,
            4,
            1,
            "cpu",
            required_boxes=1,
            landing_steps=10,
            landing_min_forward_distance=0.6,
            landing_horizontal_speed_threshold=1.2,
            landing_lateral_speed_threshold=0.35,
            landing_lateral_offset_threshold=0.4,
            landing_yaw_threshold=0.35,
        )
        restored.load_task_curriculum_state(state)
        self.assertEqual(restored.get_task_curriculum_state(), state)

        legacy_state = dict(state)
        legacy_state.update(version=2, stage=3, stable_windows=1)
        for key in tuple(legacy_state):
            if key.startswith("landing_blend"):
                legacy_state.pop(key)
        restored.load_task_curriculum_state(legacy_state)
        self.assertEqual(restored.one_box_curriculum_stage, 3)
        self.assertEqual(restored.landing_blend, 0.0)
        self.assertEqual(restored.one_box_stable_windows, 0)
        self.assertEqual(restored.get_task_curriculum_state()["version"], 3)

    def test_one_box_final_difficulty_unlocks_seeded_random_heights(self):
        env = object.__new__(self.LeggedRobotBox)
        env.uses_task_curriculum = True
        env.cfg = SimpleNamespace(
            one_box_curriculum=self.one_box_cfg.one_box_curriculum,
            terrain=SimpleNamespace(num_rows=8, num_cols=64),
        )
        env.box_progress = self.BoxProgressTracker(
            32,
            4,
            1,
            "cpu",
            required_boxes=1,
            landing_steps=10,
            landing_min_forward_distance=0.6,
            landing_horizontal_speed_threshold=1.2,
            landing_lateral_speed_threshold=0.35,
            landing_lateral_offset_threshold=0.4,
            landing_yaw_threshold=0.35,
        )
        state = {
            "version": 3,
            "stage": 3,
            "height_level": 0,
            "stage_start_iteration": 3350,
            "stable_windows": 0,
            "landing_blend": 0.4,
            "landing_blend_start_iteration": 3800,
            "landing_blend_stable_windows": 1,
            "landing_blend_regression_windows": 0,
            "landing_blend_resume_pending": False,
        }
        env.load_task_curriculum_state(state)

        self.assertEqual(env.landing_blend, 0.4)
        self.assertEqual(env.one_box_height_level, 4)
        self.assertEqual(env._allowed_one_box_layouts(), (2, 3, 4, 5, 6))
        self.assertTrue(env._randomize_one_box_heights())

        env.device = "cpu"
        env.terrain = SimpleNamespace(num_unique_layouts=7)
        env.terrain_levels = torch.zeros(32, dtype=torch.long)
        env.terrain_types = torch.zeros(32, dtype=torch.long)
        env.terrain_origins = torch.zeros(8, 64, 3)
        env.env_origins = torch.zeros(32, 3)
        env.episode_curriculum_stage = torch.zeros(32, dtype=torch.long)
        env.episode_curriculum_height_level = torch.zeros(
            32, dtype=torch.long
        )
        env_ids = torch.arange(32)

        torch.manual_seed(17)
        env._assign_one_box_layouts(env_ids)
        first_layouts = torch.remainder(
            env.terrain_levels * 64 + env.terrain_types,
            env.terrain.num_unique_layouts,
        )
        self.assertTrue(
            set(first_layouts.tolist()).issubset({2, 3, 4, 5, 6})
        )
        self.assertGreaterEqual(torch.unique(first_layouts).numel(), 4)

        torch.manual_seed(17)
        env._assign_one_box_layouts(env_ids)
        repeated_layouts = torch.remainder(
            env.terrain_levels * 64 + env.terrain_types,
            env.terrain.num_unique_layouts,
        )
        torch.testing.assert_close(first_layouts, repeated_layouts)

        stable_summary = {
            "one_box_stage_episode_count": torch.tensor(300.0),
            "one_box_stage_success_rate": torch.tensor(0.95),
            "box_1_pass_rate": torch.tensor(0.97),
            "basic_recovery_rate": torch.tensor(0.95),
            "fall_rate": torch.tensor(0.02),
            "stagnation_rate": torch.tensor(0.01),
        }
        self.assertFalse(env.update_task_curriculum(stable_summary, 3900))
        self.assertFalse(env.update_task_curriculum(stable_summary, 3950))
        self.assertEqual(env.landing_blend, 0.4)

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
        self.assertEqual(reward[5].item(), 0.0)

        env.commands[0, 0] = 0.0
        env.base_lin_vel[0, 0] = 0.0
        reward = self.LeggedRobotBox._reward_forward_speed_tracking(env)
        self.assertEqual(reward[0].item(), 0.0)

        env.commands[5, 0] = 0.0
        env.base_lin_vel[5, 0] = 0.0
        reward = self.LeggedRobotBox._reward_forward_speed_tracking(env)
        self.assertEqual(reward[5].item(), 0.0)

    def test_world_x_direction_is_capped_and_does_not_reward_waiting(self):
        env = SimpleNamespace(
            root_states=torch.zeros(3, 13),
            commands=torch.tensor(
                [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
            ),
            next_box_idx=torch.tensor([0, 0, 1]),
            box_progress=SimpleNamespace(required_boxes=1),
        )
        env.root_states[:, 6] = 1.0
        env.root_states[:, 7] = torch.tensor([0.0, 0.5, 2.0])
        half_yaw = np.pi / 4.0
        env.root_states[2, 5] = np.sin(half_yaw)
        env.root_states[2, 6] = np.cos(half_yaw)

        reward = self.LeggedRobotBox._reward_world_x_direction(env)

        torch.testing.assert_close(reward, torch.tensor([0.0, 1.0, 0.0]))

    def test_box_approach_overspeed_is_local_and_stops_after_front_contact(self):
        env = SimpleNamespace(
            num_envs=5,
            device="cpu",
            root_states=torch.zeros(5, 13),
            next_box_idx=torch.tensor([0, 0, 0, 1, 0]),
            front_contact_counter=torch.tensor([0, 0, 2, 0, 0]),
            env_box_bounds=torch.tensor(
                [[[1.0, 2.0, -0.8, 0.8, 0.12]]]
            ).repeat(5, 1, 1),
            box_progress=SimpleNamespace(
                required_boxes=1,
                front_contact_required_steps=2,
                pass_margin=0.15,
            ),
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(
                    box_approach_speed_window=0.5,
                    box_approach_speed_limit=1.0,
                    box_approach_speed_normalization=1.0,
                )
            ),
        )
        env.root_states[:, 0] = torch.tensor([0.6, 0.6, 0.6, 0.6, 1.2])
        env.root_states[:, 7] = torch.tensor([2.0, 0.5, 2.0, 2.0, 2.0])

        reward = self.LeggedRobotBox._reward_box_approach_overspeed(env)

        torch.testing.assert_close(
            reward, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        )

    def test_rear_swing_penalties_start_only_after_both_groups_contact(self):
        env = SimpleNamespace(
            next_box_idx=torch.tensor([0, 0, 0, 1]),
            front_contact_counter=torch.tensor([2, 2, 1, 2]),
            rear_contact_counter=torch.tensor([2, 1, 2, 2]),
            box_progress=SimpleNamespace(
                required_boxes=1,
                front_contact_required_steps=2,
                rear_contact_required_steps=2,
            ),
            rear_upper_joint_indices=torch.tensor([0, 1, 2, 3]),
            dof_vel=torch.tensor(
                [
                    [5.0, 8.0, 2.0, 5.0],
                    [8.0, 8.0, 8.0, 8.0],
                    [8.0, 8.0, 8.0, 8.0],
                    [8.0, 8.0, 8.0, 8.0],
                ]
            ),
            actions=torch.tensor(
                [
                    [0.0, 0.60, 0.35, 0.10],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            ),
            last_actions=torch.zeros(4, 4),
            episode_length_buf=torch.full((4,), 2),
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(
                    rear_post_contact_velocity_threshold=5.0,
                    rear_post_contact_velocity_normalization=3.0,
                    rear_post_contact_action_delta_threshold=0.35,
                    rear_post_contact_action_delta_normalization=0.25,
                )
            ),
        )

        velocity_reward = (
            self.LeggedRobotBox._reward_rear_post_contact_velocity(env)
        )
        action_reward = (
            self.LeggedRobotBox._reward_rear_post_contact_action_rate(env)
        )

        torch.testing.assert_close(
            velocity_reward, torch.tensor([0.25, 0.0, 0.0, 0.0])
        )
        torch.testing.assert_close(
            action_reward, torch.tensor([0.25, 0.0, 0.0, 0.0])
        )

    def test_front_box_motion_penalties_are_weak_and_local(self):
        env = SimpleNamespace(
            next_box_idx=torch.tensor([0, 0, 0, 1]),
            box_progress=SimpleNamespace(required_boxes=1),
            front_upper_joint_indices=torch.arange(4),
            dof_vel=torch.tensor(
                [
                    [7.0, 7.0, 7.0, 7.0],
                    [11.0, 7.0, 7.0, 7.0],
                    [11.0, 11.0, 11.0, 11.0],
                    [11.0, 11.0, 11.0, 11.0],
                ]
            ),
            dof_pos=torch.tensor(
                [
                    [0.65, 0.65, 1.35, 1.35],
                    [1.15, 0.65, 1.35, 1.35],
                    [1.15, 1.15, 1.85, 1.85],
                    [1.15, 1.15, 1.85, 1.85],
                ]
            ),
            default_dof_pos=torch.zeros(4, 4),
            actions=torch.tensor(
                [
                    [0.50, 0.50, 0.50, 0.50],
                    [0.85, 0.50, 0.50, 0.50],
                    [0.85, 0.85, 0.85, 0.85],
                    [0.85, 0.85, 0.85, 0.85],
                ]
            ),
            last_actions=torch.zeros(4, 4),
            episode_length_buf=torch.full((4,), 2),
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(
                    front_box_velocity_threshold=7.0,
                    front_box_velocity_normalization=4.0,
                    front_box_action_delta_threshold=0.50,
                    front_box_action_delta_normalization=0.35,
                    front_box_hip_allowance=0.65,
                    front_box_thigh_allowance=1.35,
                    front_box_excursion_normalization=0.50,
                )
            ),
        )
        env._near_box_for_speed_control = lambda: torch.tensor(
            [True, True, False, True]
        )

        velocity = self.LeggedRobotBox._reward_front_box_velocity(env)
        action_rate = self.LeggedRobotBox._reward_front_box_action_rate(env)
        excursion = self.LeggedRobotBox._reward_front_box_excursion(env)

        expected = torch.tensor([0.0, 0.25, 0.0, 0.0])
        torch.testing.assert_close(velocity, expected)
        torch.testing.assert_close(action_rate, expected)
        torch.testing.assert_close(excursion, expected)

    def test_four_foot_box_penalties_require_current_full_top_support(self):
        env = SimpleNamespace(
            next_box_idx=torch.tensor([0, 0, 1]),
            box_progress=SimpleNamespace(required_boxes=1),
            current_box_top_contact_mask=torch.tensor(
                [
                    [True, True, True, True],
                    [True, True, True, False],
                    [True, True, True, True],
                ]
            ),
            all_upper_joint_indices=torch.arange(8),
            dof_vel=torch.tensor(
                [
                    [8.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                    [8.0] * 8,
                    [8.0] * 8,
                ]
            ),
            dof_pos=torch.tensor(
                [
                    [1.05, 0.55, 1.20, 1.20, 0.55, 0.55, 1.20, 1.20],
                    [1.05] * 8,
                    [1.05] * 8,
                ]
            ),
            default_dof_pos=torch.zeros(3, 8),
            actions=torch.tensor(
                [
                    [0.60, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                    [0.60] * 8,
                    [0.60] * 8,
                ]
            ),
            last_actions=torch.zeros(3, 8),
            episode_length_buf=torch.full((3,), 2),
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(
                    all_feet_box_velocity_threshold=4.0,
                    all_feet_box_velocity_normalization=4.0,
                    all_feet_box_action_delta_threshold=0.25,
                    all_feet_box_action_delta_normalization=0.35,
                    all_feet_box_hip_allowance=0.55,
                    all_feet_box_thigh_allowance=1.20,
                    all_feet_box_excursion_normalization=0.50,
                )
            ),
        )

        velocity = self.LeggedRobotBox._reward_all_feet_box_velocity(env)
        action_rate = self.LeggedRobotBox._reward_all_feet_box_action_rate(env)
        excursion = self.LeggedRobotBox._reward_all_feet_box_excursion(env)

        expected = torch.tensor([0.125, 0.0, 0.0])
        torch.testing.assert_close(velocity, expected)
        torch.testing.assert_close(action_rate, expected)
        torch.testing.assert_close(excursion, expected)

    def test_full_box_window_penalizes_all_joints_and_foot_crossing(self):
        num_envs = 3
        thresholds = torch.tensor([5.0, 7.0, 9.0] * 4)
        allowances = torch.tensor([0.45, 1.0, 0.85] * 4)
        env = SimpleNamespace(
            num_envs=num_envs,
            box_joint_velocity_thresholds=thresholds,
            box_joint_excursion_allowances=allowances,
            dof_vel=torch.stack(
                (thresholds, thresholds + 4.0, thresholds + 4.0)
            ),
            dof_pos=torch.stack(
                (allowances, allowances + 0.4, allowances + 0.4)
            ),
            default_dof_pos=torch.zeros(num_envs, 12),
            actions=torch.tensor(
                [[0.30] * 12, [0.60] * 12, [0.60] * 12]
            ),
            last_actions=torch.zeros(num_envs, 12),
            episode_length_buf=torch.full((num_envs,), 2),
            feet_indices=torch.arange(4),
            left_foot_local_indices=torch.tensor([0, 2]),
            right_foot_local_indices=torch.tensor([1, 3]),
            root_states=torch.zeros(num_envs, 13),
            base_quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(
                num_envs, 1
            ),
            all_rigid_body_states=torch.zeros(num_envs, 4, 13),
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(
                    box_joint_velocity_normalization=4.0,
                    box_joint_action_delta_threshold=0.30,
                    box_joint_action_delta_normalization=0.30,
                    box_joint_excursion_normalization=0.40,
                    box_foot_side_margin=0.03,
                    box_foot_crossing_normalization=0.10,
                )
            ),
        )
        env._box_speed_blend = lambda: torch.tensor([1.0, 1.0, 0.0])
        natural_y = torch.tensor([0.20, -0.20, 0.20, -0.20])
        crossed_y = torch.tensor([-0.10, 0.10, -0.10, 0.10])
        env.all_rigid_body_states[0, :, 1] = natural_y
        env.all_rigid_body_states[1:, :, 1] = crossed_y

        velocity = self.LeggedRobotBox._reward_box_joint_velocity(env)
        action_rate = self.LeggedRobotBox._reward_box_joint_action_rate(env)
        excursion = self.LeggedRobotBox._reward_box_joint_excursion(env)
        crossing = self.LeggedRobotBox._reward_box_foot_crossing(env)

        torch.testing.assert_close(velocity, torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(action_rate, torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(excursion, torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(crossing, torch.tensor([0.0, 1.0, 0.0]))

    def test_center_and_direction_shaping_stays_below_terminal_costs(self):
        scales = self.one_box_cfg.rewards.scales
        dt = 0.02
        episode_steps = int(self.one_box_cfg.env.episode_length_s / dt)
        maximum_center_cost = (
            abs(scales.lin_pos_y) + abs(scales.flat_lateral_position)
        ) * 0.8 * dt * episode_steps
        maximum_direction_return = (
            scales.world_x_direction * dt * episode_steps
        )
        self.assertLess(maximum_center_cost, 6.0)
        self.assertLess(maximum_direction_return, 2.0)
        self.assertLess(
            maximum_center_cost,
            abs(scales.landing_timeout * dt),
        )
        self.assertLess(
            maximum_center_cost,
            abs(scales.termination * dt),
        )

    def test_one_box_effort_penalties_are_enabled_conservatively(self):
        scales = self.one_box_cfg.rewards.scales

        self.assertEqual(scales.torques, -1e-7)
        self.assertEqual(scales.energy_substeps, -2e-7)
        self.assertEqual(scales.dof_vel, -5e-5)
        self.assertEqual(scales.exceed_torque_limits_l1norm, -1.0)
        self.assertGreater(scales.box_front_foot_contact, 0.0)
        self.assertGreater(scales.box_rear_foot_contact, 0.0)
        self.assertGreater(scales.box_passed, 0.0)
        self.assertGreater(scales.success, 0.0)

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
        before_landing = (env.next_box_idx < 2).float()
        torch.testing.assert_close(
            reward, 0.25 * (1.0 - blend) * before_landing
        )

    def test_flat_base_height_band_is_support_gated_and_disabled_near_box(self):
        env = self.make_speed_reward_env()
        env.root_states[:, 2] = torch.tensor(
            [0.20, 0.25, 0.30, 0.34, 0.40, 0.50]
        )
        env.terrain = SimpleNamespace(
            get_terrain_heights=lambda points: torch.zeros(points.shape[0])
        )

        reward = self.LeggedRobotBox._reward_flat_base_height(env)
        blend = self.LeggedRobotBox._box_speed_blend(env)
        low = torch.relu((0.28 - env.root_states[:, 2]) / 0.08)
        high = torch.relu((env.root_states[:, 2] - 0.38) / 0.08)
        expected = (
            (low.square() + high.square())
            * (blend <= 0.0)
            * (env.next_box_idx < 2)
        )
        torch.testing.assert_close(reward, expected)

    def test_action_rate_is_zero_for_unchanged_actions(self):
        env = SimpleNamespace(
            actions=torch.tensor([[1.0, -1.0], [0.5, 0.5]]),
            last_actions=torch.tensor([[1.0, -1.0], [0.0, 0.0]]),
            episode_length_buf=torch.tensor([2, 2]),
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(action_rate_flat_only=False)
            ),
        )
        reward = self.LeggedRobotBox._reward_action_rate(env)

        torch.testing.assert_close(reward, torch.tensor([0.0, 0.5]))

        env.episode_length_buf[:] = 1
        reward = self.LeggedRobotBox._reward_action_rate(env)
        torch.testing.assert_close(reward, torch.zeros(2))

    def test_landing_guidance_synthetic_return_order(self):
        scales = self.env_cfg.rewards.scales
        dt = 0.02
        clean_success = 20.0 + 4.0 + 3.0 + 7.0 + 4.0 + 5.0 + 25.0 - 5.0
        landing_timeout = 20.0 + 4.0 + 3.0 + 4.0 + 5.0 - 15.0 - 6.0
        landing_overrun = 20.0 + 4.0 + 3.0 - 20.0 - 6.0
        landing_lateral_exit = 20.0 + 4.0 + 5.0 - 25.0 - 6.0
        late_failure = 18.0 + 3.5 - 26.5 - 6.0
        early_failure = 2.0 + 0.2 - 38.5 - 1.0

        self.assertGreater(clean_success, landing_timeout)
        self.assertGreater(landing_timeout, landing_overrun)
        self.assertGreater(landing_overrun, landing_lateral_exit)
        self.assertGreater(landing_lateral_exit, early_failure)
        self.assertGreater(late_failure, early_failure)
        self.assertEqual(clean_success, 63.0)
        self.assertEqual(scales.landing_quality_progress * dt, 3.0)
        self.assertEqual(scales.landing_hold_progress * dt, 7.0)
        self.assertLess(
            scales.landing_quality_progress * dt,
            scales.landing_hold_progress * dt,
        )
        self.assertEqual(scales.speed_error_square, 0.0)
        self.assertEqual(scales.overspeed, -0.2)

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
            body_contact_step_count=torch.zeros(
                num_envs, dtype=torch.long
            ),
        )
        env.rear_support_history = torch.zeros(
            num_envs, 25, dtype=torch.bool
        )
        env.airborne_history = torch.zeros_like(env.rear_support_history)
        env.rear_support_valid_history = torch.zeros_like(
            env.rear_support_history
        )
        env.rear_support_window_ratio = torch.zeros(num_envs)
        env.flat_airborne_window_ratio = torch.zeros(num_envs)
        env.current_ground_contact_feet_sum = torch.zeros(num_envs)
        env.current_rear_support_count = torch.zeros(
            num_envs, dtype=torch.long
        )
        env.flat_airborne_count = torch.zeros(num_envs, dtype=torch.long)
        env.flat_base_height_count = torch.ones(num_envs, dtype=torch.long)
        env.rear_excursion_raw_sum = torch.zeros(num_envs)
        env.rear_excursion_raw_max = torch.zeros(num_envs)
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
            rewards=SimpleNamespace(
                flat_height_min=0.28,
                flat_height_max=0.38,
                flat_height_normalization=0.08,
            ),
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

    def test_rear_upper_joint_statistics_split_flat_and_box(self):
        env = object.__new__(self.LeggedRobotBox)
        env.num_envs = 2
        env.device = "cpu"
        env.dt = 0.02
        env.rear_upper_joint_indices = torch.arange(4)
        env.rear_upper_joint_names = (
            "RL_hip_joint",
            "RR_hip_joint",
            "RL_thigh_joint",
            "RR_thigh_joint",
        )
        self.LeggedRobotBox._init_rear_upper_joint_statistics(env)
        env.rear_excursion_raw_sum = torch.zeros(2)
        env.rear_excursion_raw_max = torch.zeros(2)

        env.episode_length_buf = torch.ones(2, dtype=torch.long)
        env.dof_pos = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
        )
        env.dof_vel = torch.ones(2, 4)
        env.actions = torch.zeros(2, 4)
        env.last_actions = torch.zeros(2, 4)
        box_mask = torch.tensor([False, True])
        self.LeggedRobotBox._update_rear_upper_joint_statistics(
            env, box_mask
        )

        env.episode_length_buf[:] = 2
        env.dof_pos[:] = torch.tensor(
            [[1.0, 3.0, 5.0, 7.0], [6.0, 8.0, 10.0, 12.0]]
        )
        env.dof_vel[:] = torch.tensor(
            [[3.0] * 4, [5.0] * 4]
        )
        env.actions[:] = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]
        )
        self.LeggedRobotBox._update_rear_upper_joint_statistics(
            env, box_mask
        )

        env_ids = torch.arange(2)
        stats = self.LeggedRobotBox._get_rear_upper_joint_statistics(
            env, env_ids
        )
        self.assertTrue(all(torch.isfinite(value) for value in stats.values()))
        self.assertEqual(stats["flat_RL_hip_joint_mean_range_rad"].item(), 1.0)
        self.assertEqual(stats["box_RL_hip_joint_mean_range_rad"].item(), 2.0)
        self.assertAlmostEqual(
            stats["flat_RL_hip_joint_velocity_rms_rad_s"].item(),
            np.sqrt(5.0),
        )
        self.assertAlmostEqual(
            stats["box_RL_hip_joint_action_delta_rms"].item(), 2.0
        )

        self.LeggedRobotBox._reset_rear_upper_joint_statistics(
            env, env_ids
        )
        reset_stats = self.LeggedRobotBox._get_rear_upper_joint_statistics(
            env, env_ids
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
            generic_failure_buf=torch.zeros(4096, dtype=torch.bool),
            task_progress_buf=torch.zeros(4096),
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

    def test_one_box_task_uses_dedicated_geometry_and_training_source(self):
        env_cfg = self.one_box_cfg
        train_cfg = self.one_box_train_cfg
        terrain = env_cfg.terrain.RandomBoxTrack_kwargs

        self.assertEqual(env_cfg.box_progress.required_boxes, 1)
        self.assertEqual(env_cfg.env.episode_length_s, 15)
        self.assertEqual(terrain["track_length"], 6.2)
        self.assertEqual(terrain["track_width"], 2.0)
        self.assertEqual(terrain["spawn_margin"], 0.6)
        self.assertEqual(terrain["first_gap_range"], (1.2, 1.5))
        self.assertEqual(len(terrain["boxes"]), 1)
        self.assertEqual(
            terrain["boxes"][0]["height_choices"],
            (0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20),
        )
        self.assertEqual(terrain["boxes"][0]["length"], 1.6)
        self.assertEqual(terrain["boxes"][0]["width"], 1.6)
        self.assertEqual(env_cfg.box_progress.min_landing_zone_length, 2.0)
        self.assertEqual(
            env_cfg.box_progress.landing_min_forward_distance, 0.6
        )
        self.assertEqual(env_cfg.box_progress.recovery_steps, 3)
        self.assertEqual(
            env_cfg.box_progress.recovery_min_forward_distance, 0.25
        )
        self.assertEqual(
            env_cfg.box_progress.landing_horizontal_speed_threshold, 1.2
        )
        self.assertEqual(
            env_cfg.box_progress.landing_lateral_speed_threshold, 0.35
        )
        self.assertEqual(env_cfg.box_progress.landing_deadline_steps, 200)
        self.assertFalse(env_cfg.box_progress.stop_command_after_course)

        scales = env_cfg.rewards.scales
        self.assertEqual(scales.forward_speed_tracking, 0.5)
        self.assertEqual(scales.world_x_direction, 0.1)
        self.assertEqual(scales.course_progress, 0.0)
        self.assertEqual(scales.speed_error_square, 0.0)
        self.assertEqual(scales.overspeed, -0.2)
        self.assertEqual(scales.action_rate, -0.005)
        self.assertEqual(scales.rear_support_missing, -0.2)
        self.assertEqual(scales.flat_airborne, -0.2)
        self.assertEqual(scales.lateral_velocity_square, -0.3)
        self.assertEqual(scales.lin_pos_y, -0.15)
        self.assertEqual(scales.flat_lateral_position, -0.3)
        self.assertEqual(scales.flat_yaw_abs, -0.2)
        self.assertEqual(scales.world_overspeed, -0.5)
        self.assertEqual(scales.thigh_collision, -0.1)
        self.assertEqual(scales.calf_collision, -0.5)
        self.assertEqual(scales.box_approach_overspeed, -1.0)
        self.assertEqual(scales.box_joint_velocity, -0.5)
        self.assertEqual(scales.box_joint_action_rate, -0.1)
        self.assertEqual(scales.box_joint_excursion, -0.2)
        self.assertEqual(scales.box_foot_crossing, -0.5)
        self.assertEqual(scales.front_box_velocity, 0.0)
        self.assertEqual(scales.front_box_action_rate, 0.0)
        self.assertEqual(scales.front_box_excursion, 0.0)
        self.assertEqual(scales.rear_post_contact_velocity, 0.0)
        self.assertEqual(scales.rear_post_contact_action_rate, 0.0)
        self.assertEqual(scales.all_feet_box_velocity, 0.0)
        self.assertEqual(scales.all_feet_box_action_rate, 0.0)
        self.assertEqual(scales.all_feet_box_excursion, 0.0)
        self.assertEqual(env_cfg.rewards.box_joint_hip_allowance, 0.55)
        self.assertEqual(env_cfg.rewards.box_joint_thigh_allowance, 1.2)
        self.assertEqual(env_cfg.rewards.box_joint_calf_allowance, 1.0)
        self.assertEqual(scales.front_foot_lift_progress, 0.0)
        self.assertEqual(scales.front_foot_reach_progress, 0.0)
        self.assertEqual(scales.rear_foot_lift_progress, 0.0)
        self.assertEqual(scales.rear_foot_reach_progress, 0.0)
        expected_events = {
            "box_approach_progress": 2.0,
            "front_foot_clearance_progress": 2.0,
            "box_front_foot_contact": 6.0,
            "post_front_base_progress": 3.0,
            "rear_foot_clearance_progress": 2.0,
            "box_exit_progress": 3.0,
            "box_rear_foot_contact": 8.0,
            "box_passed": 10.0,
            "recovery_success": 15.0,
            "basic_recovery": 5.0,
            "success": 15.0,
        }
        for name, expected in expected_events.items():
            self.assertAlmostEqual(getattr(scales, name) * 0.02, expected)
        self.assertAlmostEqual(scales.landing_quality_progress * 0.02, 5.0)
        self.assertAlmostEqual(scales.landing_hold_progress * 0.02, 10.0)
        self.assertEqual(scales.landing_deceleration_progress, 0.0)
        self.assertAlmostEqual(
            scales.landing_alignment_progress * 0.02, 5.0
        )
        for name in (
            "termination",
            "landing_overrun",
            "landing_lateral_exit",
        ):
            self.assertAlmostEqual(getattr(scales, name) * 0.02, -45.0)
        self.assertAlmostEqual(scales.landing_timeout * 0.02, -40.0)
        self.assertAlmostEqual(scales.severe_body_impact * 0.02, -50.0)
        self.assertAlmostEqual(scales.stagnation * 0.02, -50.0)
        self.assertAlmostEqual(scales.incomplete * 0.02, -50.0)
        self.assertFalse(env_cfg.rewards.failure_progress_scaling)
        self.assertEqual(
            env_cfg.rewards.reward_order_mode,
            "success_above_failures",
        )
        self.assertEqual(env_cfg.rewards.foot_clearance_height, 0.04)
        self.assertEqual(env_cfg.rewards.box_approach_speed_limit, 1.0)
        self.assertEqual(env_cfg.rewards.front_box_velocity_threshold, 7.0)
        self.assertEqual(
            env_cfg.rewards.front_box_action_delta_threshold, 0.50
        )
        self.assertEqual(env_cfg.rewards.front_box_hip_allowance, 0.65)
        self.assertEqual(env_cfg.rewards.front_box_thigh_allowance, 1.35)
        self.assertEqual(
            env_cfg.rewards.all_feet_box_velocity_threshold, 4.0
        )
        self.assertEqual(
            env_cfg.rewards.all_feet_box_action_delta_threshold, 0.25
        )
        self.assertEqual(
            env_cfg.rewards.rear_post_contact_velocity_threshold, 5.0
        )
        self.assertEqual(
            env_cfg.rewards.rear_post_contact_action_delta_threshold, 0.35
        )

        runner = train_cfg.runner
        algorithm = train_cfg.algorithm
        self.assertTrue(runner.resume)
        self.assertEqual(runner.checkpoint, 4000)
        self.assertTrue(
            runner.load_run.endswith(
                "Jul21_22-01-16_one_box_v187_from2600"
            )
        )
        self.assertTrue(
            runner.reference_policy_path.endswith(
                "Jul21_18-05-59_one_box_v183_from_rough2000/"
                "model_2100_warmup.pt"
            )
        )
        self.assertEqual(
            runner.run_name,
            "one_box_v1812_final04_random_heights_from4000",
        )
        self.assertIsNone(runner.ckpt_manipulator)
        self.assertEqual(runner.max_iterations, 2000)
        self.assertEqual(runner.save_interval, 50)
        self.assertEqual(runner.log_interval, 50)
        self.assertEqual(algorithm.freeze_actor_encoder_iterations, 100)
        self.assertEqual(algorithm.actor_finetune_learning_rate, 2e-5)
        self.assertEqual(algorithm.reference_kl_min_coef, 0.02)
        self.assertEqual(algorithm.actor_finetune_entropy_coef, 0.002)
        self.assertEqual(algorithm.reference_kl_start_coef, 0.02)
        self.assertEqual(algorithm.reference_kl_max_coef, 0.02)
        self.assertEqual(env_cfg.box_progress.reference_kl_recovery_steps, 10)
        curriculum = env_cfg.one_box_curriculum
        self.assertEqual(curriculum.state_version, 3)
        self.assertEqual(
            curriculum.stage_names,
            (
                "front_contact",
                "rear_contact",
                "traversal_recovery",
                "stable_landing",
            ),
        )
        self.assertEqual(curriculum.low_height_layouts, (0, 1, 2))
        self.assertEqual(curriculum.full_height_layouts, (2, 3, 4, 5, 6))
        self.assertEqual(curriculum.promotion_success_rate, 0.65)
        self.assertEqual(curriculum.landing_blend_step, 0.1)
        self.assertEqual(curriculum.landing_blend_maximum, 0.4)
        self.assertTrue(curriculum.randomize_final_height_layouts)
        self.assertEqual(curriculum.landing_blend_minimum_iterations, 100)
        self.assertEqual(curriculum.landing_blend_start_steps, 3)
        self.assertAlmostEqual(
            curriculum.landing_blend_start_yaw_threshold, np.pi
        )
        self.assertEqual(env_cfg.box_progress.stagnation_steps, 125)
        self.assertEqual(
            env_cfg.rewards.quality_repair_min_curriculum_stage, 2
        )
        self.assertTrue(env_cfg.rewards.action_rate_flat_only)
        self.assertEqual(
            env_cfg.rewards.quality_repair_flat_speed_limit, 1.2
        )
        self.assertEqual(env_cfg.rewards.action_rate_floor, 1.0)
        self.assertEqual(env_cfg.rewards.flat_airborne_free_ratio, 0.20)

    def test_three_and_five_box_tasks_keep_the_five_box_geometry(self):
        stages = (
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

        self.assertFalse(self.three_box_train_cfg.runner.resume)
        self.assertIsNone(self.three_box_train_cfg.runner.ckpt_manipulator)
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
        self.assertEqual(
            terrain.RandomBoxTrack_kwargs["track_length"],
            18.0,
        )
        self.assertEqual(self.env_cfg.viewer.lookat, [14.0, 6.0, 0.2])
        self.assertEqual(self.env_cfg.env.num_envs // physical_tracks, 8)

    def test_reference_kl_mask_ramps_in_after_final_recovery(self):
        env = object.__new__(self.LeggedRobotBox)
        env.root_states = torch.zeros(5, 13)
        env.box_progress = SimpleNamespace(required_boxes=2)
        env.next_box_idx = torch.tensor([0, 0, 2, 1, 1])
        env.passed_box_count = torch.tensor([0, 0, 2, 1, 1])
        env.reference_kl_recovery_counter = torch.tensor([0, 0, 5, 9, 10])
        env.reference_kl_recovery_steps = 10
        env.uses_task_curriculum = True
        env.episode_curriculum_stage = torch.tensor([3, 3, 3, 3, 3])
        env._box_speed_blend = lambda: torch.tensor(
            [0.0, 1.0, 0.0, 0.0, 0.0]
        )

        mask = self.LeggedRobotBox.get_reference_kl_mask(env)

        torch.testing.assert_close(
            mask, torch.tensor([1.0, 0.0, 0.5, 0.0, 1.0])
        )

    def test_geometry_debug_task_keeps_four_physical_tracks(self):
        terrain = self.debug_cfg.terrain
        self.assertEqual((terrain.num_rows, terrain.num_cols), (1, 4))
        self.assertEqual(
            terrain.RandomBoxTrack_kwargs["track_length"],
            15.5,
        )
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
            "five_box_v18_landing_guidance_from11300",
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
        self.assertEqual(algorithm.actor_finetune_learning_rate, 1e-5)
        self.assertEqual(algorithm.freeze_actor_encoder_iterations, 200)
        self.assertEqual(runner.save_interval, 50)
        self.assertEqual(algorithm.critic_warmup_iterations, 100)
        self.assertEqual(algorithm.actor_finetune_learning_rate, 1e-5)
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
        self.assertEqual(runner.max_iterations, 200)
        self.assertEqual(runner.save_interval, 50)
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
