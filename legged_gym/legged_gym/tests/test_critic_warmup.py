"""Tests for resumable Critic-only PPO warmup."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

if (
    "tensorboardX" not in sys.modules
    and importlib.util.find_spec("tensorboardX") is None
):
    tensorboard_module = ModuleType("tensorboardX")
    tensorboard_module.SummaryWriter = object
    sys.modules["tensorboardX"] = tensorboard_module

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


CRITIC_PREFIXES = ("critic.", "memory_c.", "critic_encoders.")


class DummyActorCritic(nn.Module):
    """Small module exposing the same parameter ownership as the real policy."""

    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.std = nn.Parameter(torch.ones(1))
        self.actor = nn.Linear(1, 1, bias=False)
        self.memory_a = nn.Linear(1, 1, bias=False)
        self.encoders = nn.ModuleList([nn.Linear(1, 1, bias=False)])
        self.memory_s = nn.Linear(1, 1, bias=False)
        self.state_estimator = nn.Linear(1, 1, bias=False)
        self.critic = nn.Linear(1, 1, bias=False)
        self.memory_c = nn.Linear(1, 1, bias=False)
        self.critic_encoders = nn.ModuleList(
            [nn.Linear(1, 1, bias=False)]
        )


class OneBatchStorage:
    """Minimal storage interface used by ``PPO.update``."""

    def mini_batch_generator(self, num_mini_batches, num_epochs):
        del num_mini_batches, num_epochs
        yield None

    def clear(self):
        pass


class LossControlledPPO(PPO):
    """PPO whose losses directly cover Actor- and Critic-owned parameters."""

    def compute_losses(self, minibatch):
        del minibatch
        actor_loss = torch.zeros((), device=self.device)
        critic_loss = torch.zeros((), device=self.device)
        for name, parameter in self.actor_critic.named_parameters():
            if name.startswith(CRITIC_PREFIXES):
                critic_loss = critic_loss + torch.square(parameter).sum()
            else:
                actor_loss = actor_loss + torch.square(parameter).sum()
        return (
            {
                "surrogate_loss": actor_loss,
                "value_loss": critic_loss,
            },
            {},
            {},
        )


class CriticWarmupTest(unittest.TestCase):
    def make_ppo(self):
        torch.manual_seed(1)
        ppo = LossControlledPPO(
            DummyActorCritic(),
            num_learning_epochs=1,
            num_mini_batches=1,
            learning_rate=1e-2,
            optimizer_class_name="AdamW",
            critic_warmup_iterations=100,
        )
        ppo.storage = OneBatchStorage()
        return ppo

    @staticmethod
    def clone_parameters(ppo):
        return {
            name: parameter.detach().clone()
            for name, parameter in ppo.actor_critic.named_parameters()
        }

    def test_warmup_changes_only_critic_side(self):
        ppo = self.make_ppo()
        ppo.start_critic_warmup(11700)
        before = self.clone_parameters(ppo)

        _, stats = ppo.update(11700)
        after = self.clone_parameters(ppo)

        for name in before:
            if name.startswith(CRITIC_PREFIXES):
                self.assertFalse(torch.equal(before[name], after[name]), name)
            else:
                self.assertTrue(torch.equal(before[name], after[name]), name)
        optimizer_parameter_names = {
            name
            for name, parameter in ppo.actor_critic.named_parameters()
            if parameter in ppo.optimizer.state
        }
        self.assertTrue(optimizer_parameter_names)
        self.assertTrue(
            all(
                name.startswith(CRITIC_PREFIXES)
                for name in optimizer_parameter_names
            )
        )
        self.assertEqual(stats["actor_update_enabled"].item(), 0.0)
        self.assertEqual(stats["critic_warmup_remaining"].item(), 100.0)

    def test_actor_encoder_stays_frozen_for_first_two_hundred_iterations(self):
        ppo = LossControlledPPO(
            DummyActorCritic(),
            num_learning_epochs=1,
            num_mini_batches=1,
            learning_rate=1e-2,
            optimizer_class_name="AdamW",
            critic_warmup_iterations=100,
            freeze_actor_encoder_iterations=200,
        )
        ppo.storage = OneBatchStorage()
        ppo.start_critic_warmup(11300)
        encoder = ppo.actor_critic.encoders[0].weight
        initial = encoder.detach().clone()

        ppo.update(11400)
        self.assertTrue(torch.equal(initial, encoder.detach()))
        self.assertFalse(encoder.requires_grad)

        ppo.update(11500)
        self.assertTrue(encoder.requires_grad)
        self.assertFalse(torch.equal(initial, encoder.detach()))

    def test_actor_updates_at_the_exact_warmup_boundary(self):
        ppo = self.make_ppo()
        ppo.start_critic_warmup(11700)
        ppo.update(11700)
        before = self.clone_parameters(ppo)

        _, stats = ppo.update(11800)
        after = self.clone_parameters(ppo)

        actor_names = [
            name for name in before if not name.startswith(CRITIC_PREFIXES)
        ]
        self.assertTrue(
            any(not torch.equal(before[name], after[name]) for name in actor_names)
        )
        self.assertEqual(stats["actor_update_enabled"].item(), 1.0)
        self.assertEqual(stats["critic_warmup_remaining"].item(), 0.0)

    def test_reward_order_warning_requires_two_valid_bad_windows(self):
        ppo = LossControlledPPO(
            DummyActorCritic(), quality_min_episodes=1
        )
        window = dict(
            success_rate=0.9,
            box_pass_rate=0.95,
            fall_rate=0.05,
            episode_count=256,
            reward_order_valid=True,
            reward_order_ok=False,
        )
        ppo.update_quality_curriculum(**window)
        self.assertFalse(ppo.reward_order_warning)
        ppo.update_quality_curriculum(**window)
        self.assertTrue(ppo.reward_order_warning)

    def test_checkpoint_restores_warmup_without_restarting_it(self):
        source = self.make_ppo()
        source.start_critic_warmup(11700)
        source.update(11700)
        checkpoint = source.state_dict()

        restored = self.make_ppo()
        restored.load_state_dict(checkpoint)

        self.assertEqual(restored.critic_warmup_until_iteration, 11800)
        self.assertTrue(restored.is_critic_warmup_active(11799))
        self.assertFalse(restored.is_critic_warmup_active(11800))

    def test_speed_curriculum_obeys_dwell_regresses_and_warns(self):
        ppo = self.make_ppo()
        ppo.reference_kl_min_coef = 0.02
        ppo.reference_kl_max_coef = 1.0
        ppo.set_reference_kl_coef(0.2)
        stable = dict(
            success_rate=0.95,
            box_pass_rate=0.96,
            fall_rate=0.02,
            episode_count=256,
            flat_speed_mean=2.0,
            flat_severe_overspeed_ratio=0.7,
            mean_action_rate=5.0,
            action_saturation_ratio=0.6,
            dof_near_limit_ratio=0.2,
        )
        ppo.current_learning_iteration = 199
        for _ in range(3):
            ppo.update_quality_curriculum(**stable)
        self.assertAlmostEqual(ppo.speed_penalty_level, 0.1)

        ppo.current_learning_iteration = 200
        ppo.update_quality_curriculum(**stable)
        self.assertAlmostEqual(ppo.speed_penalty_level, 0.2)
        self.assertTrue(ppo.curriculum_promoted)
        self.assertLess(ppo.reference_kl_coef, 0.2)

        regressed = dict(stable, success_rate=0.5, fall_rate=0.6)
        ppo.update_quality_curriculum(**regressed)
        self.assertFalse(ppo.collapse_warning)
        ppo.update_quality_curriculum(**regressed)
        self.assertAlmostEqual(ppo.speed_penalty_level, 0.1)
        self.assertTrue(ppo.curriculum_regressed)
        self.assertTrue(ppo.collapse_warning)
        self.assertEqual(ppo.reference_kl_coef, 1.0)

    def test_curriculum_levels_snap_floating_point_endpoint_noise(self):
        ppo = self.make_ppo()

        ppo.set_quality_levels(0.9999999999999999, 1e-12)

        self.assertEqual(ppo.speed_penalty_level, 1.0)
        self.assertEqual(ppo.motion_quality_level, 0.0)

        ppo.current_learning_iteration = 1000
        ppo.update_quality_curriculum(
            success_rate=0.95,
            box_pass_rate=0.96,
            fall_rate=0.02,
            episode_count=256,
            flat_speed_mean=2.0,
            flat_severe_overspeed_ratio=0.7,
        )

        self.assertEqual(ppo.speed_penalty_level, 1.0)
        self.assertFalse(ppo.curriculum_promoted)

    def test_speed_mastery_transitions_to_motion_phase(self):
        ppo = self.make_ppo()
        ppo.speed_penalty_level = 1.0
        mastered = dict(
            success_rate=0.95,
            box_pass_rate=0.96,
            fall_rate=0.02,
            episode_count=256,
            flat_speed_mean=0.55,
            flat_severe_overspeed_ratio=0.02,
            mean_action_rate=5.0,
            action_saturation_ratio=0.6,
            dof_near_limit_ratio=0.2,
        )
        for _ in range(ppo.speed_master_windows_required):
            ppo.update_quality_curriculum(**mastered)
        self.assertEqual(ppo.quality_phase, 1)
        self.assertAlmostEqual(ppo.speed_penalty_level, 1.0)
        self.assertAlmostEqual(ppo.motion_quality_level, 0.1)
        self.assertTrue(ppo.curriculum_phase_transition)

    def test_reference_kl_has_actor_gradients_and_reference_is_frozen(self):
        actor_critic = ActorCritic(
            num_actor_obs=2,
            num_critic_obs=2,
            num_actions=1,
            actor_hidden_dims=[4],
            critic_hidden_dims=[4],
            init_noise_std=0.5,
        )
        ppo = PPO(actor_critic, reference_kl_initial_coef=1.0)
        ppo.snapshot_reference_policy()
        with torch.no_grad():
            ppo.actor_critic.actor[0].weight.add_(0.1)

        obs = torch.tensor([[0.2, -0.1], [0.4, 0.3]])
        ppo.actor_critic.act(obs)
        actions = ppo.actor_critic.action_mean.detach().clone()
        old_log_prob = ppo.actor_critic.get_actions_log_prob(actions).detach()
        minibatch = SimpleNamespace(
            obs=obs,
            critic_obs=obs,
            actions=actions,
            values=torch.zeros(2, 1),
            advantages=torch.ones(2, 1),
            returns=torch.zeros(2, 1),
            old_actions_log_prob=old_log_prob,
            old_mu=actions,
            old_sigma=torch.full_like(actions, 0.5),
            reference_mu=ppo.reference_actor_critic.act_inference(obs).detach(),
            reference_sigma=torch.full_like(actions, 0.5),
            hidden_states=SimpleNamespace(actor=None, critic=None),
            masks=None,
        )
        ppo._critic_warmup_active = False
        losses, _, stats = ppo.compute_losses(minibatch)
        self.assertGreater(stats["reference_kl"].item(), 0.0)
        self.assertGreater(losses["reference_kl_loss"].item(), 0.0)
        losses["reference_kl_loss"].backward()
        actor_gradients = [
            parameter.grad
            for name, parameter in ppo.actor_critic.named_parameters()
            if name.startswith("actor.")
        ]
        self.assertTrue(any(gradient is not None for gradient in actor_gradients))
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in ppo.reference_actor_critic.parameters()
            )
        )

    def test_reference_kl_retains_a_nonzero_minimum(self):
        ppo = PPO(
            ActorCritic(2, 2, 1, actor_hidden_dims=[4], critic_hidden_dims=[4]),
            reference_kl_min_coef=0.05,
            reference_kl_max_coef=1.0,
            reference_kl_start_coef=0.2,
        )
        self.assertAlmostEqual(ppo.reference_kl_coef, 0.2)
        ppo.set_reference_kl_coef(0.0)
        self.assertAlmostEqual(ppo.reference_kl_coef, 0.05)
        ppo.set_reference_kl_coef(2.0)
        self.assertAlmostEqual(ppo.reference_kl_coef, 1.0)

    def test_reference_recurrent_state_is_independent_and_resets_with_done(self):
        actor_critic = ActorCriticRecurrent(
            num_actor_obs=2,
            num_critic_obs=2,
            num_actions=1,
            actor_hidden_dims=[4],
            critic_hidden_dims=[4],
            rnn_type="gru",
            rnn_hidden_size=4,
        )
        ppo = PPO(actor_critic, reference_kl_max_coef=1.0)
        ppo.snapshot_reference_policy()
        ppo.init_storage(2, 1, [2], [2], [1])
        obs = torch.tensor([[0.2, -0.1], [0.4, 0.3]])
        ppo.act(obs, obs)

        current_hidden = ppo.actor_critic.memory_a.hidden_states
        reference_hidden = ppo.reference_actor_critic.memory_a.hidden_states
        self.assertEqual(ppo.rollout_actor_output_max_diff.item(), 0.0)
        self.assertEqual(ppo.rollout_actor_std_max_diff.item(), 0.0)
        self.assertIsNot(current_hidden, reference_hidden)
        self.assertNotEqual(
            current_hidden.data_ptr(), reference_hidden.data_ptr()
        )
        torch.testing.assert_close(current_hidden, reference_hidden)

        dones = torch.tensor([True, False])
        ppo.process_env_step(
            torch.zeros(2), dones, {}, obs, obs
        )
        self.assertTrue(
            (ppo.actor_critic.memory_a.hidden_states[:, 0] == 0.0).all()
        )
        self.assertTrue(
            (
                ppo.reference_actor_critic.memory_a.hidden_states[:, 0]
                == 0.0
            ).all()
        )
        ppo.actor_critic.reset()
        ppo.reference_actor_critic.reset()
        self.assertIsNone(ppo.actor_critic.memory_a.hidden_states)
        self.assertIsNone(
            ppo.reference_actor_critic.memory_a.hidden_states
        )

    def test_inference_rollout_statistics_are_reset_out_of_place(self):
        actor_critic = ActorCriticRecurrent(
            num_actor_obs=2,
            num_critic_obs=2,
            num_actions=1,
            actor_hidden_dims=[4],
            critic_hidden_dims=[4],
            rnn_type="gru",
            rnn_hidden_size=4,
        )
        ppo = PPO(actor_critic, reference_kl_max_coef=1.0)
        ppo.snapshot_reference_policy()
        ppo.init_storage(2, 1, [2], [2], [1])
        observations = torch.tensor([[0.2, -0.1], [0.4, 0.3]])
        with torch.inference_mode():
            ppo.act(observations, observations)

        ppo._reset_rollout_actor_equivalence_statistics()

        self.assertEqual(ppo.rollout_actor_output_max_diff.item(), 0.0)
        self.assertEqual(ppo.rollout_actor_std_max_diff.item(), 0.0)

    def test_incomplete_does_not_bootstrap_but_external_timeout_does(self):
        actor_critic = ActorCritic(
            num_actor_obs=2,
            num_critic_obs=2,
            num_actions=1,
            actor_hidden_dims=[4],
            critic_hidden_dims=[4],
        )
        ppo = PPO(actor_critic, gamma=0.9)
        ppo.init_storage(2, 1, [2], [2], [1])
        observations = torch.zeros(2, 2)
        ppo.act(observations, observations)
        predicted_values = ppo.transition.values.squeeze(1).clone()
        rewards = torch.tensor([1.0, 1.0])
        dones = torch.tensor([True, True])

        ppo.process_env_step(
            rewards,
            dones,
            {
                "episode_incompletes": torch.tensor([True, False]),
                "time_outs": torch.tensor([False, True]),
            },
            observations,
            observations,
        )

        self.assertEqual(ppo.storage.rewards[0, 0].item(), 1.0)
        self.assertAlmostEqual(
            ppo.storage.rewards[0, 1].item(),
            (1.0 + 0.9 * predicted_values[1]).item(),
            places=6,
        )

    def test_reference_and_quality_state_survive_checkpoint_restore(self):
        source = self.make_ppo()
        source.snapshot_reference_policy()
        source.set_quality_levels(0.35, 0.15)
        source.quality_phase = 1
        source.curriculum_stable_windows = 1
        source.quality_stage_start_iteration = 12345
        source.reference_kl_min_coef = 0.02
        source.reference_kl_max_coef = 1.0
        source.set_reference_kl_coef(0.3)
        source.collapse_windows = 1
        checkpoint = source.state_dict()

        restored = self.make_ppo()
        restored.load_state_dict(checkpoint)
        self.assertAlmostEqual(restored.speed_penalty_level, 0.35)
        self.assertAlmostEqual(restored.motion_quality_level, 0.15)
        self.assertEqual(restored.quality_phase, 1)
        self.assertEqual(restored.curriculum_stable_windows, 1)
        self.assertEqual(restored.quality_stage_start_iteration, 12345)
        self.assertAlmostEqual(restored.reference_kl_coef, 0.3)
        self.assertEqual(restored.collapse_windows, 1)
        for name, value in source._reference_actor_state_dict().items():
            self.assertTrue(
                torch.equal(
                    value,
                    restored._reference_actor_state_dict()[name],
                ),
                name,
            )

    def test_required_v11_state_rejects_ordinary_legacy_resume(self):
        source = self.make_ppo().state_dict()
        del source["algorithm_state_dict"]["curriculum_state_version"]
        target = LossControlledPPO(
            DummyActorCritic(), require_v11_curriculum_state=True
        )
        with self.assertRaises(RuntimeError):
            target.load_state_dict(source)

    def test_episode_info_aggregation_uses_true_result_counts(self):
        runner = OnPolicyRunner.__new__(OnPolicyRunner)
        runner.device = "cpu"
        infos = [
            {
                "num_terminated": torch.tensor(40.0),
                "success_rate": torch.tensor(1.0),
                "raw/success_mean_return": torch.tensor(10.0),
                "raw/success_episode_count": torch.tensor(40.0),
                "raw/fall_failure_mean_return": torch.tensor(0.0),
                "raw/fall_failure_episode_count": torch.tensor(0.0),
                "raw/landing_overrun_mean_return": torch.tensor(-5.0),
                "raw/landing_overrun_episode_count": torch.tensor(20.0),
                "raw/landing_lateral_exit_mean_return": torch.tensor(-10.0),
                "raw/landing_lateral_exit_episode_count": torch.tensor(20.0),
                "raw/landing_timeout_mean_return": torch.tensor(5.0),
                "raw/landing_timeout_episode_count": torch.tensor(40.0),
                "raw/late_failure_mean_return": torch.tensor(-9.0),
                "raw/late_failure_episode_count": torch.tensor(40.0),
                "raw/early_failure_mean_return": torch.tensor(-30.0),
                "raw/early_failure_episode_count": torch.tensor(20.0),
                "layout_1_success_rate": torch.tensor(1.0),
                "layout_1_episode_count": torch.tensor(40.0),
            },
            {
                "num_terminated": torch.tensor(60.0),
                "success_rate": torch.tensor(1.0 / 3.0),
                "raw/success_mean_return": torch.tensor(6.0),
                "raw/success_episode_count": torch.tensor(20.0),
                "raw/fall_failure_mean_return": torch.tensor(-30.0),
                "raw/fall_failure_episode_count": torch.tensor(40.0),
                "raw/landing_overrun_mean_return": torch.tensor(-7.0),
                "raw/landing_overrun_episode_count": torch.tensor(20.0),
                "raw/landing_lateral_exit_mean_return": torch.tensor(-10.0),
                "raw/landing_lateral_exit_episode_count": torch.tensor(20.0),
                "raw/landing_timeout_mean_return": torch.tensor(5.0),
                "raw/landing_timeout_episode_count": torch.tensor(40.0),
                "raw/late_failure_mean_return": torch.tensor(-9.0),
                "raw/late_failure_episode_count": torch.tensor(40.0),
                "raw/early_failure_mean_return": torch.tensor(-35.0),
                "raw/early_failure_episode_count": torch.tensor(20.0),
                "layout_1_success_rate": torch.tensor(1.0 / 3.0),
                "layout_1_episode_count": torch.tensor(60.0),
            },
        ]

        summary = runner._aggregate_episode_infos(infos)

        self.assertEqual(summary["num_terminated"].item(), 100.0)
        self.assertAlmostEqual(summary["success_rate"].item(), 0.6)
        self.assertAlmostEqual(
            summary["raw/success_mean_return"].item(),
            (40.0 * 10.0 + 20.0 * 6.0) / 60.0,
            places=6,
        )
        self.assertEqual(
            summary["raw/fall_failure_mean_return"].item(), -30.0
        )
        self.assertAlmostEqual(
            summary["layout_1_success_rate"].item(),
            0.6,
            places=6,
        )
        self.assertAlmostEqual(
            summary["raw/success_minus_best_failure_return"].item(),
            (40.0 * 10.0 + 20.0 * 6.0) / 60.0 - 5.0,
            places=5,
        )
        self.assertEqual(summary["reward_order_valid"].item(), 1.0)
        self.assertEqual(summary["reward_order_ok"].item(), 1.0)

    def test_runner_saves_unique_warmup_boundary_checkpoint(self):
        runner = OnPolicyRunner.__new__(OnPolicyRunner)
        runner.alg = self.make_ppo()
        runner.alg.start_critic_warmup(11700)
        runner.alg.snapshot_reference_policy()
        runner.current_learning_iteration = 11800
        with tempfile.TemporaryDirectory() as temp_dir:
            runner.log_dir = temp_dir
            runner._save_warmup_boundary_checkpoint()
            checkpoint_path = (
                Path(temp_dir) / "model_11800_warmup.pt"
            )
            self.assertTrue(checkpoint_path.is_file())
            self.assertTrue((Path(temp_dir) / "model_11800.pt").is_file())
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertEqual(checkpoint["iter"], 11800)

    def test_runner_starts_warmup_only_for_explicit_critic_reset(self):
        source = self.make_ppo().state_dict()
        source.update(iter=11700, infos={"source": "v5"})

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.pt"
            torch.save(source, source_path)

            runner = OnPolicyRunner.__new__(OnPolicyRunner)
            runner.device = "cpu"
            runner.cfg = {
                "ckpt_manipulator": "reset_critic_and_optimizer"
            }
            runner.alg = self.make_ppo()
            runner.log_dir = temp_dir
            runner.load(str(source_path))

            self.assertEqual(
                runner.alg.critic_warmup_until_iteration, 11800
            )
            self.assertIsNone(runner.alg.reference_actor_critic)
            manipulated_path = Path(temp_dir) / "model_11700.pt"
            self.assertTrue(manipulated_path.is_file())

            resumed = OnPolicyRunner.__new__(OnPolicyRunner)
            resumed.device = "cpu"
            resumed.cfg = {"ckpt_manipulator": None}
            resumed.alg = self.make_ppo()
            resumed.log_dir = temp_dir
            resumed.load(str(manipulated_path))

            self.assertEqual(
                resumed.alg.critic_warmup_until_iteration, 11800
            )

    def test_negative_warmup_length_is_rejected(self):
        with self.assertRaises(ValueError):
            LossControlledPPO(
                DummyActorCritic(), critic_warmup_iterations=-1
            )

    def test_negative_actor_equivalence_tolerance_is_rejected(self):
        with self.assertRaises(ValueError):
            LossControlledPPO(
                DummyActorCritic(),
                actor_output_equivalence_tolerance=-1e-4,
            )


if __name__ == "__main__":
    unittest.main()
