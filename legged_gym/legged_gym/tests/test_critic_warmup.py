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

    def test_quality_curriculum_promotes_regresses_and_warns(self):
        ppo = self.make_ppo()
        ppo.update_quality_curriculum(0.9, 0.95, 0.1, 256)
        self.assertEqual(ppo.quality_level, 0.0)
        ppo.update_quality_curriculum(0.9, 0.95, 0.1, 256)
        self.assertAlmostEqual(ppo.quality_level, 0.05)

        ppo.update_quality_curriculum(0.5, 0.6, 0.6, 255)
        self.assertAlmostEqual(ppo.quality_level, 0.05)
        self.assertFalse(ppo.collapse_warning)
        ppo.update_quality_curriculum(0.5, 0.6, 0.6, 256)
        ppo.update_quality_curriculum(0.5, 0.6, 0.6, 256)
        self.assertEqual(ppo.quality_level, 0.0)
        self.assertTrue(ppo.collapse_warning)

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
        )
        ppo.set_quality_level(0.0)
        self.assertAlmostEqual(ppo.reference_kl_coef, 1.0)
        ppo.set_quality_level(1.0)
        self.assertAlmostEqual(ppo.reference_kl_coef, 0.05)

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
        source.set_quality_level(0.35)
        source.quality_up_windows = 1
        source.collapse_windows = 1
        checkpoint = source.state_dict()

        restored = self.make_ppo()
        restored.load_state_dict(checkpoint)
        self.assertAlmostEqual(restored.quality_level, 0.35)
        self.assertEqual(restored.quality_up_windows, 1)
        self.assertEqual(restored.collapse_windows, 1)
        for name, value in source._reference_actor_state_dict().items():
            self.assertTrue(
                torch.equal(
                    value,
                    restored._reference_actor_state_dict()[name],
                ),
                name,
            )

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
