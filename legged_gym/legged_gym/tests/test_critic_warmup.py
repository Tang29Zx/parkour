"""Tests for resumable Critic-only PPO warmup."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
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


if __name__ == "__main__":
    unittest.main()
