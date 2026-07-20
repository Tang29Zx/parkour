# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import copy
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 clip_min_std= 1e-15, # clip the policy.std if it supports, check update()
                 optimizer_class_name= "Adam",
                 schedule="fixed",
                 desired_kl=0.01,
                 critic_warmup_iterations=0,
                 actor_finetune_learning_rate=None,
                 actor_finetune_clip_param=None,
                 actor_finetune_entropy_coef=None,
                 reference_kl_initial_coef=0.0,
                 quality_min_episodes=256,
                 quality_required_windows=2,
                 quality_increase=0.05,
                 quality_decrease=0.10,
                 quality_success_up=0.80,
                 quality_box_pass_up=0.90,
                 quality_fall_up=0.20,
                 quality_success_down=0.60,
                 quality_box_pass_down=0.70,
                 quality_fall_down=0.35,
                 collapse_success_threshold=0.60,
                 collapse_fall_threshold=0.50,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.warmup_learning_rate = float(learning_rate)

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = getattr(optim, optimizer_class_name)(self.actor_critic.parameters(), lr=learning_rate)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_min_std = torch.tensor(clip_min_std, device= self.device) if isinstance(clip_min_std, (tuple, list)) else clip_min_std
        self.critic_warmup_iterations = int(critic_warmup_iterations)
        if self.critic_warmup_iterations < 0:
            raise ValueError("critic_warmup_iterations must be non-negative.")
        self.critic_warmup_until_iteration = None
        self.actor_finetune_learning_rate = float(
            learning_rate
            if actor_finetune_learning_rate is None
            else actor_finetune_learning_rate
        )
        self.actor_finetune_clip_param = float(
            clip_param
            if actor_finetune_clip_param is None
            else actor_finetune_clip_param
        )
        self.actor_finetune_entropy_coef = float(
            entropy_coef
            if actor_finetune_entropy_coef is None
            else actor_finetune_entropy_coef
        )
        self.actor_finetune_active = False
        self.reference_kl_initial_coef = float(reference_kl_initial_coef)
        if self.reference_kl_initial_coef < 0.0:
            raise ValueError("reference_kl_initial_coef must be non-negative.")
        self.reference_actor_critic = None
        self.quality_level = 0.0
        self.quality_up_windows = 0
        self.quality_down_windows = 0
        self.collapse_windows = 0
        self.collapse_warning = False
        self.quality_min_episodes = int(quality_min_episodes)
        self.quality_required_windows = int(quality_required_windows)
        self.quality_increase = float(quality_increase)
        self.quality_decrease = float(quality_decrease)
        self.quality_success_up = float(quality_success_up)
        self.quality_box_pass_up = float(quality_box_pass_up)
        self.quality_fall_up = float(quality_fall_up)
        self.quality_success_down = float(quality_success_down)
        self.quality_box_pass_down = float(quality_box_pass_down)
        self.quality_fall_down = float(quality_fall_down)
        self.collapse_success_threshold = float(collapse_success_threshold)
        self.collapse_fall_threshold = float(collapse_fall_threshold)
        if self.quality_min_episodes < 1:
            raise ValueError("quality_min_episodes must be positive.")
        if self.quality_required_windows < 1:
            raise ValueError("quality_required_windows must be positive.")
        
        # algorithm status
        self.current_learning_iteration = 0

    def start_critic_warmup(self, start_iteration):
        """Train only the Critic for a fixed number of loaded iterations."""
        if self.critic_warmup_iterations == 0:
            self.critic_warmup_until_iteration = None
            return
        self.critic_warmup_until_iteration = (
            int(start_iteration) + self.critic_warmup_iterations
        )

    @staticmethod
    def _is_actor_side_parameter(name):
        critic_prefixes = ("critic.", "memory_c.", "critic_encoders.")
        return not name.startswith(critic_prefixes)

    def snapshot_reference_policy(self):
        """Freeze the currently loaded Actor side as the behavior reference."""
        self.reference_actor_critic = copy.deepcopy(self.actor_critic).to(
            self.device
        )
        self.reference_actor_critic.eval()
        for parameter in self.reference_actor_critic.parameters():
            parameter.requires_grad_(False)

    def _reference_actor_state_dict(self):
        if self.reference_actor_critic is None:
            return None
        return OrderedDict(
            (name, value.detach().clone())
            for name, value in self.reference_actor_critic.state_dict().items()
            if self._is_actor_side_parameter(name)
        )

    def _restore_reference_policy(self, reference_state_dict):
        if reference_state_dict is None:
            self.reference_actor_critic = None
            return
        expected = {
            name
            for name in self.actor_critic.state_dict()
            if self._is_actor_side_parameter(name)
        }
        actual = set(reference_state_dict)
        if actual != expected:
            raise KeyError(
                "Reference Actor keys do not match the current policy. "
                f"Missing: {sorted(expected - actual)}; "
                f"unexpected: {sorted(actual - expected)}."
            )
        self.reference_actor_critic = copy.deepcopy(self.actor_critic).to(
            self.device
        )
        target_state = self.reference_actor_critic.state_dict()
        for name, value in reference_state_dict.items():
            if target_state[name].shape != value.shape:
                raise ValueError(
                    f"Reference parameter {name!r} has shape "
                    f"{tuple(value.shape)}, expected {tuple(target_state[name].shape)}."
                )
            target_state[name] = value.to(
                device=target_state[name].device,
                dtype=target_state[name].dtype,
            )
        self.reference_actor_critic.load_state_dict(target_state)
        self.reference_actor_critic.eval()
        for parameter in self.reference_actor_critic.parameters():
            parameter.requires_grad_(False)

    @property
    def reference_kl_coef(self):
        return self.reference_kl_initial_coef * (1.0 - self.quality_level)

    def set_quality_level(self, level):
        """Clamp and store the global action-quality curriculum level."""
        self.quality_level = min(max(float(level), 0.0), 1.0)

    def activate_actor_finetune(self):
        """Switch from Critic warmup to conservative joint PPO updates."""
        if self.actor_finetune_active:
            return
        self.actor_finetune_active = True
        self.learning_rate = self.actor_finetune_learning_rate
        self.clip_param = self.actor_finetune_clip_param
        self.entropy_coef = self.actor_finetune_entropy_coef
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def update_quality_curriculum(
        self, success_rate, box_pass_rate, fall_rate, episode_count
    ):
        """Update quality difficulty from one aggregated logging window."""
        self.collapse_warning = False
        if self.is_critic_warmup_active() or episode_count < self.quality_min_episodes:
            return

        promote = (
            success_rate >= self.quality_success_up
            and box_pass_rate >= self.quality_box_pass_up
            and fall_rate <= self.quality_fall_up
        )
        regress = (
            success_rate < self.quality_success_down
            or box_pass_rate < self.quality_box_pass_down
            or fall_rate > self.quality_fall_down
        )
        collapse = (
            success_rate < self.collapse_success_threshold
            or fall_rate > self.collapse_fall_threshold
        )

        self.quality_up_windows = self.quality_up_windows + 1 if promote else 0
        self.quality_down_windows = self.quality_down_windows + 1 if regress else 0
        self.collapse_windows = self.collapse_windows + 1 if collapse else 0

        if self.quality_up_windows >= self.quality_required_windows:
            self.set_quality_level(self.quality_level + self.quality_increase)
            self.quality_up_windows = 0
            self.quality_down_windows = 0
        elif self.quality_down_windows >= self.quality_required_windows:
            self.set_quality_level(self.quality_level - self.quality_decrease)
            self.quality_down_windows = 0
            self.quality_up_windows = 0

        if self.collapse_windows >= self.quality_required_windows:
            self.collapse_warning = True
            self.collapse_windows = 0

    def is_critic_warmup_active(self, iteration=None):
        """Return whether Actor-side losses must remain frozen."""
        if self.critic_warmup_until_iteration is None:
            return False
        if iteration is None:
            iteration = self.current_learning_iteration
        return int(iteration) < self.critic_warmup_until_iteration

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.transition = RolloutStorage.Transition()
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, current_learning_iteration):
        self.current_learning_iteration = current_learning_iteration
        critic_warmup_active = self.is_critic_warmup_active(
            current_learning_iteration
        )
        if not critic_warmup_active:
            self.activate_actor_finetune()
            if (
                self.reference_kl_coef > 0.0
                and self.reference_actor_critic is None
            ):
                raise RuntimeError(
                    "Protected Actor fine-tuning requires a saved reference "
                    "policy. Start from the original checkpoint with "
                    "reset_critic_and_optimizer, or resume a v10 checkpoint."
                )
        self._critic_warmup_active = critic_warmup_active
        mean_losses = defaultdict(lambda :0.)
        average_stats = defaultdict(lambda :0.)
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for minibatch in generator:

                losses, _, stats = self.compute_losses(minibatch)

                loss = 0.
                for k, v in losses.items():
                    if not critic_warmup_active or k == "value_loss":
                        loss += getattr(self, k + "_coef", 1.) * v
                    mean_losses[k] = mean_losses[k] + v.detach()
                mean_losses["total_loss"] = mean_losses["total_loss"] + loss.detach()
                for k, v in stats.items():
                    average_stats[k] = average_stats[k] + v.detach()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for k in mean_losses.keys():
            mean_losses[k] = mean_losses[k] / num_updates
        for k in average_stats.keys():
            average_stats[k] = average_stats[k] / num_updates
        warmup_remaining = 0
        if self.critic_warmup_until_iteration is not None:
            warmup_remaining = max(
                self.critic_warmup_until_iteration
                - int(current_learning_iteration),
                0,
            )
        average_stats["actor_update_enabled"] = torch.tensor(
            0.0 if critic_warmup_active else 1.0, device=self.device
        )
        average_stats["critic_warmup_remaining"] = torch.tensor(
            float(warmup_remaining), device=self.device
        )
        average_stats["quality_level"] = torch.tensor(
            self.quality_level, device=self.device
        )
        average_stats["reference_kl_coef"] = torch.tensor(
            self.reference_kl_coef, device=self.device
        )
        average_stats["collapse_warning"] = torch.tensor(
            float(self.collapse_warning), device=self.device
        )
        self.storage.clear()
        if hasattr(self.actor_critic, "clip_std"):
            self.actor_critic.clip_std(min= self.clip_min_std)

        return mean_losses, average_stats

    def compute_losses(self, minibatch):
        self.actor_critic.act(minibatch.obs, masks=minibatch.masks, hidden_states=minibatch.hidden_states.actor)
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(minibatch.actions)
        value_batch = self.actor_critic.evaluate(minibatch.critic_obs, masks=minibatch.masks, hidden_states=minibatch.hidden_states.critic)
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        try:
            entropy_batch = self.actor_critic.entropy
        except:
            entropy_batch = None

        # KL
        if self.desired_kl != None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                            torch.log(sigma_batch / minibatch.old_sigma + 1.e-5) + (torch.square(minibatch.old_sigma) + torch.square(minibatch.old_mu - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate


        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(minibatch.old_actions_log_prob))
        surrogate = -torch.squeeze(minibatch.advantages) * ratio
        surrogate_clipped = -torch.squeeze(minibatch.advantages) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = minibatch.values + (value_batch - minibatch.values).clamp(-self.clip_param,
                                                                                                    self.clip_param)
            value_losses = (value_batch - minibatch.returns).pow(2)
            value_losses_clipped = (value_clipped - minibatch.returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (minibatch.returns - value_batch).pow(2).mean()
        
        return_ = dict(
            surrogate_loss= surrogate_loss,
            value_loss= value_loss,
        )
        if entropy_batch is not None:
            return_["entropy"] = - entropy_batch.mean()

        if (
            not getattr(self, "_critic_warmup_active", False)
            and self.reference_kl_coef > 0.0
        ):
            with torch.no_grad():
                self.reference_actor_critic.act(
                    minibatch.obs,
                    masks=minibatch.masks,
                    hidden_states=minibatch.hidden_states.actor,
                )
                reference_mu = self.reference_actor_critic.action_mean.detach()
                reference_sigma = (
                    self.reference_actor_critic.action_std.detach()
                )
            current_sigma = sigma_batch.clamp_min(1e-6)
            reference_sigma = reference_sigma.clamp_min(1e-6)
            reference_kl = torch.sum(
                torch.log(reference_sigma / current_sigma)
                + (
                    torch.square(current_sigma)
                    + torch.square(mu_batch - reference_mu)
                )
                / (2.0 * torch.square(reference_sigma))
                - 0.5,
                dim=-1,
            ).mean()
            return_["reference_kl"] = reference_kl
        
        inter_vars = dict(
            ratio= ratio,
            surrogate= surrogate,
            surrogate_clipped= surrogate_clipped,
        )
        if self.desired_kl != None and self.schedule == 'adaptive':
            inter_vars["kl"] = kl
        if self.use_clipped_value_loss:
            inter_vars["value_clipped"] = value_clipped
        return return_, inter_vars, dict()

    def state_dict(self):
        state_dict = {
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "reference_model_state_dict": self._reference_actor_state_dict(),
            "algorithm_state_dict": {
                "critic_warmup_until_iteration": (
                    self.critic_warmup_until_iteration
                ),
                "quality_level": self.quality_level,
                "quality_up_windows": self.quality_up_windows,
                "quality_down_windows": self.quality_down_windows,
                "collapse_windows": self.collapse_windows,
                "collapse_warning": self.collapse_warning,
                "actor_finetune_active": self.actor_finetune_active,
                "learning_rate": self.learning_rate,
                "clip_param": self.clip_param,
                "entropy_coef": self.entropy_coef,
            },
        }
        if hasattr(self, "lr_scheduler"):
            state_dict["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.actor_critic.load_state_dict(state_dict["model_state_dict"])
        if "optimizer_state_dict" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler_state_dict"])
        elif "lr_scheduler_state_dict" in state_dict:
            print("Warning: lr scheduler state dict loaded but no lr scheduler is initialized. Ignored.")
        algorithm_state = state_dict.get("algorithm_state_dict", {})
        self.critic_warmup_until_iteration = algorithm_state.get(
            "critic_warmup_until_iteration"
        )
        self.set_quality_level(algorithm_state.get("quality_level", 0.0))
        self.quality_up_windows = int(
            algorithm_state.get("quality_up_windows", 0)
        )
        self.quality_down_windows = int(
            algorithm_state.get("quality_down_windows", 0)
        )
        self.collapse_windows = int(
            algorithm_state.get("collapse_windows", 0)
        )
        self.collapse_warning = bool(
            algorithm_state.get("collapse_warning", False)
        )
        self.actor_finetune_active = bool(
            algorithm_state.get("actor_finetune_active", False)
        )
        self.learning_rate = float(
            algorithm_state.get("learning_rate", self.learning_rate)
        )
        self.clip_param = float(
            algorithm_state.get("clip_param", self.clip_param)
        )
        self.entropy_coef = float(
            algorithm_state.get("entropy_coef", self.entropy_coef)
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate
        self._restore_reference_policy(
            state_dict.get("reference_model_state_dict")
        )
