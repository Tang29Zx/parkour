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
                 reference_kl_initial_coef=None,
                 reference_kl_min_coef=0.0,
                 reference_kl_max_coef=0.0,
                 actor_parameter_equivalence_tolerance=0.0,
                 actor_output_equivalence_tolerance=1e-4,
                 actor_std_equivalence_tolerance=1e-7,
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
                 require_v11_curriculum_state=False,
                 speed_penalty_initial_level=0.10,
                 motion_quality_initial_level=0.0,
                 curriculum_level_step=0.10,
                 curriculum_stage_min_iterations=200,
                 curriculum_stable_windows=3,
                 curriculum_regression_windows=2,
                 speed_success_up=0.90,
                 speed_box_pass_up=0.92,
                 speed_fall_up=0.10,
                 curriculum_success_down=0.85,
                 curriculum_box_pass_down=0.88,
                 curriculum_fall_down=0.15,
                 speed_master_windows=5,
                 speed_master_flat_min=0.45,
                 speed_master_flat_max=0.70,
                 speed_master_severe_overspeed=0.05,
                 motion_success_up=0.88,
                 motion_box_pass_up=0.90,
                 motion_fall_up=0.12,
                 motion_flat_speed_max=0.75,
                 motion_severe_overspeed_max=0.10,
                 motion_master_windows=5,
                 motion_master_action_rate=2.0,
                 motion_master_action_saturation=0.20,
                 motion_master_dof_near_limit=0.10,
                 reference_kl_start_coef=0.20,
                 reference_kl_stable_windows=2,
                 reference_kl_stable_decrease=0.02,
                 reference_kl_regression_increase=0.10,
                 reference_kl_regression_floor=0.50,
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
        self.optimizer_class = getattr(optim, optimizer_class_name)
        self.optimizer = self.optimizer_class(
            self.actor_critic.parameters(), lr=learning_rate
        )

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
        if reference_kl_initial_coef is not None:
            # Backward-compatible interpretation for older task configs.
            reference_kl_max_coef = float(reference_kl_initial_coef)
        self.reference_kl_min_coef = float(reference_kl_min_coef)
        self.reference_kl_max_coef = float(reference_kl_max_coef)
        if (
            self.reference_kl_min_coef < 0.0
            or self.reference_kl_max_coef < self.reference_kl_min_coef
        ):
            raise ValueError(
                "Reference KL coefficients must satisfy 0 <= min <= max."
            )
        self.actor_parameter_equivalence_tolerance = float(
            actor_parameter_equivalence_tolerance
        )
        self.actor_output_equivalence_tolerance = float(
            actor_output_equivalence_tolerance
        )
        self.actor_std_equivalence_tolerance = float(
            actor_std_equivalence_tolerance
        )
        if min(
            self.actor_parameter_equivalence_tolerance,
            self.actor_output_equivalence_tolerance,
            self.actor_std_equivalence_tolerance,
        ) < 0.0:
            raise ValueError("Actor equivalence tolerances must be non-negative.")
        self.reference_actor_critic = None
        self._reset_rollout_actor_equivalence_statistics()
        self.require_v11_curriculum_state = bool(
            require_v11_curriculum_state
        )
        self.speed_penalty_initial_level = float(
            speed_penalty_initial_level
        )
        self.motion_quality_initial_level = float(
            motion_quality_initial_level
        )
        self.curriculum_level_step = float(curriculum_level_step)
        self.curriculum_stage_min_iterations = int(
            curriculum_stage_min_iterations
        )
        self.curriculum_stable_windows_required = int(
            curriculum_stable_windows
        )
        self.curriculum_regression_windows_required = int(
            curriculum_regression_windows
        )
        self.speed_success_up = float(speed_success_up)
        self.speed_box_pass_up = float(speed_box_pass_up)
        self.speed_fall_up = float(speed_fall_up)
        self.curriculum_success_down = float(curriculum_success_down)
        self.curriculum_box_pass_down = float(curriculum_box_pass_down)
        self.curriculum_fall_down = float(curriculum_fall_down)
        self.speed_master_windows_required = int(speed_master_windows)
        self.speed_master_flat_min = float(speed_master_flat_min)
        self.speed_master_flat_max = float(speed_master_flat_max)
        self.speed_master_severe_overspeed = float(
            speed_master_severe_overspeed
        )
        self.motion_success_up = float(motion_success_up)
        self.motion_box_pass_up = float(motion_box_pass_up)
        self.motion_fall_up = float(motion_fall_up)
        self.motion_flat_speed_max = float(motion_flat_speed_max)
        self.motion_severe_overspeed_max = float(
            motion_severe_overspeed_max
        )
        self.motion_master_windows_required = int(motion_master_windows)
        self.motion_master_action_rate = float(motion_master_action_rate)
        self.motion_master_action_saturation = float(
            motion_master_action_saturation
        )
        self.motion_master_dof_near_limit = float(
            motion_master_dof_near_limit
        )
        self.reference_kl_start_coef = float(reference_kl_start_coef)
        self.reference_kl_stable_windows_required = int(
            reference_kl_stable_windows
        )
        self.reference_kl_stable_decrease = float(
            reference_kl_stable_decrease
        )
        self.reference_kl_regression_increase = float(
            reference_kl_regression_increase
        )
        self.reference_kl_regression_floor = float(
            reference_kl_regression_floor
        )
        self.speed_penalty_level = 0.0
        self.motion_quality_level = 0.0
        self.quality_phase = 0
        self.quality_stage_start_iteration = 0
        self.curriculum_stable_windows = 0
        self.curriculum_regression_windows = 0
        self.speed_master_windows = 0
        self.motion_master_windows = 0
        self.reference_kl_stable_window_count = 0
        self.collapse_windows = 0
        self.collapse_warning = False
        self.curriculum_promoted = False
        self.curriculum_regressed = False
        self.curriculum_phase_transition = False
        self.reward_order_warning = False
        self.current_reference_kl_coef = min(
            max(self.reference_kl_start_coef, self.reference_kl_min_coef),
            self.reference_kl_max_coef,
        )
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
        if not 0.0 <= self.speed_penalty_initial_level <= 1.0:
            raise ValueError("speed_penalty_initial_level must be in [0, 1].")
        if not 0.0 <= self.motion_quality_initial_level <= 1.0:
            raise ValueError("motion_quality_initial_level must be in [0, 1].")
        if self.curriculum_level_step <= 0.0:
            raise ValueError("curriculum_level_step must be positive.")
        if self.curriculum_stage_min_iterations < 1:
            raise ValueError(
                "curriculum_stage_min_iterations must be positive."
            )
        if min(
            self.curriculum_stable_windows_required,
            self.curriculum_regression_windows_required,
            self.speed_master_windows_required,
            self.motion_master_windows_required,
            self.reference_kl_stable_windows_required,
        ) < 1:
            raise ValueError("Curriculum window counts must be positive.")
        self.set_quality_levels(
            self.speed_penalty_initial_level,
            self.motion_quality_initial_level,
        )
        
        # algorithm status
        self.current_learning_iteration = 0

    def _rebuild_optimizer(self, actor_enabled, learning_rate):
        """Build an optimizer containing exactly the parameters for this phase."""
        if actor_enabled:
            parameters = list(self.actor_critic.parameters())
        else:
            parameters = [
                parameter
                for name, parameter in self.actor_critic.named_parameters()
                if not self._is_actor_side_parameter(name)
            ]
        if not parameters:
            raise RuntimeError("No trainable parameters selected for PPO phase.")
        self.optimizer = self.optimizer_class(
            parameters, lr=float(learning_rate)
        )

    def _reset_rollout_actor_equivalence_statistics(self):
        """Replace inference tensors instead of mutating them out of mode."""
        self.rollout_actor_output_max_diff = torch.zeros(
            (), device=self.device
        )
        self.rollout_actor_std_max_diff = torch.zeros(
            (), device=self.device
        )

    def start_critic_warmup(self, start_iteration):
        """Train only the Critic for a fixed number of loaded iterations."""
        if self.critic_warmup_iterations == 0:
            self.critic_warmup_until_iteration = None
            return
        self.critic_warmup_until_iteration = (
            int(start_iteration) + self.critic_warmup_iterations
        )
        self.actor_finetune_active = False
        self.set_quality_levels(0.0, 0.0)
        self.quality_phase = 0
        self.quality_stage_start_iteration = self.critic_warmup_until_iteration
        self.learning_rate = self.warmup_learning_rate
        self._rebuild_optimizer(
            actor_enabled=False,
            learning_rate=self.learning_rate,
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
        for memory_name in ("memory_a", "memory_s", "memory_c"):
            memory = getattr(self.reference_actor_critic, memory_name, None)
            if memory is not None and hasattr(memory, "hidden_states"):
                memory.hidden_states = None
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
        for memory_name in ("memory_a", "memory_s", "memory_c"):
            memory = getattr(self.reference_actor_critic, memory_name, None)
            if memory is not None and hasattr(memory, "hidden_states"):
                memory.hidden_states = None
        self.reference_actor_critic.eval()
        for parameter in self.reference_actor_critic.parameters():
            parameter.requires_grad_(False)

    @property
    def reference_kl_coef(self):
        return self.current_reference_kl_coef

    def set_reference_kl_coef(self, coefficient):
        """Clamp the adaptive reference-policy KL coefficient."""
        self.current_reference_kl_coef = min(
            max(float(coefficient), self.reference_kl_min_coef),
            self.reference_kl_max_coef,
        )

    def set_quality_levels(self, speed_penalty_level, motion_quality_level):
        """Clamp and store independent speed and motion penalty levels."""
        self.speed_penalty_level = self._normalize_curriculum_level(
            speed_penalty_level
        )
        self.motion_quality_level = self._normalize_curriculum_level(
            motion_quality_level
        )

    @staticmethod
    def _normalize_curriculum_level(level):
        """Clamp a level and snap floating-point endpoint noise to 0 or 1."""
        level = min(max(float(level), 0.0), 1.0)
        if level <= 1e-8:
            return 0.0
        if level >= 1.0 - 1e-8:
            return 1.0
        return level

    def set_quality_level(self, level):
        """Backward-compatible helper that sets both curriculum levels."""
        self.set_quality_levels(level, level)

    def activate_actor_finetune(self):
        """Switch from Critic warmup to conservative joint PPO updates."""
        if self.actor_finetune_active:
            return
        self.actor_finetune_active = True
        self.learning_rate = self.actor_finetune_learning_rate
        self.clip_param = self.actor_finetune_clip_param
        self.entropy_coef = self.actor_finetune_entropy_coef
        self._rebuild_optimizer(
            actor_enabled=True,
            learning_rate=self.learning_rate,
        )
        if self.speed_penalty_level <= 0.0:
            self.set_quality_levels(
                self.speed_penalty_initial_level,
                self.motion_quality_initial_level,
            )
            self.quality_stage_start_iteration = self.current_learning_iteration

    def update_quality_curriculum(
        self,
        success_rate,
        box_pass_rate,
        fall_rate,
        episode_count,
        flat_speed_mean=float("inf"),
        flat_severe_overspeed_ratio=1.0,
        mean_action_rate=float("inf"),
        action_saturation_ratio=1.0,
        dof_near_limit_ratio=1.0,
        reward_order_ok=True,
    ):
        """Update staged penalties and adaptive reference-policy protection."""
        self.collapse_warning = False
        self.curriculum_promoted = False
        self.curriculum_regressed = False
        self.curriculum_phase_transition = False
        self.reward_order_warning = reward_order_ok is False
        if (
            self.is_critic_warmup_active()
            or episode_count < self.quality_min_episodes
        ):
            return

        speed_stable = (
            success_rate >= self.speed_success_up
            and box_pass_rate >= self.speed_box_pass_up
            and fall_rate <= self.speed_fall_up
        )
        regress = (
            success_rate < self.curriculum_success_down
            or box_pass_rate < self.curriculum_box_pass_down
            or fall_rate > self.curriculum_fall_down
        )
        collapse = (
            success_rate < self.collapse_success_threshold
            or fall_rate > self.collapse_fall_threshold
        )
        motion_stable = (
            success_rate >= self.motion_success_up
            and box_pass_rate >= self.motion_box_pass_up
            and fall_rate <= self.motion_fall_up
            and flat_speed_mean <= self.motion_flat_speed_max
            and flat_severe_overspeed_ratio
            <= self.motion_severe_overspeed_max
        )
        active_stable = speed_stable if self.quality_phase == 0 else motion_stable

        self.curriculum_stable_windows = (
            self.curriculum_stable_windows + 1 if active_stable else 0
        )
        self.curriculum_regression_windows = (
            self.curriculum_regression_windows + 1 if regress else 0
        )
        self.collapse_windows = self.collapse_windows + 1 if collapse else 0

        if speed_stable:
            self.reference_kl_stable_window_count += 1
            if (
                self.reference_kl_stable_window_count
                >= self.reference_kl_stable_windows_required
            ):
                self.set_reference_kl_coef(
                    self.reference_kl_coef
                    - self.reference_kl_stable_decrease
                )
                self.reference_kl_stable_window_count = 0
        else:
            self.reference_kl_stable_window_count = 0
        if regress:
            self.set_reference_kl_coef(
                self.reference_kl_coef
                + self.reference_kl_regression_increase
            )

        if self.collapse_windows >= self.quality_required_windows:
            self.collapse_warning = True
            self.set_reference_kl_coef(self.reference_kl_max_coef)
            self.collapse_windows = 0

        if (
            self.curriculum_regression_windows
            >= self.curriculum_regression_windows_required
        ):
            if self.quality_phase == 0:
                self.set_quality_levels(
                    max(
                        self.speed_penalty_initial_level,
                        self.speed_penalty_level
                        - self.curriculum_level_step,
                    ),
                    self.motion_quality_level,
                )
            elif self.quality_phase == 1:
                self.set_quality_levels(
                    self.speed_penalty_level,
                    max(
                        0.0,
                        self.motion_quality_level
                        - self.curriculum_level_step,
                    ),
                )
            self.set_reference_kl_coef(
                max(
                    self.reference_kl_coef,
                    self.reference_kl_regression_floor,
                )
            )
            self.curriculum_regressed = True
            self.curriculum_regression_windows = 0
            self.curriculum_stable_windows = 0
            self.quality_stage_start_iteration = self.current_learning_iteration
            return

        stage_age = (
            self.current_learning_iteration
            - self.quality_stage_start_iteration
        )
        can_advance = (
            stage_age >= self.curriculum_stage_min_iterations
            and self.curriculum_stable_windows
            >= self.curriculum_stable_windows_required
        )
        if self.quality_phase == 0:
            if self.speed_penalty_level < 1.0 and can_advance:
                self.set_quality_levels(
                    self.speed_penalty_level
                    + self.curriculum_level_step,
                    self.motion_quality_level,
                )
                self.curriculum_promoted = True
                self.curriculum_stable_windows = 0
                self.quality_stage_start_iteration = (
                    self.current_learning_iteration
                )
            speed_mastered = (
                self.speed_penalty_level >= 1.0
                and speed_stable
                and self.speed_master_flat_min <= flat_speed_mean
                <= self.speed_master_flat_max
                and flat_severe_overspeed_ratio
                <= self.speed_master_severe_overspeed
            )
            self.speed_master_windows = (
                self.speed_master_windows + 1 if speed_mastered else 0
            )
            if self.speed_master_windows >= self.speed_master_windows_required:
                self.quality_phase = 1
                self.set_quality_levels(
                    self.speed_penalty_level,
                    max(
                        self.motion_quality_level,
                        self.motion_quality_initial_level,
                        self.curriculum_level_step,
                    ),
                )
                self.curriculum_phase_transition = True
                self.curriculum_stable_windows = 0
                self.speed_master_windows = 0
                self.quality_stage_start_iteration = (
                    self.current_learning_iteration
                )
        elif self.quality_phase == 1:
            if self.motion_quality_level < 1.0 and can_advance:
                self.set_quality_levels(
                    self.speed_penalty_level,
                    self.motion_quality_level
                    + self.curriculum_level_step,
                )
                self.curriculum_promoted = True
                self.curriculum_stable_windows = 0
                self.quality_stage_start_iteration = (
                    self.current_learning_iteration
                )
            motion_mastered = (
                self.motion_quality_level >= 1.0
                and motion_stable
                and mean_action_rate <= self.motion_master_action_rate
                and action_saturation_ratio
                <= self.motion_master_action_saturation
                and dof_near_limit_ratio
                <= self.motion_master_dof_near_limit
            )
            self.motion_master_windows = (
                self.motion_master_windows + 1 if motion_mastered else 0
            )
            if self.motion_master_windows >= self.motion_master_windows_required:
                self.quality_phase = 2
                self.curriculum_phase_transition = True
                self.motion_master_windows = 0

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
        if self.reference_actor_critic is not None:
            with torch.no_grad():
                self.reference_actor_critic.act(obs)
                self.transition.reference_action_mean = (
                    self.reference_actor_critic.action_mean.detach()
                )
                self.transition.reference_action_sigma = (
                    self.reference_actor_critic.action_std.detach()
                )
                # Both policies run the same one-step recurrent path here.
                # This is the valid equivalence check during Critic warmup;
                # comparing this rollout against padded batch replay also
                # includes expected CUDA GRU accumulation-order differences.
                self.rollout_actor_output_max_diff = torch.maximum(
                    self.rollout_actor_output_max_diff,
                    torch.max(
                        torch.abs(
                            self.transition.action_mean
                            - self.transition.reference_action_mean
                        )
                    ),
                )
                self.rollout_actor_std_max_diff = torch.maximum(
                    self.rollout_actor_std_max_diff,
                    torch.max(
                        torch.abs(
                            self.transition.action_sigma
                            - self.transition.reference_action_sigma
                        )
                    ),
                )
        else:
            self.transition.reference_action_mean = (
                self.transition.action_mean
            )
            self.transition.reference_action_sigma = (
                self.transition.action_sigma
            )
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
        if self.reference_actor_critic is not None:
            self.reference_actor_critic.reset(dones)
    
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
        maximum_stat_names = {
            "reference_kl_max",
            "actor_batched_replay_output_max_diff",
            "actor_batched_replay_std_max_diff",
        }
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
                    if k in maximum_stat_names:
                        average_stats[k] = torch.maximum(
                            torch.as_tensor(
                                average_stats[k], device=v.device
                            ),
                            v.detach(),
                        )
                    else:
                        average_stats[k] = average_stats[k] + v.detach()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                optimized_parameters = [
                    parameter
                    for group in self.optimizer.param_groups
                    for parameter in group["params"]
                ]
                nn.utils.clip_grad_norm_(
                    optimized_parameters, self.max_grad_norm
                )
                self.optimizer.step()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for k in mean_losses.keys():
            mean_losses[k] = mean_losses[k] / num_updates
        for k in average_stats.keys():
            if k not in maximum_stat_names:
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
        average_stats["quality_phase"] = torch.tensor(
            float(self.quality_phase), device=self.device
        )
        average_stats["speed_penalty_level"] = torch.tensor(
            self.speed_penalty_level, device=self.device
        )
        average_stats["motion_quality_level"] = torch.tensor(
            self.motion_quality_level, device=self.device
        )
        average_stats["quality_stage_age"] = torch.tensor(
            float(
                max(
                    self.current_learning_iteration
                    - self.quality_stage_start_iteration,
                    0,
                )
            ),
            device=self.device,
        )
        average_stats["curriculum_promoted"] = torch.tensor(
            float(self.curriculum_promoted), device=self.device
        )
        average_stats["curriculum_regressed"] = torch.tensor(
            float(self.curriculum_regressed), device=self.device
        )
        average_stats["curriculum_phase_transition"] = torch.tensor(
            float(self.curriculum_phase_transition), device=self.device
        )
        average_stats["reference_kl_coef"] = torch.tensor(
            self.reference_kl_coef, device=self.device
        )
        average_stats["collapse_warning"] = torch.tensor(
            float(self.collapse_warning), device=self.device
        )
        average_stats["reward_order_warning"] = torch.tensor(
            float(self.reward_order_warning), device=self.device
        )
        average_stats["actor_parameter_max_diff"] = torch.tensor(
            self.actor_parameter_max_diff(), device=self.device
        )
        average_stats["actor_output_max_diff"] = (
            self.rollout_actor_output_max_diff.detach().clone()
        )
        average_stats["actor_std_max_diff"] = (
            self.rollout_actor_std_max_diff.detach().clone()
        )
        if critic_warmup_active:
            parameter_diff = average_stats[
                "actor_parameter_max_diff"
            ].item()
            output_diff = average_stats.get(
                "actor_output_max_diff",
                torch.zeros((), device=self.device),
            ).item()
            std_diff = average_stats.get(
                "actor_std_max_diff",
                torch.zeros((), device=self.device),
            ).item()
            if (
                parameter_diff
                > self.actor_parameter_equivalence_tolerance
                or output_diff > self.actor_output_equivalence_tolerance
                or std_diff > self.actor_std_equivalence_tolerance
            ):
                raise RuntimeError(
                    "Actor equivalence check failed during Critic-only "
                    "warmup: parameter_max_diff="
                    f"{parameter_diff:.3e} (limit "
                    f"{self.actor_parameter_equivalence_tolerance:.3e}), "
                    f"output_max_diff={output_diff:.3e} (limit "
                    f"{self.actor_output_equivalence_tolerance:.3e}), "
                    f"std_max_diff={std_diff:.3e} (limit "
                    f"{self.actor_std_equivalence_tolerance:.3e})."
                )
        self._reset_rollout_actor_equivalence_statistics()
        self.storage.clear()
        if (
            not critic_warmup_active
            and hasattr(self.actor_critic, "clip_std")
        ):
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
            reference_kl_loss=value_batch.sum() * 0.0,
        )
        if entropy_batch is not None:
            return_["entropy"] = - entropy_batch.mean()

        return_variance = torch.var(minibatch.returns, unbiased=False)
        explained_variance = torch.where(
            return_variance > 1e-8,
            1.0
            - torch.var(
                minibatch.returns - value_batch,
                unbiased=False,
            )
            / return_variance,
            torch.zeros_like(return_variance),
        )
        stats = {"explained_variance": explained_variance}

        if (
            not getattr(self, "_critic_warmup_active", False)
            and self.reference_kl_coef > 0.0
        ):
            reference_mu = minibatch.reference_mu.detach()
            reference_sigma = minibatch.reference_sigma.detach()
            current_sigma = sigma_batch.clamp_min(1e-6)
            reference_sigma = reference_sigma.clamp_min(1e-6)
            # KL(current || frozen reference): sum over action dimensions,
            # then average over all valid rollout samples below.
            reference_kl_per_sample = torch.sum(
                torch.log(reference_sigma / current_sigma)
                + (
                    torch.square(current_sigma)
                    + torch.square(mu_batch - reference_mu)
                )
                / (2.0 * torch.square(reference_sigma))
                - 0.5,
                dim=-1,
            )
            reference_kl = reference_kl_per_sample.mean()
            reference_kl_loss = self.reference_kl_coef * reference_kl
            return_["reference_kl_loss"] = reference_kl_loss
            stats["reference_kl"] = reference_kl.detach()
            stats["reference_kl_max"] = (
                reference_kl_per_sample.detach().max()
            )
            stats["reference_kl_p95"] = torch.quantile(
                reference_kl_per_sample.detach().float(), 0.95
            )
            stats["reference_to_surrogate_ratio"] = (
                reference_kl_loss.detach().abs()
                / surrogate_loss.detach().abs().clamp_min(1e-8)
            )
            stats["actor_batched_replay_output_max_diff"] = torch.max(
                torch.abs(mu_batch.detach() - reference_mu)
            )
            stats["actor_batched_replay_std_max_diff"] = torch.max(
                torch.abs(sigma_batch.detach() - reference_sigma)
            )
        else:
            zero = torch.zeros((), device=value_batch.device)
            stats["reference_kl"] = zero
            stats["reference_kl_max"] = zero
            stats["reference_kl_p95"] = zero
            stats["reference_to_surrogate_ratio"] = zero
            if self.reference_actor_critic is not None:
                stats["actor_batched_replay_output_max_diff"] = torch.max(
                    torch.abs(
                        mu_batch.detach()
                        - minibatch.reference_mu.detach()
                    )
                )
                stats["actor_batched_replay_std_max_diff"] = torch.max(
                    torch.abs(
                        sigma_batch.detach()
                        - minibatch.reference_sigma.detach()
                    )
                )
            else:
                stats["actor_batched_replay_output_max_diff"] = zero
                stats["actor_batched_replay_std_max_diff"] = zero
        
        inter_vars = dict(
            ratio= ratio,
            surrogate= surrogate,
            surrogate_clipped= surrogate_clipped,
        )
        if self.desired_kl != None and self.schedule == 'adaptive':
            inter_vars["kl"] = kl
        if self.use_clipped_value_loss:
            inter_vars["value_clipped"] = value_clipped
        return return_, inter_vars, stats

    def actor_parameter_max_diff(self):
        """Return the maximum Actor-side difference from the frozen source."""
        if self.reference_actor_critic is None:
            return 0.0
        reference_state = self.reference_actor_critic.state_dict()
        maximum = 0.0
        for name, value in self.actor_critic.state_dict().items():
            if not self._is_actor_side_parameter(name):
                continue
            difference = torch.max(
                torch.abs(value.detach() - reference_state[name].detach())
            ).item()
            maximum = max(maximum, difference)
        return maximum

    def state_dict(self):
        state_dict = {
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "reference_model_state_dict": self._reference_actor_state_dict(),
            "algorithm_state_dict": {
                "curriculum_state_version": 11,
                "critic_warmup_until_iteration": (
                    self.critic_warmup_until_iteration
                ),
                "quality_phase": self.quality_phase,
                "speed_penalty_level": self.speed_penalty_level,
                "motion_quality_level": self.motion_quality_level,
                "quality_stage_start_iteration": (
                    self.quality_stage_start_iteration
                ),
                "curriculum_stable_windows": (
                    self.curriculum_stable_windows
                ),
                "curriculum_regression_windows": (
                    self.curriculum_regression_windows
                ),
                "speed_master_windows": self.speed_master_windows,
                "motion_master_windows": self.motion_master_windows,
                "reference_kl_stable_window_count": (
                    self.reference_kl_stable_window_count
                ),
                "collapse_windows": self.collapse_windows,
                "collapse_warning": self.collapse_warning,
                "reward_order_warning": self.reward_order_warning,
                "actor_finetune_active": self.actor_finetune_active,
                "learning_rate": self.learning_rate,
                "clip_param": self.clip_param,
                "entropy_coef": self.entropy_coef,
                "reference_kl_min_coef": self.reference_kl_min_coef,
                "reference_kl_max_coef": self.reference_kl_max_coef,
                "current_reference_kl_coef": (
                    self.current_reference_kl_coef
                ),
            },
        }
        if hasattr(self, "lr_scheduler"):
            state_dict["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        
        return state_dict
    
    def load_state_dict(
        self, state_dict, allow_missing_curriculum_state=False
    ):
        self.actor_critic.load_state_dict(state_dict["model_state_dict"])
        algorithm_state = state_dict.get("algorithm_state_dict", {})
        curriculum_version = algorithm_state.get("curriculum_state_version")
        if (
            self.require_v11_curriculum_state
            and curriculum_version != 11
            and not allow_missing_curriculum_state
        ):
            raise RuntimeError(
                "This task requires a v11 curriculum checkpoint. Use the "
                "explicit initialize_v11_from_v10_warmup manipulator once "
                "for model_11800_warmup.pt; ordinary resume cannot silently "
                "initialize missing curriculum state."
            )
        self.critic_warmup_until_iteration = algorithm_state.get(
            "critic_warmup_until_iteration"
        )
        self.quality_phase = int(algorithm_state.get("quality_phase", 0))
        self.set_quality_levels(
            algorithm_state.get(
                "speed_penalty_level",
                self.speed_penalty_initial_level,
            ),
            algorithm_state.get(
                "motion_quality_level",
                self.motion_quality_initial_level,
            ),
        )
        self.quality_stage_start_iteration = int(
            algorithm_state.get(
                "quality_stage_start_iteration",
                state_dict.get("iter", 0),
            )
        )
        self.curriculum_stable_windows = int(
            algorithm_state.get("curriculum_stable_windows", 0)
        )
        self.curriculum_regression_windows = int(
            algorithm_state.get("curriculum_regression_windows", 0)
        )
        self.speed_master_windows = int(
            algorithm_state.get("speed_master_windows", 0)
        )
        self.motion_master_windows = int(
            algorithm_state.get("motion_master_windows", 0)
        )
        self.reference_kl_stable_window_count = int(
            algorithm_state.get("reference_kl_stable_window_count", 0)
        )
        self.collapse_windows = int(
            algorithm_state.get("collapse_windows", 0)
        )
        self.collapse_warning = bool(
            algorithm_state.get("collapse_warning", False)
        )
        self.reward_order_warning = bool(
            algorithm_state.get("reward_order_warning", False)
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
        self.reference_kl_min_coef = float(
            algorithm_state.get(
                "reference_kl_min_coef", self.reference_kl_min_coef
            )
        )
        self.reference_kl_max_coef = float(
            algorithm_state.get(
                "reference_kl_max_coef", self.reference_kl_max_coef
            )
        )
        self.set_reference_kl_coef(
            algorithm_state.get(
                "current_reference_kl_coef",
                self.reference_kl_start_coef,
            )
        )
        checkpoint_iteration = int(state_dict.get("iter", 0))
        optimizer_is_critic_only = (
            not self.actor_finetune_active
            and self.critic_warmup_until_iteration is not None
            and checkpoint_iteration <= self.critic_warmup_until_iteration
        )
        self._rebuild_optimizer(
            actor_enabled=not optimizer_is_critic_only,
            learning_rate=self.learning_rate,
        )
        if "optimizer_state_dict" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler_state_dict"])
        elif "lr_scheduler_state_dict" in state_dict:
            print("Warning: lr scheduler state dict loaded but no lr scheduler is initialized. Ignored.")
        self._restore_reference_policy(
            state_dict.get("reference_model_state_dict")
        )
