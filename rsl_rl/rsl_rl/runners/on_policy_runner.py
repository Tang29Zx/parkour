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

import time
import os
import json
from collections import deque
import statistics

from tensorboardX import SummaryWriter
import torch

import rsl_rl.algorithms as algorithms
import rsl_rl.modules as modules
from rsl_rl.env import VecEnv
from rsl_rl.utils import ckpt_manipulator


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        actor_critic = modules.build_actor_critic(
            self.env,
            self.cfg["policy_class_name"],
            self.policy_cfg,
        ).to(self.device)

        alg_class = getattr(algorithms, self.cfg["algorithm_class_name"]) # PPO
        self.alg: algorithms.PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.log_interval = self.cfg.get("log_interval", 1)

        _, _ = self.env.reset()
        self._apply_quality_levels_to_env()

    def _apply_quality_levels_to_env(self):
        if hasattr(self, "env") and hasattr(self.env, "set_quality_levels"):
            self.env.set_quality_levels(
                self.alg.speed_penalty_level,
                self.alg.motion_quality_level,
            )
        elif hasattr(self, "env") and hasattr(self.env, "set_quality_level"):
            self.env.set_quality_level(self.alg.speed_penalty_level)

    @staticmethod
    def _config_child(value, key):
        if isinstance(value, dict):
            return value[key]
        return getattr(value, key)

    @classmethod
    def _config_subset(cls, value, template):
        if isinstance(template, dict):
            return {
                key: cls._config_subset(cls._config_child(value, key), child)
                for key, child in template.items()
            }
        if isinstance(template, list):
            return list(value)
        return value

    @classmethod
    def _config_values_equal(cls, left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(
                cls._config_values_equal(left[key], right[key])
                for key in left
            )
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(
                cls._config_values_equal(a, b) for a, b in zip(left, right)
            )
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return abs(float(left) - float(right)) <= 1e-8
        return left == right

    def _verify_actor_runtime_config(
        self,
        checkpoint_path,
        allow_height_grid_expansion=False,
    ):
        """Reject runtime transforms that would change an identical Actor."""
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(checkpoint_path)), "config.json"
        )
        if not os.path.isfile(config_path):
            print(
                "Warning: source config.json is missing; Actor runtime "
                "compatibility could not be checked."
            )
            return False
        with open(config_path, "r") as config_file:
            source = json.load(config_file)
        paths = (
            ("env", "obs_components"),
            ("terrain", "measured_points_x"),
            ("terrain", "measured_points_y"),
            ("normalization", "obs_scales"),
            ("normalization", "height_measurements_offset"),
            ("normalization", "clip_observations"),
            ("normalization", "clip_actions"),
            ("normalization", "clip_actions_method"),
            ("control", "action_scale"),
            ("control", "stiffness"),
            ("control", "damping"),
            ("init_state", "default_joint_angles"),
        )
        mismatches = []
        height_paths = {
            ("terrain", "measured_points_x"),
            ("terrain", "measured_points_y"),
        }
        if allow_height_grid_expansion:
            try:
                source_x = source["terrain"]["measured_points_x"]
                source_y = source["terrain"]["measured_points_y"]
                target_x = self.env.cfg.terrain.measured_points_x
                target_y = self.env.cfg.terrain.measured_points_y
                lateral_padding = len(target_y) - len(source_y)
                aligned = (
                    len(target_x) >= len(source_x)
                    and lateral_padding >= 0
                    and lateral_padding % 2 == 0
                    and self._config_values_equal(
                        source_x, target_x[: len(source_x)]
                    )
                    and self._config_values_equal(
                        source_y,
                        target_y[
                            lateral_padding // 2 :
                            lateral_padding // 2 + len(source_y)
                        ],
                    )
                )
            except (AttributeError, KeyError, TypeError):
                aligned = False
            if not aligned:
                mismatches.append(
                    (
                        "terrain.measured_points",
                        "source grid is not aligned inside the target grid",
                    )
                )
        for path in paths:
            if allow_height_grid_expansion and path in height_paths:
                continue
            source_value = source
            current_value = self.env.cfg
            try:
                for key in path:
                    source_value = self._config_child(source_value, key)
                    current_value = self._config_child(current_value, key)
                current_value = self._config_subset(
                    current_value, source_value
                )
            except (AttributeError, KeyError, TypeError) as error:
                mismatches.append((".".join(path), f"missing: {error}"))
                continue
            if not self._config_values_equal(source_value, current_value):
                mismatches.append(
                    (".".join(path), "source and current values differ")
                )
        for key in (
            "estimator_obs_components",
            "estimator_target_components",
            "replace_state_prob",
            "use_actor_rnn",
            "encoder_component_names",
            "encoder_output_size",
            "rnn_type",
        ):
            if key not in source.get("policy", {}):
                continue
            current_value = self.policy_cfg.get(key)
            if not self._config_values_equal(
                source["policy"][key], current_value
            ):
                mismatches.append(
                    (f"policy.{key}", "source and current values differ")
                )
        if mismatches:
            detail = "; ".join(
                f"{name} ({reason})" for name, reason in mismatches
            )
            raise ValueError(
                "Actor runtime configuration is incompatible with the "
                f"checkpoint: {detail}."
            )
        expansion_note = (
            " with an aligned height-grid expansion"
            if allow_height_grid_expansion
            else ""
        )
        print(
            "Verified Actor runtime compatibility"
            f"{expansion_note}: observations, scaling, history inputs, "
            "action scale, default pose, and PD gains match."
        )
        return True

    @staticmethod
    def _scalar(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().mean().item())
        return float(value)

    @staticmethod
    def _is_maximum_episode_key(key):
        return (
            key.startswith("max_")
            or "_max_" in key
            or key.endswith("_max")
        )

    @staticmethod
    def _is_minimum_episode_key(key):
        return (
            key.startswith("min_")
            or "_min_" in key
            or key.endswith("_min")
        )

    @staticmethod
    def _episode_weight_key(key):
        if key.startswith("raw/") and key.endswith("_mean_return"):
            result_name = key[len("raw/") : -len("_mean_return")]
            return f"raw/{result_name}_episode_count"
        if key.startswith("raw/"):
            for result_name in (
                "success",
                "curriculum_success",
                "severe_body_impact",
                "stagnation",
                "fall_failure",
                "missed_box_failure",
                "out_of_track_failure",
                "landing_overrun",
                "landing_lateral_exit",
                "landing_timeout",
                "incomplete",
                "late_failure",
                "early_failure",
            ):
                if key.startswith(f"raw/{result_name}_"):
                    return f"raw/{result_name}_episode_count"
        if key.startswith("layout_") and key.endswith("_success_rate"):
            return key[: -len("_success_rate")] + "_episode_count"
        if key == "one_box_stage_success_rate":
            return "one_box_stage_episode_count"
        return "num_terminated"

    def _aggregate_episode_infos(self, ep_infos):
        """Aggregate reset-batch summaries with their true episode weights."""
        if not ep_infos:
            return {}
        keys = sorted({key for info in ep_infos for key in info})
        summary = {}
        count_keys = {
            key
            for key in keys
            if key == "num_terminated" or key.endswith("_episode_count")
        }
        for key in keys:
            available = [info for info in ep_infos if key in info]
            if not available:
                continue
            values = [self._scalar(info[key]) for info in available]
            finite_values = [
                value for value in values if torch.isfinite(torch.tensor(value))
            ]
            if self._is_maximum_episode_key(key):
                value = max(finite_values) if finite_values else float("nan")
            elif self._is_minimum_episode_key(key):
                value = min(finite_values) if finite_values else float("nan")
            elif key in count_keys:
                value = sum(values)
            else:
                weight_key = self._episode_weight_key(key)
                weighted_total = 0.0
                total_weight = 0.0
                for info, item_value in zip(available, values):
                    if weight_key not in info:
                        continue
                    weight = self._scalar(info[weight_key])
                    if weight <= 0.0:
                        continue
                    weighted_total += item_value * weight
                    total_weight += weight
                value = (
                    weighted_total / total_weight
                    if total_weight > 0.0
                    else 0.0
                )
            summary[key] = torch.tensor(value, device=self.device)

        success_count = self._scalar(
            summary.get("raw/success_episode_count", 0.0)
        )
        overrun_count = self._scalar(
            summary.get("raw/landing_overrun_episode_count", 0.0)
        )
        lateral_exit_count = self._scalar(
            summary.get("raw/landing_lateral_exit_episode_count", 0.0)
        )
        landing_timeout_count = self._scalar(
            summary.get("raw/landing_timeout_episode_count", 0.0)
        )
        late_count = self._scalar(
            summary.get("raw/late_failure_episode_count", 0.0)
        )
        early_count = self._scalar(
            summary.get("raw/early_failure_episode_count", 0.0)
        )
        reward_order_valid = min(
            success_count,
            landing_timeout_count,
            overrun_count,
            lateral_exit_count,
            late_count,
            early_count,
        ) >= 32.0
        if reward_order_valid:
            success_return = self._scalar(
                summary["raw/success_mean_return"]
            )
            landing_timeout_return = self._scalar(
                summary["raw/landing_timeout_mean_return"]
            )
            overrun_return = self._scalar(
                summary["raw/landing_overrun_mean_return"]
            )
            lateral_exit_return = self._scalar(
                summary["raw/landing_lateral_exit_mean_return"]
            )
            late_return = self._scalar(
                summary["raw/late_failure_mean_return"]
            )
            early_return = self._scalar(
                summary["raw/early_failure_mean_return"]
            )
            best_failure_return = max(
                landing_timeout_return,
                overrun_return,
                lateral_exit_return,
                late_return,
                early_return,
            )
            margin = success_return - best_failure_return
            summary["raw/success_minus_best_failure_return"] = torch.tensor(
                margin, device=self.device
            )
            summary["raw/landing_timeout_minus_overrun_return"] = (
                torch.tensor(
                    landing_timeout_return - overrun_return,
                    device=self.device,
                )
            )
            summary["raw/landing_overrun_minus_late_failure_return"] = (
                torch.tensor(overrun_return - late_return, device=self.device)
            )
            summary["raw/landing_overrun_minus_lateral_exit_return"] = (
                torch.tensor(
                    overrun_return - lateral_exit_return,
                    device=self.device,
                )
            )
            summary["raw/landing_lateral_exit_minus_late_failure_return"] = (
                torch.tensor(
                    lateral_exit_return - late_return,
                    device=self.device,
                )
            )
            summary["raw/landing_overrun_minus_early_failure_return"] = (
                torch.tensor(overrun_return - early_return, device=self.device)
            )
            summary["raw/late_minus_early_failure_return"] = torch.tensor(
                late_return - early_return, device=self.device
            )
            env_cfg = getattr(getattr(self, "env", None), "cfg", None)
            rewards_cfg = getattr(env_cfg, "rewards", None)
            reward_order_mode = getattr(
                rewards_cfg, "reward_order_mode", "ordered_failures"
            )
            if reward_order_mode == "success_above_failures":
                reward_order_ok = success_return > best_failure_return
            elif reward_order_mode == "ordered_failures":
                reward_order_ok = (
                    success_return > landing_timeout_return
                    and landing_timeout_return > overrun_return
                    and overrun_return > lateral_exit_return
                    and success_return > late_return
                    and lateral_exit_return > early_return
                    and late_return > early_return
                )
            else:
                raise ValueError(
                    "Unknown rewards.reward_order_mode "
                    f"{reward_order_mode!r}."
                )
            summary["reward_order_ok"] = torch.tensor(
                float(reward_order_ok), device=self.device
            )
        else:
            summary["reward_order_ok"] = torch.tensor(1.0, device=self.device)
        summary["reward_order_valid"] = torch.tensor(
            float(reward_order_valid), device=self.device
        )
        continuous_warning = False
        for key, value in summary.items():
            if not key.startswith("raw/"):
                continue
            scalar = self._scalar(value)
            if (
                key.endswith("_continuous_penalty_total")
                and scalar < -10.0
            ) or (
                key.endswith("_penalty")
                and not key.endswith("_continuous_penalty_total")
                and scalar < -3.0
            ):
                continuous_warning = True
        summary["continuous_penalty_warning"] = torch.tensor(
            float(continuous_warning), device=self.device
        )
        return summary

    def _get_quality_window(self, episode_summary):
        final_box_key = "box_{}_pass_rate".format(
            self.env.box_progress.required_boxes
        )
        required = (
            "num_terminated",
            "success_rate",
            final_box_key,
            "fall_rate",
            "flat_forward_speed_mean_mps",
            "flat_severe_overspeed_ratio",
            "mean_action_rate_l2",
            "action_saturation_ratio",
            "dof_near_limit_ratio",
        )
        if not all(key in episode_summary for key in required):
            return None
        episode_count = self._scalar(episode_summary["num_terminated"])
        if episode_count <= 0.0:
            return None
        reward_order_valid = bool(
            self._scalar(episode_summary.get("reward_order_valid", 0.0))
        )
        return dict(
            episode_count=episode_count,
            success_rate=self._scalar(episode_summary["success_rate"]),
            box_pass_rate=self._scalar(
                episode_summary[final_box_key]
            ),
            fall_rate=self._scalar(episode_summary["fall_rate"]),
            flat_speed_mean=self._scalar(
                episode_summary["flat_forward_speed_mean_mps"]
            ),
            flat_severe_overspeed_ratio=self._scalar(
                episode_summary["flat_severe_overspeed_ratio"]
            ),
            mean_action_rate=self._scalar(
                episode_summary["mean_action_rate_l2"]
            ),
            action_saturation_ratio=self._scalar(
                episode_summary["action_saturation_ratio"]
            ),
            dof_near_limit_ratio=self._scalar(
                episode_summary["dof_near_limit_ratio"]
            ),
            reward_order_valid=reward_order_valid,
            reward_order_ok=bool(
                self._scalar(episode_summary.get("reward_order_ok", 1.0))
            ),
        )

    def _update_quality_curriculum(self, episode_summary, stats):
        window = self._get_quality_window(episode_summary)
        uses_task_curriculum = bool(
            getattr(self.env, "uses_task_curriculum", False)
        )
        if window is not None and not uses_task_curriculum:
            self.alg.update_quality_curriculum(**window)
            self._apply_quality_levels_to_env()
        stats["quality_phase"] = torch.tensor(
            float(self.alg.quality_phase), device=self.device
        )
        stats["speed_penalty_level"] = torch.tensor(
            self.alg.speed_penalty_level, device=self.device
        )
        stats["motion_quality_level"] = torch.tensor(
            self.alg.motion_quality_level, device=self.device
        )
        stats["quality_stage_age"] = torch.tensor(
            float(
                max(
                    self.alg.current_learning_iteration
                    - self.alg.quality_stage_start_iteration,
                    0,
                )
            ),
            device=self.device,
        )
        stats["curriculum_promoted"] = torch.tensor(
            float(self.alg.curriculum_promoted), device=self.device
        )
        stats["curriculum_regressed"] = torch.tensor(
            float(self.alg.curriculum_regressed), device=self.device
        )
        stats["curriculum_phase_transition"] = torch.tensor(
            float(self.alg.curriculum_phase_transition), device=self.device
        )
        stats["reference_kl_coef"] = torch.tensor(
            self.alg.reference_kl_coef, device=self.device
        )
        stats["collapse_warning"] = torch.tensor(
            float(self.alg.collapse_warning), device=self.device
        )
        stats["reward_order_warning"] = torch.tensor(
            float(self.alg.reward_order_warning), device=self.device
        )
        stats["actor_runtime_config_compatible"] = torch.tensor(
            float(getattr(self, "actor_runtime_config_compatible", False)),
            device=self.device,
        )
        action_scale = self.env.cfg.control.action_scale
        if isinstance(action_scale, torch.Tensor):
            max_action_scale = float(action_scale.abs().max().item())
        elif isinstance(action_scale, (list, tuple)):
            max_action_scale = max(abs(float(value)) for value in action_scale)
        else:
            max_action_scale = abs(float(action_scale))
        stats["actor_scaled_output_max_diff"] = (
            stats.get(
                "actor_output_max_diff",
                torch.zeros((), device=self.device),
            )
            * max_action_scale
        )
        if self.alg.collapse_warning:
            protection_message = (
                " and a stronger reference-policy constraint"
                if self.alg.reference_kl_max_coef > 0.0
                else ""
            )
            print(
                "\033[1;31m"
                "WARNING: Actor metrics indicate policy collapse. Training "
                "continues with reduced quality difficulty"
                f"{protection_message}."
                "\033[0m"
            )
        if self.alg.reward_order_warning:
            print(
                "\033[1;31m"
                "WARNING: successful episodes no longer outperform the "
                "best sufficiently sampled failure outcome."
                "\033[0m"
            )

    def _update_task_curriculum(self, episode_summary, stats):
        """Update an environment-owned task curriculum at log boundaries."""
        if not hasattr(self.env, "update_task_curriculum"):
            return
        if not self.alg.is_critic_warmup_active(
            self.current_learning_iteration
        ):
            self.env.update_task_curriculum(
                episode_summary, self.current_learning_iteration
            )
        for name, value in self.env.get_task_curriculum_statistics(
            self.current_learning_iteration
        ).items():
            stats[name] = torch.tensor(float(value), device=self.device)

    def _save_warmup_boundary_checkpoint(self):
        warmup_until = self.alg.critic_warmup_until_iteration
        if (
            self.log_dir is None
            or warmup_until is None
            or self.current_learning_iteration != warmup_until
            or self.alg.actor_finetune_active
        ):
            return
        paths = (
            os.path.join(self.log_dir, f"model_{warmup_until}.pt"),
            os.path.join(
                self.log_dir,
                f"model_{warmup_until}_warmup.pt",
            ),
        )
        for path in paths:
            if not os.path.exists(path):
                self.save(path)
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rframebuffer = deque(maxlen=2000)
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        print("Initialization done, start learning.")
        print("Non-finite rewards or PPO losses stop this run immediately.")
        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        tot_start_time = time.time()
        start = time.time()
        while self.current_learning_iteration < tot_iter:
            # Rollout
            with torch.inference_mode(self.cfg.get("inference_mode_rollout", True)):
                for i in range(self.num_steps_per_env):
                    obs, critic_obs, rewards, dones, infos = self.rollout_step(obs, critic_obs)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        # ``env.extras`` persists between resets in this
                        # codebase. Only consume episode summaries on a step
                        # that actually terminated at least one environment.
                        if 'episode' in infos and new_ids.numel() > 0:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        rframebuffer.extend(rewards[dones < 1].cpu().numpy().tolist())
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            losses, stats = self.alg.update(self.current_learning_iteration)
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None and self.current_learning_iteration % self.log_interval == 0:
                episode_summary = self._aggregate_episode_infos(ep_infos)
                self._update_task_curriculum(episode_summary, stats)
                self._update_quality_curriculum(episode_summary, stats)
                self.log(locals())
                ep_infos.clear()
                if self.alg.reward_order_warning:
                    self.save(
                        os.path.join(
                            self.log_dir,
                            "model_{}_reward_order_stop.pt".format(
                                self.current_learning_iteration
                            ),
                        )
                    )
                    raise RuntimeError(
                        "Reward ordering failed for two consecutive valid "
                        "log windows; training stopped before further PPO "
                        "updates."
                    )
            is_warmup_boundary = (
                self.alg.critic_warmup_until_iteration is not None
                and self.current_learning_iteration
                == self.alg.critic_warmup_until_iteration
            )
            if (
                self.current_learning_iteration % self.save_interval == 0
                and self.current_learning_iteration > start_iter
                and not is_warmup_boundary
            ):
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            self.current_learning_iteration = self.current_learning_iteration + 1
            self._save_warmup_boundary_checkpoint()
            start = time.time()
        
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def rollout_step(self, obs, critic_obs):
        actions = self.alg.act(obs, critic_obs)
        obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
        self.alg.process_env_step(rewards, dones, infos, obs, critic_obs)
        return obs, critic_obs, rewards, dones, infos

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time = time.time() - locs['tot_start_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        episode_summary = locs.get("episode_summary", {})
        if episode_summary:
            for key, value in episode_summary.items():
                if key.startswith("raw/"):
                    tensorboard_key = "EpisodeRaw/" + key[len("raw/"):]
                else:
                    tensorboard_key = "Episode/" + key
                self.writer.add_scalar(tensorboard_key, value, self.current_learning_iteration)
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        for k, v in locs["losses"].items():
            self.writer.add_scalar("Loss/" + k, v.item(), self.current_learning_iteration)
        loss_stat_names = {
            "reference_kl",
            "reference_kl_max",
            "reference_kl_p95",
            "reference_to_surrogate_ratio",
            "explained_variance",
        }
        for k, v in locs["stats"].items():
            namespace = "Loss/" if k in loss_stat_names else "Train/"
            self.writer.add_scalar(
                namespace + k, v.item(), self.current_learning_iteration
            )
        
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, self.current_learning_iteration)
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), self.current_learning_iteration)
        self.writer.add_scalar('Perf/total_fps', fps, self.current_learning_iteration)
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], self.current_learning_iteration)
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], self.current_learning_iteration)
        if torch.device(self.device).type == "cuda":
            self.writer.add_scalar('Perf/gpu_allocated', torch.cuda.memory_allocated(self.device) / 1024 ** 3, self.current_learning_iteration)
            self.writer.add_scalar('Perf/gpu_global_free_mem', torch.cuda.mem_get_info(self.device)[0] / 1024 ** 3, self.current_learning_iteration)
            self.writer.add_scalar('Perf/gpu_total', torch.cuda.mem_get_info(self.device)[1] / 1024 ** 3, self.current_learning_iteration)
        self.writer.add_scalar('Train/mean_reward_each_timestep', statistics.mean(locs['rframebuffer']), self.current_learning_iteration)
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), self.current_learning_iteration)
            self.writer.add_scalar('Train/ratio_above_mean_reward', statistics.mean([(1. if rew > statistics.mean(locs['rewbuffer']) else 0) for rew in locs['rewbuffer']]), self.current_learning_iteration)
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), self.current_learning_iteration)
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {self.current_learning_iteration}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            )
            for k, v in locs["losses"].items():
                log_string += f"""{k:>{pad}} {v.item():.4f}\n"""
            log_string += (
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                # f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                # f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            )
            for k, v in locs["losses"].items():
                log_string += f"""{k:>{pad}} {v.item():.4f}\n"""
            log_string += (
                f"""{'Value function loss:':>{pad}} {locs["losses"]['value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs["losses"]['surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                # f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                # f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (self.current_learning_iteration + 1 - locs["start_iter"]) * (
                               locs['tot_iter'] - self.current_learning_iteration):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        run_state_dict = self.alg.state_dict()
        run_state_dict.update({
            'iter': self.current_learning_iteration,
            'infos': infos,
        })
        env = getattr(self, "env", None)
        if env is not None and hasattr(env, "get_task_curriculum_state"):
            task_state = env.get_task_curriculum_state()
            if task_state is not None:
                run_state_dict["task_curriculum_state_dict"] = task_state
        torch.save(run_state_dict, path)

    def load(self, path, load_optimizer=True):
        manipulator_name = self.cfg.get("ckpt_manipulator", False)
        expands_rough_height_grid = (
            manipulator_name == "initialize_one_box_from_rough2000"
        )
        self.actor_runtime_config_compatible = (
            self._verify_actor_runtime_config(
                path,
                allow_height_grid_expansion=expands_rough_height_grid,
            )
        )
        loaded_dict = torch.load(path, map_location=self.device)
        if manipulator_name:
            # suppose to be a string specifying which function to use
            print("\033[1;36m Warning: using a hacky way to load the model. \033[0m")
            loaded_dict = getattr(ckpt_manipulator, manipulator_name)(
                loaded_dict,
                self.alg.state_dict(),
            )
            print("\033[1;36m Done: using a hacky way to load the model. \033[0m")
        self.alg.load_state_dict(
            loaded_dict,
            allow_missing_curriculum_state=bool(manipulator_name),
        )
        self.current_learning_iteration = loaded_dict['iter']
        if manipulator_name in {
            "reset_critic_and_optimizer",
            "initialize_one_box_from_rough2000",
        }:
            self.alg.start_critic_warmup(self.current_learning_iteration)
            self.alg.set_quality_levels(0.0, 0.0)
            if self.alg.reference_kl_max_coef > 0.0:
                self.alg.snapshot_reference_policy()
            else:
                self.alg.reference_actor_critic = None
        env = getattr(self, "env", None)
        if bool(getattr(env, "uses_task_curriculum", False)):
            if manipulator_name == "initialize_one_box_from_rough2000":
                curriculum_start_iteration = (
                    self.alg.critic_warmup_until_iteration
                    if self.alg.critic_warmup_until_iteration is not None
                    else self.current_learning_iteration
                )
                env.initialize_task_curriculum(
                    curriculum_start_iteration
                )
            else:
                env.load_task_curriculum_state(
                    loaded_dict.get("task_curriculum_state_dict")
                )
            # Apply the restored stage and height to every environment before
            # the first rollout. learn() fetches fresh observations afterward.
            env.reset()
        self._apply_quality_levels_to_env()
        if manipulator_name:
            try:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            except:
                print("\033[1;36m Save manipulated checkpoint failed, ignored... \033[0m")
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
