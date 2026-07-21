"""Go2 environment support for ordered random-box parkour courses."""

import numpy as np
import torch
from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse

from .box_progress import BoxProgressTracker
from .legged_robot import LeggedRobot


class LeggedRobotBox(LeggedRobot):
    """Add ordered box-course progress and termination to ``LeggedRobot``."""

    def _init_buffers(self):
        super()._init_buffers()
        if not hasattr(self.terrain, "box_bounds_pyt"):
            raise TypeError("LeggedRobotBox requires terrain box-bound tensors.")
        if len(self.feet_indices) != 4:
            raise ValueError("LeggedRobotBox currently requires exactly four feet.")

        self.front_foot_local_indices = self._foot_local_indices(
            ("FL", "FR"), "front"
        )
        self.rear_foot_local_indices = self._foot_local_indices(
            ("RL", "RR"), "rear"
        )
        self.rear_upper_joint_names = (
            "RL_hip_joint",
            "RR_hip_joint",
            "RL_thigh_joint",
            "RR_thigh_joint",
        )
        missing_rear_upper_joints = [
            name
            for name in self.rear_upper_joint_names
            if name not in self.dof_names
        ]
        if missing_rear_upper_joints:
            raise ValueError(
                "The robot asset is missing rear upper-leg joints: "
                + ", ".join(missing_rear_upper_joints)
            )
        self.rear_upper_joint_indices = torch.tensor(
            [self.dof_names.index(name) for name in self.rear_upper_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        progress_cfg = self.cfg.box_progress
        if progress_cfg.flat_low_base_height_threshold <= 0.0:
            raise ValueError(
                "flat_low_base_height_threshold must be positive."
            )
        if progress_cfg.flat_low_base_height_steps <= 0:
            raise ValueError("flat_low_base_height_steps must be positive.")
        self.box_progress = BoxProgressTracker(
            num_envs=self.num_envs,
            num_feet=len(self.feet_indices),
            num_boxes=self.terrain.num_boxes,
            device=self.device,
            required_boxes=progress_cfg.required_boxes,
            pass_margin=progress_cfg.pass_margin,
            top_contact_tolerance=progress_cfg.top_contact_tolerance,
            contact_force_threshold=progress_cfg.contact_force_threshold,
            front_foot_indices=self.front_foot_local_indices.tolist(),
            rear_foot_indices=self.rear_foot_local_indices.tolist(),
            front_contact_required_steps=(
                progress_cfg.front_contact_required_steps
            ),
            rear_contact_required_steps=(
                progress_cfg.rear_contact_required_steps
            ),
            landing_steps=progress_cfg.landing_steps,
            landing_min_current_feet=progress_cfg.landing_min_current_feet,
            landing_require_rear_foot=progress_cfg.landing_require_rear_foot,
            landing_roll_threshold=progress_cfg.landing_roll_threshold,
            landing_pitch_threshold=progress_cfg.landing_pitch_threshold,
            landing_base_height_threshold=(
                progress_cfg.landing_base_height_threshold
            ),
            landing_vertical_speed_threshold=(
                progress_cfg.landing_vertical_speed_threshold
            ),
            landing_forward_speed_threshold=(
                progress_cfg.landing_forward_speed_threshold
            ),
            landing_deadline_steps=progress_cfg.landing_deadline_steps,
            body_contact_window_steps=progress_cfg.body_contact_window_steps,
            body_contact_failure_steps=(
                progress_cfg.body_contact_failure_steps
            ),
            severe_body_impact_force=progress_cfg.severe_body_impact_force,
            roll_threshold=progress_cfg.roll_threshold,
            pitch_threshold=progress_cfg.pitch_threshold,
            base_height_threshold=progress_cfg.base_height_threshold,
            lateral_limit=progress_cfg.lateral_limit,
        )
        self._bind_progress_buffers()
        self._refresh_box_course_data()
        self.forward_speed_sum = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.max_forward_speed = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.flat_forward_speed_sum = torch.zeros_like(self.forward_speed_sum)
        self.flat_forward_speed_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.flat_forward_speed_max = torch.zeros_like(self.forward_speed_sum)
        self.box_forward_speed_sum = torch.zeros_like(self.forward_speed_sum)
        self.box_forward_speed_count = torch.zeros_like(
            self.flat_forward_speed_count
        )
        self.box_forward_speed_max = torch.zeros_like(self.forward_speed_sum)
        self.flat_overspeed_count = torch.zeros_like(
            self.flat_forward_speed_count
        )
        self.flat_severe_overspeed_count = torch.zeros_like(
            self.flat_forward_speed_count
        )
        self.box_overspeed_count = torch.zeros_like(
            self.flat_forward_speed_count
        )
        self.flat_base_height_sum = torch.zeros_like(self.forward_speed_sum)
        self.flat_base_height_count = torch.zeros_like(
            self.flat_forward_speed_count
        )
        self.min_flat_base_height = torch.full_like(
            self.forward_speed_sum,
            10.0,
        )
        self.flat_low_base_height_count = torch.zeros_like(
            self.flat_forward_speed_count
        )
        self.flat_low_base_height_counter = torch.zeros_like(
            self.flat_forward_speed_count
        )
        self.max_flat_low_base_height_steps = torch.zeros_like(
            self.flat_forward_speed_count
        )
        self.flat_low_base_height_failure_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.flat_foot_contact_count = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.long,
            device=self.device,
        )
        self.flat_foot_contact_transition_count = torch.zeros_like(
            self.flat_foot_contact_count
        )
        self.previous_flat_foot_contact = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
        )
        self.previous_flat_contact_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.abs_roll_sum = torch.zeros_like(self.forward_speed_sum)
        self.max_abs_roll = torch.zeros_like(self.forward_speed_sum)
        self.abs_pitch_sum = torch.zeros_like(self.forward_speed_sum)
        self.max_abs_pitch = torch.zeros_like(self.forward_speed_sum)
        self.action_rate_l2_sum = torch.zeros_like(self.forward_speed_sum)
        self.max_action_rate_l2 = torch.zeros_like(self.forward_speed_sum)
        self.named_dof_error_sum = torch.zeros_like(self.forward_speed_sum)
        self.max_named_dof_error = torch.zeros_like(self.forward_speed_sum)
        self.named_joint_error_sum = torch.zeros(
            self.num_envs,
            len(self.dof_error_named_indices),
            dtype=torch.float,
            device=self.device,
        )
        self.named_joint_error_max = torch.zeros_like(
            self.named_joint_error_sum
        )
        self._init_rear_upper_joint_statistics()
        self.dof_near_limit_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.action_saturation_count = torch.zeros_like(
            self.dof_near_limit_count
        )
        self.action_abs_sum = torch.zeros_like(self.forward_speed_sum)
        self.left_foot_crossing_count = torch.zeros_like(
            self.dof_near_limit_count
        )
        self.right_foot_crossing_count = torch.zeros_like(
            self.dof_near_limit_count
        )
        self.min_lr_foot_lateral_distance = torch.full_like(
            self.forward_speed_sum, 10.0
        )
        self.thigh_collision_count = torch.zeros_like(
            self.dof_near_limit_count
        )
        self.calf_collision_count = torch.zeros_like(
            self.dof_near_limit_count
        )
        self.body_collision_count = torch.zeros_like(
            self.dof_near_limit_count
        )
        rear_support_window_steps = int(
            self.cfg.rewards.rear_support_window_steps
        )
        if rear_support_window_steps <= 0:
            raise ValueError("rear_support_window_steps must be positive.")
        self.rear_support_history = torch.zeros(
            self.num_envs,
            rear_support_window_steps,
            dtype=torch.bool,
            device=self.device,
        )
        self.airborne_history = torch.zeros_like(self.rear_support_history)
        self.rear_support_valid_history = torch.zeros_like(
            self.rear_support_history
        )
        self.rear_support_history_index = 0
        self.rear_support_window_ratio = torch.zeros_like(
            self.forward_speed_sum
        )
        self.flat_airborne_window_ratio = torch.zeros_like(
            self.forward_speed_sum
        )
        self.rear_support_missing_count = torch.zeros_like(
            self.dof_near_limit_count
        )
        self.max_rear_support_missing_steps = torch.zeros_like(
            self.dof_near_limit_count
        )
        self.speed_penalty_level = float(
            getattr(
                self.cfg.rewards,
                "speed_penalty_initial_level",
                0.0,
            )
        )
        self.motion_quality_level = float(
            getattr(
                self.cfg.rewards,
                "motion_quality_initial_level",
                0.0,
            )
        )
        self.progress_reward_start = torch.zeros_like(
            self.forward_speed_sum
        )
        self.rewarded_progress_ratio = torch.zeros_like(
            self.forward_speed_sum
        )
        self.course_progress_delta_buf = torch.zeros_like(
            self.forward_speed_sum
        )
        self.progress_reward_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_ground_contact_feet_sum = torch.zeros_like(
            self.forward_speed_sum
        )
        self.current_rear_support_count = torch.zeros_like(
            self.dof_near_limit_count
        )
        self.flat_airborne_count = torch.zeros_like(
            self.dof_near_limit_count
        )
        self.rear_excursion_raw_sum = torch.zeros_like(
            self.forward_speed_sum
        )
        self.rear_excursion_raw_max = torch.zeros_like(
            self.forward_speed_sum
        )

        rigid_body_names = self.gym.get_actor_rigid_body_names(
            self.envs[0], self.actor_handles[0]
        )
        base_names = [name for name in rigid_body_names if "base" in name]
        if not base_names:
            raise ValueError("The robot asset does not contain a base rigid body.")
        self.base_contact_indices = torch.tensor(
            [
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], name
                )
                for name in base_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.thigh_contact_indices = self._body_indices_with_token(
            rigid_body_names, "thigh"
        )
        self.calf_contact_indices = self._body_indices_with_token(
            rigid_body_names, "calf"
        )
        self.left_foot_local_indices = torch.tensor(
            [
                idx
                for idx, name in enumerate(self.feet_names)
                if name.startswith(("FL", "RL"))
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.right_foot_local_indices = torch.tensor(
            [
                idx
                for idx, name in enumerate(self.feet_names)
                if name.startswith(("FR", "RR"))
            ],
            dtype=torch.long,
            device=self.device,
        )
        if (
            len(self.left_foot_local_indices) != 2
            or len(self.right_foot_local_indices) != 2
        ):
            raise ValueError(
                "Go2 leg-crossing metrics require FL/RL and FR/RR foot names."
            )

    def _foot_local_indices(self, prefixes, group_name):
        indices = torch.tensor(
            [
                index
                for index, name in enumerate(self.feet_names)
                if name.startswith(prefixes)
            ],
            dtype=torch.long,
            device=self.device,
        )
        if len(indices) != 2:
            raise ValueError(
                f"Go2 {group_name} foot group requires exactly two feet; "
                f"found {[self.feet_names[index] for index in indices.tolist()]}."
            )
        return indices

    def _body_indices_with_token(self, body_names, token):
        names = [name for name in body_names if token in name]
        if not names:
            raise ValueError(f"The robot asset has no {token!r} rigid bodies.")
        return torch.tensor(
            [
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], name
                )
                for name in names
            ],
            dtype=torch.long,
            device=self.device,
        )

    def _bind_progress_buffers(self):
        tracker = self.box_progress
        self.next_box_idx = tracker.next_box_idx
        self.passed_box_count = tracker.passed_box_count
        self.foot_contact_mask = tracker.foot_contact_mask
        self.front_contact_counter = tracker.front_contact_counter
        self.rear_contact_counter = tracker.rear_contact_counter
        self.box_passed_buf = tracker.box_passed_buf
        self.front_foot_contact_buf = tracker.front_foot_contact_buf
        self.rear_foot_contact_buf = tracker.rear_foot_contact_buf
        self.success_buf = tracker.success_buf
        self.missed_box_buf = tracker.missed_box_buf
        self.out_of_track_buf = tracker.out_of_track_buf
        self.landing_overrun_buf = tracker.landing_overrun_buf
        self.landing_timeout_buf = tracker.landing_timeout_buf
        self.landing_phase_entry_buf = tracker.landing_phase_entry_buf
        self.lateral_out_of_track_buf = tracker.lateral_out_of_track_buf
        self.backward_out_of_track_buf = tracker.backward_out_of_track_buf
        self.generic_failure_buf = tracker.generic_failure_buf
        self.fall_buf = tracker.fall_buf
        self.incomplete_buf = tracker.incomplete_buf
        self.task_progress_buf = tracker.task_progress_buf
        self.body_contact_window_count = tracker.body_contact_window_count
        self.body_contact_window_failure_buf = (
            tracker.body_contact_window_failure_buf
        )
        self.severe_body_impact_buf = tracker.severe_body_impact_buf
        self.landing_counter = tracker.landing_counter
        self.landing_phase_start_step = tracker.landing_phase_start_step
        self.steps_in_landing_phase = tracker.steps_in_landing_phase
        self.best_landing_score = tracker.best_landing_score
        self.best_landing_hold_steps = tracker.best_landing_hold_steps
        self.landing_quality_delta = tracker.landing_quality_delta
        self.landing_hold_delta = tracker.landing_hold_delta
        self.episode_timeout_buf = tracker.episode_timeout_buf

    def _update_course_progress_reward(self):
        """Reward only a new per-episode high-water mark in course progress."""
        current_progress = self.task_progress_buf.clamp(0.0, 1.0)
        uninitialized = ~self.progress_reward_initialized
        if uninitialized.any():
            self.progress_reward_start[uninitialized] = current_progress[
                uninitialized
            ]
            self.rewarded_progress_ratio[uninitialized] = 0.0
            self.progress_reward_initialized[uninitialized] = True

        denominator = (1.0 - self.progress_reward_start).clamp_min(1e-6)
        normalized_progress = (
            (current_progress - self.progress_reward_start) / denominator
        ).clamp(0.0, 1.0)
        new_high_water_mark = torch.maximum(
            self.rewarded_progress_ratio,
            normalized_progress,
        )
        self.course_progress_delta_buf[:] = (
            new_high_water_mark - self.rewarded_progress_ratio
        ).clamp_min(0.0)
        self.course_progress_delta_buf[uninitialized] = 0.0
        self.rewarded_progress_ratio[:] = new_high_water_mark

    def _refresh_box_course_data(self):
        track_indices = torch.stack(
            (self.terrain_levels, self.terrain_types), dim=1
        )
        self.env_box_bounds = self.terrain.get_box_bounds(track_indices)
        physical_track_indices = (
            self.terrain_levels * self.cfg.terrain.num_cols
            + self.terrain_types
        )
        self.layout_indices = torch.remainder(
            physical_track_indices, self.terrain.num_unique_layouts
        )
        spawn_margin = float(self.terrain.track_kwargs["spawn_margin"])
        self.track_start_x = self.env_origins[:, 0] - spawn_margin
        self.track_end_x = self.track_start_x + float(self.terrain.env_length)

        required_boxes = self.box_progress.required_boxes
        if required_boxes < self.terrain.num_boxes:
            self.course_landing_end_x = self.env_box_bounds[
                :, required_boxes, 0
            ]
        else:
            self.course_landing_end_x = self.track_end_x.clone()
        minimum_length = float(self.cfg.box_progress.min_landing_zone_length)
        all_box_bounds = self.terrain.box_bounds_pyt
        all_course_rear = all_box_bounds[:, :, required_boxes - 1, 1]
        if required_boxes < self.terrain.num_boxes:
            all_landing_end_x = all_box_bounds[:, :, required_boxes, 0]
        else:
            all_landing_end_x = (
                self.terrain.env_origins_pyt[:, :, 0]
                - spawn_margin
                + float(self.terrain.env_length)
            )
        actual_minimum = float(
            torch.min(all_landing_end_x - all_course_rear).item()
        )
        if actual_minimum + 1e-6 < minimum_length:
            raise ValueError(
                "The shortest course landing zone is "
                f"{actual_minimum:.3f} m, below the configured minimum "
                f"of {minimum_length:.3f} m."
            )

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        forward_speed = self.base_lin_vel[:, 0]
        box_mask = self._near_box_for_speed_control()
        landing_phase = (
            self.next_box_idx >= self.box_progress.required_boxes
        )
        flat_mask = ~box_mask & ~landing_phase
        self._update_speed_statistics(
            forward_speed, box_mask, landing_phase
        )
        self._update_flat_gait_statistics(flat_mask)
        self._update_motion_quality_statistics()
        self._update_rear_upper_joint_statistics(box_mask)
        self._update_rear_support_state(box_mask)

    def _init_rear_upper_joint_statistics(self):
        """Allocate flat/box statistics for the rear hip and thigh joints."""
        shape = (self.num_envs, len(self.rear_upper_joint_indices))
        self.rear_upper_flat_sample_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.rear_upper_box_sample_count = torch.zeros_like(
            self.rear_upper_flat_sample_count
        )
        self.rear_upper_flat_action_sample_count = torch.zeros_like(
            self.rear_upper_flat_sample_count
        )
        self.rear_upper_box_action_sample_count = torch.zeros_like(
            self.rear_upper_flat_sample_count
        )
        self.rear_upper_flat_pos_min = torch.full(
            shape, 10.0, dtype=torch.float, device=self.device
        )
        self.rear_upper_flat_pos_max = torch.full(
            shape, -10.0, dtype=torch.float, device=self.device
        )
        self.rear_upper_box_pos_min = torch.full_like(
            self.rear_upper_flat_pos_min, 10.0
        )
        self.rear_upper_box_pos_max = torch.full_like(
            self.rear_upper_flat_pos_max, -10.0
        )
        self.rear_upper_flat_vel_sq_sum = torch.zeros_like(
            self.rear_upper_flat_pos_min
        )
        self.rear_upper_box_vel_sq_sum = torch.zeros_like(
            self.rear_upper_flat_pos_min
        )
        self.rear_upper_flat_action_delta_sq_sum = torch.zeros_like(
            self.rear_upper_flat_pos_min
        )
        self.rear_upper_box_action_delta_sq_sum = torch.zeros_like(
            self.rear_upper_flat_pos_min
        )

    def _update_rear_upper_joint_statistics(self, box_mask):
        """Track physical rear-joint swing separately on flat and box areas."""
        joint_indices = self.rear_upper_joint_indices
        joint_pos = self.dof_pos[:, joint_indices]
        joint_vel = self.dof_vel[:, joint_indices]
        action_delta = (
            self.actions[:, joint_indices]
            - self.last_actions[:, joint_indices]
        )
        action_valid = self.episode_length_buf > 1

        for region, region_mask in (("flat", ~box_mask), ("box", box_mask)):
            mask = region_mask.unsqueeze(1)
            sample_count = getattr(
                self, f"rear_upper_{region}_sample_count"
            )
            action_sample_count = getattr(
                self, f"rear_upper_{region}_action_sample_count"
            )
            position_min = getattr(
                self, f"rear_upper_{region}_pos_min"
            )
            position_max = getattr(
                self, f"rear_upper_{region}_pos_max"
            )
            velocity_sum = getattr(
                self, f"rear_upper_{region}_vel_sq_sum"
            )
            action_delta_sum = getattr(
                self, f"rear_upper_{region}_action_delta_sq_sum"
            )

            sample_count += region_mask
            position_min[:] = torch.where(
                mask, torch.minimum(position_min, joint_pos), position_min
            )
            position_max[:] = torch.where(
                mask, torch.maximum(position_max, joint_pos), position_max
            )
            velocity_sum += joint_vel.square() * mask

            valid_action_mask = region_mask & action_valid
            action_sample_count += valid_action_mask
            action_delta_sum += (
                action_delta.square() * valid_action_mask.unsqueeze(1)
            )

    def _update_rear_support_state(self, box_mask):
        """Track rear support and airborne ratios on valid flat-ground steps."""
        foot_contact = (
            torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
            > self.cfg.box_progress.contact_force_threshold
        )
        rear_support = foot_contact[:, self.rear_foot_local_indices].any(dim=1)
        airborne = ~foot_contact.any(dim=1)
        landing_phase = self.next_box_idx >= self.box_progress.required_boxes
        valid_flat = ~box_mask & ~landing_phase
        history_index = self.rear_support_history_index
        self.rear_support_history[:, history_index] = (
            rear_support & valid_flat
        )
        self.airborne_history[:, history_index] = airborne & valid_flat
        self.rear_support_valid_history[:, history_index] = valid_flat
        self.rear_support_history_index = (
            self.rear_support_history_index + 1
        ) % self.rear_support_history.shape[1]
        valid_count = self.rear_support_valid_history.sum(dim=1)
        denominator = valid_count.clamp_min(1).float()
        self.rear_support_window_ratio[:] = (
            self.rear_support_history.sum(dim=1).float() / denominator
        )
        self.flat_airborne_window_ratio[:] = (
            self.airborne_history.sum(dim=1).float() / denominator
        )
        target_ratio = float(self.cfg.rewards.rear_support_target_ratio)
        rear_deficit = torch.relu(
            (target_ratio - self.rear_support_window_ratio)
            / max(target_ratio, 1e-6)
        )
        self.max_rear_support_missing_steps[:] = torch.maximum(
            self.max_rear_support_missing_steps,
            (rear_deficit * valid_count.float()).to(torch.long),
        )
        self.rear_support_missing_count += (
            valid_flat
            & (self.episode_length_buf > self.rear_support_history.shape[1])
            & (rear_deficit > 0.0)
        )
        self.current_ground_contact_feet_sum += foot_contact.sum(dim=1)
        self.current_rear_support_count += rear_support
        self.flat_airborne_count += valid_flat & airborne

        rear_excursion = self._rear_upper_joint_excursion_raw(box_mask)
        self.rear_excursion_raw_sum += rear_excursion
        self.rear_excursion_raw_max[:] = torch.maximum(
            self.rear_excursion_raw_max, rear_excursion
        )

    def _rear_upper_joint_excursion_raw(self, box_mask):
        """Return the mean squared excess of the four rear upper joints."""
        offsets = torch.abs(
            self.dof_pos[:, self.rear_upper_joint_indices]
            - self.default_dof_pos[:, self.rear_upper_joint_indices]
        )
        flat_allowance = torch.tensor(
            [
                self.cfg.rewards.flat_hip_allowance,
                self.cfg.rewards.flat_hip_allowance,
                self.cfg.rewards.flat_thigh_allowance,
                self.cfg.rewards.flat_thigh_allowance,
            ],
            dtype=offsets.dtype,
            device=offsets.device,
        )
        box_allowance = torch.tensor(
            [
                self.cfg.rewards.box_hip_allowance,
                self.cfg.rewards.box_hip_allowance,
                self.cfg.rewards.box_thigh_allowance,
                self.cfg.rewards.box_thigh_allowance,
            ],
            dtype=offsets.dtype,
            device=offsets.device,
        )
        allowance = torch.where(
            box_mask.unsqueeze(1), box_allowance, flat_allowance
        )
        return torch.square(torch.relu(offsets - allowance)).mean(
            dim=1
        ).clamp(max=1.0)

    def _base_height_above_terrain(self):
        """Return base height above the surface directly below the base."""
        terrain_height = self.terrain.get_terrain_heights(
            self.root_states[:, :3]
        )
        return self.root_states[:, 2] - terrain_height

    def _filtered_foot_contacts(self):
        """Filter one-frame mesh contact dropouts for gait statistics."""
        threshold = self.cfg.box_progress.contact_force_threshold
        current = (
            torch.norm(
                self.contact_forces[:, self.feet_indices, :], dim=-1
            )
            > threshold
        )
        previous = (
            torch.norm(
                self.last_contact_forces[:, self.feet_indices, :], dim=-1
            )
            > threshold
        )
        return current | previous

    def _update_flat_gait_statistics(self, flat_mask):
        """Track low-body posture, support duty, and flat-ground cadence."""
        base_height = self._base_height_above_terrain()
        valid_flat = flat_mask & torch.isfinite(base_height)
        safe_height = torch.where(
            valid_flat, base_height, torch.zeros_like(base_height)
        )
        self.flat_base_height_sum += safe_height
        self.flat_base_height_count += valid_flat
        self.min_flat_base_height[:] = torch.where(
            valid_flat,
            torch.minimum(self.min_flat_base_height, base_height),
            self.min_flat_base_height,
        )

        low_height = valid_flat & (
            base_height
            < self.cfg.box_progress.flat_low_base_height_threshold
        )
        self.flat_low_base_height_counter[:] = torch.where(
            low_height,
            self.flat_low_base_height_counter + 1,
            torch.zeros_like(self.flat_low_base_height_counter),
        )
        self.max_flat_low_base_height_steps[:] = torch.maximum(
            self.max_flat_low_base_height_steps,
            self.flat_low_base_height_counter,
        )
        self.flat_low_base_height_count += low_height
        self.flat_low_base_height_failure_buf[:] = (
            self.flat_low_base_height_counter
            >= int(self.cfg.box_progress.flat_low_base_height_steps)
        )

        foot_contact = self._filtered_foot_contacts()
        self.flat_foot_contact_count += (
            foot_contact & valid_flat.unsqueeze(1)
        )
        transition_valid = (
            valid_flat & self.previous_flat_contact_valid
        ).unsqueeze(1)
        self.flat_foot_contact_transition_count += (
            (foot_contact != self.previous_flat_foot_contact)
            & transition_valid
        )
        self.previous_flat_foot_contact[:] = foot_contact
        self.previous_flat_contact_valid[:] = valid_flat

    def _update_motion_quality_statistics(self):
        """Accumulate posture and action-change metrics for each episode."""
        roll, pitch, _ = get_euler_xyz(self.base_quat)
        roll = torch.where(roll > np.pi, roll - 2.0 * np.pi, roll).abs()
        pitch = torch.where(pitch > np.pi, pitch - 2.0 * np.pi, pitch).abs()
        action_rate_l2 = torch.norm(
            self.actions - self.last_actions, dim=1
        )
        action_rate_l2 *= (self.episode_length_buf > 1).float()

        self.abs_roll_sum += roll
        self.max_abs_roll = torch.maximum(self.max_abs_roll, roll)
        self.abs_pitch_sum += pitch
        self.max_abs_pitch = torch.maximum(self.max_abs_pitch, pitch)
        self.action_rate_l2_sum += action_rate_l2
        self.max_action_rate_l2 = torch.maximum(
            self.max_action_rate_l2, action_rate_l2
        )

        named_error = torch.abs(
            self.dof_pos[:, self.dof_error_named_indices]
            - self.default_dof_pos[:, self.dof_error_named_indices]
        )
        named_error_mean = named_error.mean(dim=1)
        self.named_dof_error_sum += named_error_mean
        self.max_named_dof_error = torch.maximum(
            self.max_named_dof_error, named_error.max(dim=1).values
        )
        self.named_joint_error_sum += named_error
        self.named_joint_error_max = torch.maximum(
            self.named_joint_error_max, named_error
        )

        dof_range = (self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0])
        near_margin = (
            dof_range * self.cfg.rewards.dof_near_limit_fraction
        )
        near_limit = (
            (self.dof_pos <= self.dof_pos_limits[:, 0] + near_margin)
            | (self.dof_pos >= self.dof_pos_limits[:, 1] - near_margin)
        )
        self.dof_near_limit_count += near_limit.sum(dim=1)
        self.action_saturation_count += (
            torch.abs(self.actions)
            > self.cfg.rewards.action_saturation_threshold
        ).sum(dim=1)
        self.action_abs_sum += torch.abs(self.actions).sum(dim=1)

        body_states = self.all_rigid_body_states.view(self.num_envs, -1, 13)
        feet_world_offset = (
            body_states[:, self.feet_indices, :3]
            - self.root_states[:, None, :3]
        )
        feet_local = quat_rotate_inverse(
            self.base_quat[:, None, :].expand(-1, len(self.feet_indices), -1)
            .reshape(-1, 4),
            feet_world_offset.reshape(-1, 3),
        ).reshape(self.num_envs, len(self.feet_indices), 3)
        left_y = feet_local[:, self.left_foot_local_indices, 1]
        right_y = feet_local[:, self.right_foot_local_indices, 1]
        self.left_foot_crossing_count += (left_y < 0.0).sum(dim=1)
        self.right_foot_crossing_count += (right_y > 0.0).sum(dim=1)
        pairwise_lateral_distance = torch.abs(
            left_y.unsqueeze(2) - right_y.unsqueeze(1)
        )
        self.min_lr_foot_lateral_distance = torch.minimum(
            self.min_lr_foot_lateral_distance,
            pairwise_lateral_distance.flatten(1).min(dim=1).values,
        )

        contact_norm = torch.norm(self.contact_forces, dim=-1)
        leg_threshold = self.cfg.rewards.leg_contact_force_threshold
        body_threshold = self.cfg.rewards.body_collision_force_threshold
        self.thigh_collision_count += torch.any(
            contact_norm[:, self.thigh_contact_indices] > leg_threshold,
            dim=1,
        )
        self.calf_collision_count += torch.any(
            contact_norm[:, self.calf_contact_indices] > leg_threshold,
            dim=1,
        )
        self.body_collision_count += torch.any(
            contact_norm[:, self.base_contact_indices] > body_threshold,
            dim=1,
        )

    def _update_speed_statistics(
        self, forward_speed, box_mask, landing_mask=None
    ):
        """Accumulate finite whole-course and mutually exclusive region stats."""
        if landing_mask is None:
            landing_mask = torch.zeros_like(box_mask)
        flat_mask = ~box_mask & ~landing_mask
        self.forward_speed_sum += forward_speed
        self.max_forward_speed = torch.maximum(
            self.max_forward_speed, forward_speed
        )
        self.flat_forward_speed_sum += torch.where(
            flat_mask, forward_speed, torch.zeros_like(forward_speed)
        )
        self.flat_forward_speed_count += flat_mask
        self.flat_forward_speed_max = torch.where(
            flat_mask,
            torch.maximum(self.flat_forward_speed_max, forward_speed),
            self.flat_forward_speed_max,
        )
        self.box_forward_speed_sum += torch.where(
            box_mask, forward_speed, torch.zeros_like(forward_speed)
        )
        self.box_forward_speed_count += box_mask
        self.box_forward_speed_max = torch.where(
            box_mask,
            torch.maximum(self.box_forward_speed_max, forward_speed),
            self.box_forward_speed_max,
        )
        rewards_cfg = self.cfg.rewards
        self.flat_overspeed_count += flat_mask & (
            forward_speed > rewards_cfg.flat_speed_limit
        )
        self.flat_severe_overspeed_count += flat_mask & (
            forward_speed > rewards_cfg.flat_severe_speed_limit
        )
        self.box_overspeed_count += box_mask & (
            forward_speed > rewards_cfg.box_speed_limit
        )

    def _get_speed_statistics(self, env_ids):
        """Summarize completed episodes without zero-count divisions."""
        episode_lengths = self.episode_length_buf[env_ids].clamp_min(1)
        flat_count = self.flat_forward_speed_count[env_ids].sum()
        box_count = self.box_forward_speed_count[env_ids].sum()
        return {
            "mean_forward_speed_mps": torch.mean(
                self.forward_speed_sum[env_ids] / episode_lengths
            ),
            "max_forward_speed_mps": torch.max(
                self.max_forward_speed[env_ids]
            ),
            "flat_forward_speed_mean_mps": (
                self.flat_forward_speed_sum[env_ids].sum()
                / flat_count.clamp_min(1)
            ),
            "flat_forward_speed_max_mps": torch.max(
                self.flat_forward_speed_max[env_ids]
            ),
            "flat_speed_sample_count": (
                self.flat_forward_speed_count[env_ids].float().mean()
            ),
            "flat_overspeed_ratio": (
                self.flat_overspeed_count[env_ids].sum()
                / flat_count.clamp_min(1)
            ),
            "flat_severe_overspeed_ratio": (
                self.flat_severe_overspeed_count[env_ids].sum()
                / flat_count.clamp_min(1)
            ),
            "box_forward_speed_mean_mps": (
                self.box_forward_speed_sum[env_ids].sum()
                / box_count.clamp_min(1)
            ),
            "box_forward_speed_max_mps": torch.max(
                self.box_forward_speed_max[env_ids]
            ),
            "box_speed_sample_count": (
                self.box_forward_speed_count[env_ids].float().mean()
            ),
            "box_overspeed_ratio": (
                self.box_overspeed_count[env_ids].sum()
                / box_count.clamp_min(1)
            ),
        }

    def _reset_speed_statistics(self, env_ids):
        """Clear speed accumulators for selected environments."""
        self.forward_speed_sum[env_ids] = 0.0
        self.max_forward_speed[env_ids] = 0.0
        self.flat_forward_speed_sum[env_ids] = 0.0
        self.flat_forward_speed_count[env_ids] = 0
        self.flat_forward_speed_max[env_ids] = 0.0
        self.box_forward_speed_sum[env_ids] = 0.0
        self.box_forward_speed_count[env_ids] = 0
        self.box_forward_speed_max[env_ids] = 0.0
        self.flat_overspeed_count[env_ids] = 0
        self.flat_severe_overspeed_count[env_ids] = 0
        self.box_overspeed_count[env_ids] = 0

    def _get_motion_quality_statistics(self, env_ids):
        """Summarize posture, action changes, and early task failures."""
        episode_lengths = self.episode_length_buf[env_ids].clamp_min(1)
        failure_mask = self.box_progress.failure_buf[env_ids]
        failure_count = failure_mask.sum()
        early_failure_steps = int(np.ceil(1.0 / self.dt))
        early_failure = failure_mask & (
            self.episode_length_buf[env_ids] <= early_failure_steps
        )
        failure_steps = torch.where(
            failure_mask,
            self.episode_length_buf[env_ids],
            torch.zeros_like(self.episode_length_buf[env_ids]),
        )
        named_episode_mean = (
            self.named_dof_error_sum[env_ids] / episode_lengths
        )
        action_rate_episode_mean = (
            self.action_rate_l2_sum[env_ids] / episode_lengths
        )
        named_joint_episode_mean = (
            self.named_joint_error_sum[env_ids]
            / episode_lengths.unsqueeze(1)
        )
        step_count = episode_lengths.sum().clamp_min(1)
        dof_sample_count = (step_count * self.num_dof).clamp_min(1)
        action_sample_count = (step_count * self.num_actions).clamp_min(1)
        left_sample_count = (
            step_count * len(self.left_foot_local_indices)
        ).clamp_min(1)
        right_sample_count = (
            step_count * len(self.right_foot_local_indices)
        ).clamp_min(1)
        stats = {
            "mean_abs_roll_rad": torch.mean(
                self.abs_roll_sum[env_ids] / episode_lengths
            ),
            "max_abs_roll_rad": torch.max(self.max_abs_roll[env_ids]),
            "mean_abs_pitch_rad": torch.mean(
                self.abs_pitch_sum[env_ids] / episode_lengths
            ),
            "max_abs_pitch_rad": torch.max(self.max_abs_pitch[env_ids]),
            "mean_action_rate_l2": torch.mean(
                self.action_rate_l2_sum[env_ids] / episode_lengths
            ),
            "max_action_rate_l2": torch.max(
                self.max_action_rate_l2[env_ids]
            ),
            # P95 values are taken across completed episodes after each
            # episode's time average is computed. This avoids retaining every
            # control-step sample for all 4096 environments.
            "p95_action_rate_l2": torch.quantile(
                action_rate_episode_mean.float(), 0.95
            ),
            "mean_named_dof_error": named_episode_mean.mean(),
            "p95_named_dof_error": torch.quantile(
                named_episode_mean.float(), 0.95
            ),
            "max_named_dof_error": torch.max(
                self.max_named_dof_error[env_ids]
            ),
            "dof_near_limit_ratio": (
                self.dof_near_limit_count[env_ids].sum()
                / dof_sample_count
            ),
            "action_saturation_ratio": (
                self.action_saturation_count[env_ids].sum()
                / action_sample_count
            ),
            "mean_action_abs": (
                self.action_abs_sum[env_ids].sum()
                / action_sample_count
            ),
            "left_foot_crossing_ratio": (
                self.left_foot_crossing_count[env_ids].sum()
                / left_sample_count
            ),
            "right_foot_crossing_ratio": (
                self.right_foot_crossing_count[env_ids].sum()
                / right_sample_count
            ),
            "leg_crossing_ratio": (
                self.left_foot_crossing_count[env_ids].sum()
                + self.right_foot_crossing_count[env_ids].sum()
            ) / (left_sample_count + right_sample_count),
            "min_left_right_foot_lateral_distance_m": torch.min(
                self.min_lr_foot_lateral_distance[env_ids]
            ),
            "thigh_collision_rate": (
                self.thigh_collision_count[env_ids].sum() / step_count
            ),
            "calf_collision_rate": (
                self.calf_collision_count[env_ids].sum() / step_count
            ),
            "body_collision_rate": (
                self.body_collision_count[env_ids].sum() / step_count
            ),
            "body_contact_ratio": (
                self.box_progress.body_contact_step_count[env_ids].sum()
                / step_count
            ),
            "current_ground_contact_feet_mean": (
                self.current_ground_contact_feet_sum[env_ids].sum()
                / step_count
            ),
            "current_rear_support_ratio": (
                self.current_rear_support_count[env_ids].sum()
                / step_count
            ),
            "rear_support_window_ratio": self.rear_support_window_ratio[
                env_ids
            ].mean(),
            "flat_airborne_ratio": (
                self.flat_airborne_count[env_ids].sum()
                / self.flat_base_height_count[env_ids].sum().clamp_min(1)
            ),
            "rear_support_missing_rate": (
                self.rear_support_missing_count[env_ids].sum() / step_count
            ),
            "max_rear_support_missing_steps": torch.max(
                self.max_rear_support_missing_steps[env_ids]
            ).float(),
            "max_body_contact_window_steps": torch.max(
                self.box_progress.max_body_contact_window_count[env_ids]
            ).float(),
            "early_failure_rate": early_failure.float().mean(),
            "mean_failure_time_s": (
                failure_steps.float().sum()
                * self.dt
                / failure_count.clamp_min(1)
            ),
        }
        for joint_idx, dof_idx in enumerate(
            self.dof_error_named_indices.tolist()
        ):
            joint_name = self.dof_names[dof_idx]
            joint_prefix = f"{joint_name}_abs_error_rad"
            stats[f"mean_{joint_prefix}"] = named_joint_episode_mean[
                :, joint_idx
            ].mean()
            stats[f"p95_{joint_prefix}"] = torch.quantile(
                named_joint_episode_mean[:, joint_idx].float(), 0.95
            )
            stats[f"max_{joint_prefix}"] = torch.max(
                self.named_joint_error_max[env_ids, joint_idx]
            )
        return stats

    def _get_flat_gait_statistics(self, env_ids):
        """Summarize flat-ground body height, support duty, and cadence."""
        flat_count = self.flat_base_height_count[env_ids].sum()
        denominator = flat_count.clamp_min(1)
        valid_envs = self.flat_base_height_count[env_ids] > 0
        fallback_height = torch.full_like(
            self.min_flat_base_height[env_ids],
            10.0,
        )
        valid_minima = torch.where(
            valid_envs,
            self.min_flat_base_height[env_ids],
            fallback_height,
        )
        minimum_height = torch.where(
            valid_envs.any(),
            valid_minima.min(),
            torch.zeros((), device=self.device),
        )
        contact_counts = self.flat_foot_contact_count[env_ids].sum(dim=0)
        transition_count = self.flat_foot_contact_transition_count[
            env_ids
        ].sum()
        num_feet = len(self.feet_indices)
        stats = {
            "flat_base_height_mean_m": (
                self.flat_base_height_sum[env_ids].sum() / denominator
            ),
            "flat_base_height_min_m": minimum_height,
            "flat_low_base_height_ratio": (
                self.flat_low_base_height_count[env_ids].sum()
                / denominator
            ),
            "flat_low_base_height_failure_rate": (
                self.flat_low_base_height_failure_buf[env_ids]
                .float()
                .mean()
            ),
            "max_flat_low_base_height_steps": torch.max(
                self.max_flat_low_base_height_steps[env_ids]
            ).float(),
            "flat_mean_support_feet": contact_counts.sum() / denominator,
            # A complete stance cycle has approximately one contact-on and
            # one contact-off transition per foot.
            "flat_step_frequency_hz": (
                transition_count
                / (2.0 * num_feet * denominator * self.dt)
            ),
        }
        front_count = contact_counts[self.front_foot_local_indices].sum()
        rear_count = contact_counts[self.rear_foot_local_indices].sum()
        stats["flat_front_contact_duty_ratio"] = (
            front_count
            / (denominator * len(self.front_foot_local_indices))
        )
        stats["flat_rear_contact_duty_ratio"] = (
            rear_count
            / (denominator * len(self.rear_foot_local_indices))
        )
        for foot_idx, foot_name in enumerate(self.feet_names):
            stats[f"flat_{foot_name}_contact_duty_ratio"] = (
                contact_counts[foot_idx] / denominator
            )
        return stats

    def _get_rear_upper_joint_statistics(self, env_ids):
        """Summarize rear hip/thigh swing without affecting training."""
        stats = {}
        for region in ("flat", "box"):
            sample_count = getattr(
                self, f"rear_upper_{region}_sample_count"
            )[env_ids]
            action_sample_count = getattr(
                self, f"rear_upper_{region}_action_sample_count"
            )[env_ids]
            position_min = getattr(
                self, f"rear_upper_{region}_pos_min"
            )[env_ids]
            position_max = getattr(
                self, f"rear_upper_{region}_pos_max"
            )[env_ids]
            velocity_sum = getattr(
                self, f"rear_upper_{region}_vel_sq_sum"
            )[env_ids]
            action_delta_sum = getattr(
                self, f"rear_upper_{region}_action_delta_sq_sum"
            )[env_ids]

            valid_episode = sample_count > 0
            valid_count = valid_episode.sum().clamp_min(1)
            episode_range = torch.where(
                valid_episode.unsqueeze(1),
                (position_max - position_min).clamp_min(0.0),
                torch.zeros_like(position_min),
            )
            total_samples = sample_count.sum().clamp_min(1)
            total_action_samples = action_sample_count.sum().clamp_min(1)
            stats[f"{region}_rear_upper_joint_sample_count"] = (
                sample_count.sum().float()
            )
            stats[f"{region}_rear_upper_joint_action_sample_count"] = (
                action_sample_count.sum().float()
            )
            for joint_idx, joint_name in enumerate(
                self.rear_upper_joint_names
            ):
                prefix = f"{region}_{joint_name}"
                stats[f"{prefix}_mean_range_rad"] = (
                    episode_range[:, joint_idx].sum() / valid_count
                )
                stats[f"{prefix}_max_range_rad"] = torch.max(
                    episode_range[:, joint_idx]
                )
                stats[f"{prefix}_velocity_rms_rad_s"] = torch.sqrt(
                    velocity_sum[:, joint_idx].sum() / total_samples
                )
                stats[f"{prefix}_action_delta_rms"] = torch.sqrt(
                    action_delta_sum[:, joint_idx].sum()
                    / total_action_samples
                )
        episode_lengths = self.episode_length_buf[env_ids].clamp_min(1)
        episode_excursion_mean = (
            self.rear_excursion_raw_sum[env_ids] / episode_lengths
        )
        stats["rear_upper_joint_excursion_raw_mean"] = (
            episode_excursion_mean.mean()
        )
        stats["rear_upper_joint_excursion_raw_p95"] = torch.quantile(
            episode_excursion_mean.float(), 0.95
        )
        stats["rear_upper_joint_excursion_raw_max"] = torch.max(
            self.rear_excursion_raw_max[env_ids]
        )
        return stats

    def _get_landing_statistics(self, env_ids):
        """Summarize landing-phase progress and individual pass conditions."""
        tracker = self.box_progress
        landing_steps = tracker.landing_phase_step_count[env_ids].sum()
        denominator = landing_steps.clamp_min(1)
        stats = {
            "landing_phase_entry_rate": (
                (tracker.landing_phase_start_step[env_ids] >= 0)
                .float()
                .mean()
            ),
            "landing_score_mean": (
                tracker.landing_score_sum[env_ids].sum() / denominator
            ),
            "landing_score_max": torch.max(
                tracker.best_landing_score[env_ids]
            ),
            "best_landing_score_mean": tracker.best_landing_score[
                env_ids
            ].mean(),
            "max_consecutive_valid_landing_steps": torch.max(
                tracker.max_consecutive_valid_landing_steps[env_ids]
            ).float(),
            "mean_best_landing_hold_steps": tracker.best_landing_hold_steps[
                env_ids
            ].float().mean(),
        }
        for name in (
            "landing_zone",
            "landing_forward_speed",
            "landing_vertical_speed",
            "landing_roll",
            "landing_pitch",
            "landing_height",
            "landing_two_feet",
            "landing_rear_support",
            "landing_no_body_contact",
        ):
            stats[f"{name}_pass_rate"] = (
                getattr(tracker, f"{name}_pass_count")[env_ids].sum()
                / denominator
            )
        return stats

    def _reset_motion_quality_statistics(self, env_ids):
        """Clear posture and action-change accumulators."""
        self.abs_roll_sum[env_ids] = 0.0
        self.max_abs_roll[env_ids] = 0.0
        self.abs_pitch_sum[env_ids] = 0.0
        self.max_abs_pitch[env_ids] = 0.0
        self.action_rate_l2_sum[env_ids] = 0.0
        self.max_action_rate_l2[env_ids] = 0.0
        self.named_dof_error_sum[env_ids] = 0.0
        self.max_named_dof_error[env_ids] = 0.0
        self.named_joint_error_sum[env_ids] = 0.0
        self.named_joint_error_max[env_ids] = 0.0
        self.dof_near_limit_count[env_ids] = 0
        self.action_saturation_count[env_ids] = 0
        self.action_abs_sum[env_ids] = 0.0
        self.left_foot_crossing_count[env_ids] = 0
        self.right_foot_crossing_count[env_ids] = 0
        self.min_lr_foot_lateral_distance[env_ids] = 10.0
        self.thigh_collision_count[env_ids] = 0
        self.calf_collision_count[env_ids] = 0
        self.body_collision_count[env_ids] = 0
        self.rear_support_history[env_ids] = False
        self.airborne_history[env_ids] = False
        self.rear_support_valid_history[env_ids] = False
        self.rear_support_window_ratio[env_ids] = 0.0
        self.flat_airborne_window_ratio[env_ids] = 0.0
        self.rear_support_missing_count[env_ids] = 0
        self.max_rear_support_missing_steps[env_ids] = 0
        self.current_ground_contact_feet_sum[env_ids] = 0.0
        self.current_rear_support_count[env_ids] = 0
        self.flat_airborne_count[env_ids] = 0
        self.rear_excursion_raw_sum[env_ids] = 0.0
        self.rear_excursion_raw_max[env_ids] = 0.0

    def _reset_flat_gait_statistics(self, env_ids):
        """Clear flat-ground posture and contact statistics."""
        self.flat_base_height_sum[env_ids] = 0.0
        self.flat_base_height_count[env_ids] = 0
        self.min_flat_base_height[env_ids] = 10.0
        self.flat_low_base_height_count[env_ids] = 0
        self.flat_low_base_height_counter[env_ids] = 0
        self.max_flat_low_base_height_steps[env_ids] = 0
        self.flat_low_base_height_failure_buf[env_ids] = False
        self.flat_foot_contact_count[env_ids] = 0
        self.flat_foot_contact_transition_count[env_ids] = 0
        self.previous_flat_foot_contact[env_ids] = False
        self.previous_flat_contact_valid[env_ids] = False

    def _reset_rear_upper_joint_statistics(self, env_ids):
        """Clear rear hip/thigh swing statistics for selected environments."""
        for region in ("flat", "box"):
            getattr(self, f"rear_upper_{region}_sample_count")[env_ids] = 0
            getattr(
                self, f"rear_upper_{region}_action_sample_count"
            )[env_ids] = 0
            getattr(self, f"rear_upper_{region}_pos_min")[env_ids] = 10.0
            getattr(self, f"rear_upper_{region}_pos_max")[env_ids] = -10.0
            getattr(
                self, f"rear_upper_{region}_vel_sq_sum"
            )[env_ids] = 0.0
            getattr(
                self, f"rear_upper_{region}_action_delta_sq_sum"
            )[env_ids] = 0.0

    def check_termination(self):
        super().check_termination()
        # The base class uses ``>`` for historical truncation semantics. This
        # task has a real 45 s completion deadline, so terminate at the exact
        # configured control step and classify it as incomplete below.
        natural_timeout = (
            self.time_out_buf
            | (self.episode_length_buf >= self.max_episode_length)
        )

        roll, pitch, _ = get_euler_xyz(self.base_quat)
        roll = torch.where(roll > np.pi, roll - 2.0 * np.pi, roll)
        pitch = torch.where(pitch > np.pi, pitch - 2.0 * np.pi, pitch)
        body_states = self.all_rigid_body_states.view(self.num_envs, -1, 13)
        feet_positions = body_states[:, self.feet_indices, :3]
        feet_contact_forces = self.contact_forces[:, self.feet_indices, :]
        feet_terrain_heights = self.terrain.get_terrain_heights(feet_positions)
        body_contact_forces = torch.norm(
            self.contact_forces[:, self.base_contact_indices, :], dim=-1
        )
        body_contact_force = body_contact_forces.max(dim=1).values
        body_contact = (
            body_contact_force
            > self.cfg.box_progress.contact_force_threshold
        )

        self.box_progress.update(
            box_bounds=self.env_box_bounds,
            env_origins=self.env_origins,
            track_start_x=self.track_start_x,
            track_end_x=self.track_end_x,
            feet_positions=feet_positions,
            feet_terrain_heights=feet_terrain_heights,
            feet_contact_forces=feet_contact_forces,
            base_positions=self.root_states[:, :3],
            roll=roll,
            pitch=pitch,
            body_contact=body_contact,
            body_contact_force=body_contact_force,
            base_vertical_velocity=self.root_states[:, 9],
            base_forward_velocity=self.base_lin_vel[:, 0],
            natural_timeout=natural_timeout,
            episode_step=self.episode_length_buf,
            landing_end_x=self.course_landing_end_x,
            external_fall=self.flat_low_base_height_failure_buf,
        )
        self._apply_landing_commands()
        self._update_course_progress_reward()
        self.box_progress.apply_termination(
            self.reset_buf, self.time_out_buf
        )

    def _apply_landing_commands(self):
        """Command a full stop immediately after the required final box."""
        landing_phase = (
            self.next_box_idx >= self.box_progress.required_boxes
        )
        self.commands[landing_phase, :3] = 0.0

    def _fill_extras(self, env_ids):
        raw_reward_sums = {
            name: values[env_ids].clone()
            for name, values in self.episode_sums.items()
        }
        raw_total_return = torch.zeros(
            len(env_ids), dtype=torch.float, device=self.device
        )
        for values in raw_reward_sums.values():
            raw_total_return += values
        super()._fill_extras(env_ids)
        episode = self.extras["episode"]
        episode["success_rate"] = self.success_buf[env_ids].float().mean()
        episode["episode_incomplete_rate"] = (
            self.incomplete_buf[env_ids].float().mean()
        )
        episode["episode_timeout_rate"] = (
            self.episode_timeout_buf[env_ids].float().mean()
        )
        episode["mean_passed_boxes"] = (
            self.passed_box_count[env_ids].float().mean()
        )
        episode["missed_box_rate"] = (
            self.missed_box_buf[env_ids].float().mean()
        )
        episode["out_of_track_rate"] = (
            self.out_of_track_buf[env_ids].float().mean()
        )
        episode["landing_overrun_rate"] = (
            self.landing_overrun_buf[env_ids].float().mean()
        )
        episode["landing_timeout_rate"] = (
            self.landing_timeout_buf[env_ids].float().mean()
        )
        episode["lateral_out_of_track_rate"] = (
            self.lateral_out_of_track_buf[env_ids].float().mean()
        )
        episode["backward_out_of_track_rate"] = (
            self.backward_out_of_track_buf[env_ids].float().mean()
        )
        episode["fall_rate"] = self.fall_buf[env_ids].float().mean()
        episode["body_contact_window_failure_rate"] = (
            self.body_contact_window_failure_buf[env_ids].float().mean()
        )
        episode["severe_body_impact_rate"] = (
            self.severe_body_impact_buf[env_ids].float().mean()
        )
        episode["required_boxes"] = torch.tensor(
            float(self.box_progress.required_boxes), device=self.device
        )
        episode["mean_progress_ratio"] = (
            self.passed_box_count[env_ids].float().mean()
            / self.box_progress.required_boxes
        )
        episode["rewarded_progress_ratio"] = self.rewarded_progress_ratio[
            env_ids
        ].mean()
        episode.update(self._get_speed_statistics(env_ids))
        episode.update(self._get_motion_quality_statistics(env_ids))
        episode.update(self._get_flat_gait_statistics(env_ids))
        episode.update(self._get_rear_upper_joint_statistics(env_ids))
        episode.update(self._get_landing_statistics(env_ids))
        for name, values in raw_reward_sums.items():
            episode[f"raw/rew_{name}"] = values.mean()
        collision_parts = (
            "body_collision",
            "thigh_collision",
            "calf_collision",
        )
        if all(name in raw_reward_sums for name in collision_parts):
            episode["raw/rew_collision"] = sum(
                raw_reward_sums[name] for name in collision_parts
            ).mean()
        episode["raw/total_return"] = raw_total_return.mean()

        result_masks = {
            "success": self.success_buf[env_ids],
            "fall_failure": self.fall_buf[env_ids],
            "missed_box_failure": self.missed_box_buf[env_ids],
            "out_of_track_failure": (
                self.out_of_track_buf[env_ids]
            ),
            "landing_overrun": self.landing_overrun_buf[env_ids],
            "landing_timeout": self.landing_timeout_buf[env_ids],
            "incomplete": self.incomplete_buf[env_ids],
            "early_failure": self.box_progress.failure_buf[env_ids]
            & (
                self.episode_length_buf[env_ids]
                <= int(np.ceil(1.0 / self.dt))
            ),
            "late_failure": self.generic_failure_buf[env_ids]
            & (self.task_progress_buf[env_ids] >= 0.75),
        }
        for result_name, result_mask in result_masks.items():
            result_count = result_mask.sum()
            result_total = torch.where(
                result_mask,
                raw_total_return,
                torch.zeros_like(raw_total_return),
            ).sum()
            episode[f"raw/{result_name}_mean_return"] = (
                result_total / result_count.clamp_min(1)
            )
            episode[f"raw/{result_name}_episode_count"] = (
                result_count.float()
            )
        continuous_groups = {
            "torque_penalty": ("exceed_torque_limits_l1norm",),
            "action_rate_penalty": ("action_rate",),
            "dof_vel_penalty": ("dof_vel",),
            "height_penalty": ("flat_base_height",),
            "collision_penalty": (
                "body_collision",
                "thigh_collision",
                "calf_collision",
            ),
        }
        continuous_names = (
            "energy_substeps",
            "torques",
            "dof_error_named",
            "dof_error",
            "action_rate",
            "dof_vel",
            "flat_orientation",
            "flat_base_height",
            "body_collision",
            "thigh_collision",
            "calf_collision",
            "rear_support_missing",
            "flat_airborne",
            "rear_upper_joint_excursion",
            "exceed_dof_pos_limits",
            "exceed_torque_limits_l1norm",
        )
        for result_name in ("success", "landing_overrun", "early_failure"):
            result_mask = result_masks[result_name]
            result_count = result_mask.sum().clamp_min(1)
            for metric_name, reward_names in continuous_groups.items():
                values = sum(
                    (
                        raw_reward_sums[name]
                        for name in reward_names
                        if name in raw_reward_sums
                    ),
                    torch.zeros_like(raw_total_return),
                )
                episode[f"raw/{result_name}_{metric_name}"] = (
                    torch.where(
                        result_mask,
                        values,
                        torch.zeros_like(values),
                    ).sum()
                    / result_count
                )
            continuous_values = sum(
                (
                    raw_reward_sums[name]
                    for name in continuous_names
                    if name in raw_reward_sums
                ),
                torch.zeros_like(raw_total_return),
            )
            episode[f"raw/{result_name}_continuous_penalty_total"] = (
                torch.where(
                    result_mask,
                    continuous_values,
                    torch.zeros_like(continuous_values),
                ).sum()
                / result_count
            )
        for box_idx in range(self.box_progress.required_boxes):
            episode[f"box_{box_idx + 1}_pass_rate"] = (
                (self.passed_box_count[env_ids] > box_idx).float().mean()
            )
            episode[f"box_{box_idx + 1}_rear_contact_rate"] = (
                self.box_progress.rear_contacted_boxes[
                    env_ids, box_idx
                ].float().mean()
            )
        for layout_idx in range(self.terrain.num_unique_layouts):
            layout_mask = self.layout_indices[env_ids] == layout_idx
            layout_count = layout_mask.sum()
            layout_successes = (
                self.success_buf[env_ids] & layout_mask
            ).float().sum()
            episode[f"layout_{layout_idx + 1}_success_rate"] = (
                layout_successes / layout_count.clamp_min(1)
            )
            episode[f"layout_{layout_idx + 1}_episode_count"] = (
                layout_count.float()
            )

        self.extras["successes"] = self.success_buf.clone()
        self.extras["episode_incompletes"] = self.incomplete_buf.clone()
        self.extras["episode_timeouts"] = self.episode_timeout_buf.clone()

    def _reset_buffers(self, env_ids):
        super()._reset_buffers(env_ids)
        if hasattr(self, "box_progress"):
            self.box_progress.reset(env_ids)
        if hasattr(self, "forward_speed_sum"):
            self._reset_speed_statistics(env_ids)
        if hasattr(self, "abs_roll_sum"):
            self._reset_motion_quality_statistics(env_ids)
        if hasattr(self, "flat_base_height_sum"):
            self._reset_flat_gait_statistics(env_ids)
        if hasattr(self, "rear_upper_flat_sample_count"):
            self._reset_rear_upper_joint_statistics(env_ids)
        if hasattr(self, "progress_reward_initialized"):
            self.progress_reward_start[env_ids] = 0.0
            self.rewarded_progress_ratio[env_ids] = 0.0
            self.course_progress_delta_buf[env_ids] = 0.0
            self.progress_reward_initialized[env_ids] = False

    def _near_box_for_speed_control(self):
        """Return environments with a non-zero current-box speed blend."""
        return self._box_speed_blend() > 0.0

    @staticmethod
    def _smoothstep(value):
        value = value.clamp(0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    def _box_speed_blend(self):
        """Smoothly blend flat and obstacle limits for the current box only."""
        cfg = self.cfg.rewards
        num_envs = self.root_states.shape[0]
        env_ids = torch.arange(num_envs, device=self.root_states.device)
        # Landing is a dedicated stop phase and must not inherit the final
        # obstacle's acceleration allowance.
        active = self.next_box_idx < self.box_progress.required_boxes
        target_indices = self.next_box_idx.clamp(
            max=self.box_progress.required_boxes - 1
        )
        target_bounds = self.env_box_bounds[env_ids, target_indices]
        base_x = self.root_states[:, 0]
        base_y = self.root_states[:, 1]
        ramp_up_distance = max(float(cfg.box_speed_ramp_up_distance), 1e-6)
        ramp_down_distance = max(
            float(cfg.box_speed_ramp_down_distance), 1e-6
        )
        ramp_up = self._smoothstep(
            (
                base_x
                - (target_bounds[:, 0] - ramp_up_distance)
            ) / ramp_up_distance
        )
        ramp_down = 1.0 - self._smoothstep(
            (base_x - target_bounds[:, 1]) / ramp_down_distance
        )
        near_y = (
            (base_y >= target_bounds[:, 2] - cfg.box_lateral_margin)
            & (base_y <= target_bounds[:, 3] + cfg.box_lateral_margin)
        )
        return (
            ramp_up
            * ramp_down
            * active.to(ramp_up.dtype)
            * near_y.to(ramp_up.dtype)
        ).clamp(0.0, 1.0)

    def _positive_speed_allowance(self):
        """Allow a small positive speed error only near a box."""
        return self._box_speed_blend().to(
            self.base_lin_vel.dtype
        ) * self.cfg.rewards.box_speed_allowance

    def set_quality_levels(self, speed_penalty_level, motion_quality_level):
        """Set independent speed and motion-quality curriculum levels."""
        levels = (float(speed_penalty_level), float(motion_quality_level))
        if not all(np.isfinite(level) for level in levels):
            raise ValueError("Quality curriculum levels must be finite.")
        self.speed_penalty_level = min(max(levels[0], 0.0), 1.0)
        self.motion_quality_level = min(max(levels[1], 0.0), 1.0)

    def set_quality_level(self, level):
        """Backward-compatible helper that sets both curriculum levels."""
        self.set_quality_levels(level, level)

    def _effective_motion_quality_level(self, floor_name):
        """Blend a non-zero safety floor with the motion curriculum level."""
        rewards_cfg = getattr(getattr(self, "cfg", None), "rewards", None)
        floor = float(getattr(rewards_cfg, floor_name, 0.0))
        level = float(getattr(self, "motion_quality_level", 1.0))
        return floor + (1.0 - floor) * level

    def _effective_speed_penalty_level(self, floor_name):
        """Blend an always-on speed safety floor with the speed curriculum."""
        rewards_cfg = getattr(getattr(self, "cfg", None), "rewards", None)
        floor = float(getattr(rewards_cfg, floor_name, 0.0))
        level = float(getattr(self, "speed_penalty_level", 1.0))
        return floor + (1.0 - floor) * level

    def _reward_speed_error_square(self):
        """Penalize command error while preserving a short box-speed allowance."""
        speed_error = self.base_lin_vel[:, 0] - self.commands[:, 0]
        allowance = self._positive_speed_allowance()
        adjusted_error = torch.where(
            speed_error > 0.0,
            torch.relu(speed_error - allowance),
            speed_error,
        )
        return torch.square(adjusted_error) * (
            LeggedRobotBox._effective_speed_penalty_level(
                self, "speed_error_floor"
            )
        )

    def _reward_forward_speed_tracking(self):
        """Reward forward motion near the command without rewarding waiting."""
        speed_error = self.base_lin_vel[:, 0] - self.commands[:, 0]
        tracking = torch.exp(
            -torch.square(speed_error)
            / self.cfg.rewards.forward_speed_tracking_sigma
        )
        forward_fraction = torch.clamp(
            self.base_lin_vel[:, 0]
            / self.commands[:, 0].clamp_min(1e-6),
            min=0.0,
            max=1.0,
        )
        command_gate = torch.where(
            self.commands[:, 0].abs() > 1e-4,
            forward_fraction,
            torch.ones_like(forward_fraction),
        )
        return tracking * command_gate

    def _reward_tracking_ang_vel(self):
        """Zero-centered yaw tracking; a zero yaw error gives zero."""
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2]
        )
        return (
            torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
            - 1.0
        )

    def _reward_action_rate(self):
        """Penalize action changes except across an episode reset boundary."""
        action_rate = torch.sum(
            torch.square(self.last_actions - self.actions), dim=1
        )
        return (
            action_rate
            * (self.episode_length_buf > 1).float()
            * LeggedRobotBox._effective_motion_quality_level(
                self, "action_rate_floor"
            )
        )

    def _reward_overspeed(self):
        """Quadratically penalize speed above the local absolute limit."""
        blend = self._box_speed_blend()
        speed_limit = (
            self.cfg.rewards.flat_speed_limit
            + blend
            * (
                self.cfg.rewards.box_speed_limit
                - self.cfg.rewards.flat_speed_limit
            )
        )
        excess_speed = torch.relu(self.base_lin_vel[:, 0] - speed_limit)
        return torch.square(excess_speed) * (
            LeggedRobotBox._effective_speed_penalty_level(
                self, "overspeed_floor"
            )
        )

    def _reward_course_progress(self):
        """Return the non-repeatable normalized course-progress increment."""
        return self.course_progress_delta_buf

    def _reward_termination(self):
        """Penalize ordinary failure from -40 to -25 by spatial progress."""
        progress_multiplier = 1.0 - 0.375 * self.task_progress_buf
        return self.generic_failure_buf.float() * progress_multiplier

    def _reward_incomplete(self):
        """Penalize the 45-second deadline from -40 to -30 by progress."""
        progress_multiplier = 1.0 - 0.25 * self.task_progress_buf
        return self.incomplete_buf.float() * progress_multiplier

    def _reward_landing_overrun(self):
        """Penalize crossing the landing-zone end before stabilizing."""
        return self.landing_overrun_buf.float()

    def _reward_landing_timeout(self):
        """Penalize spending two seconds in the landing phase unsuccessfully."""
        return self.landing_timeout_buf.float()

    def _reward_landing_quality_progress(self):
        """Reward only a new high-water mark in landing quality."""
        return self.landing_quality_delta

    def _reward_landing_hold_progress(self):
        """Reward only a new high-water mark in stable landing duration."""
        return self.landing_hold_delta

    def _reward_flat_orientation(self):
        """Penalize base tilt only outside the active box maneuver window."""
        tilt_square = torch.sum(
            torch.square(self.projected_gravity[:, :2]), dim=1
        )
        return (
            tilt_square
            * (1.0 - self._box_speed_blend())
            * (self.next_box_idx < self.box_progress.required_boxes).float()
            * LeggedRobotBox._effective_motion_quality_level(
                self, "flat_orientation_floor"
            )
        )

    def _reward_flat_base_height(self):
        """Penalize height only outside a support-aware neutral band."""
        base_height = self._base_height_above_terrain()
        low_error = torch.relu(
            (self.cfg.rewards.flat_height_min - base_height)
            / self.cfg.rewards.flat_height_normalization
        )
        high_error = torch.relu(
            (base_height - self.cfg.rewards.flat_height_max)
            / self.cfg.rewards.flat_height_normalization
        )
        height_penalty = torch.square(low_error) + torch.square(high_error)
        foot_contact = (
            torch.norm(
                self.contact_forces[:, self.feet_indices, :], dim=-1
            )
            > self.cfg.box_progress.contact_force_threshold
        )
        enough_feet = foot_contact.sum(dim=1) >= 2
        rear_support = foot_contact[:, self.rear_foot_local_indices].any(
            dim=1
        )
        flat_mask = self._box_speed_blend() <= 0.0
        before_landing = (
            self.next_box_idx < self.box_progress.required_boxes
        )
        finite_height = torch.isfinite(base_height)
        return (
            torch.where(
                finite_height,
                height_penalty,
                torch.zeros_like(height_penalty),
            )
            * flat_mask.float()
            * before_landing.float()
            * enough_feet.float()
            * rear_support.float()
            * LeggedRobotBox._effective_motion_quality_level(
                self, "flat_base_height_floor"
            )
        )

    def _reward_dof_error_named(self):
        return super()._reward_dof_error_named() * (
            LeggedRobotBox._effective_motion_quality_level(
                self, "dof_error_floor"
            )
        )

    def _reward_dof_error(self):
        return super()._reward_dof_error() * (
            LeggedRobotBox._effective_motion_quality_level(
                self, "dof_error_floor"
            )
        )

    def _reward_dof_vel(self):
        """Lightly suppress high-frequency joint motion at every stage."""
        return super()._reward_dof_vel() * (
            LeggedRobotBox._effective_motion_quality_level(
                self, "dof_vel_floor"
            )
        )

    def _contact_count(self, indices, threshold):
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, indices, :], dim=-1)
                > threshold
            ).float(),
            dim=1,
        )

    def _reward_body_collision(self):
        """Keep base impacts as an always-on safety penalty."""
        return self._contact_count(
            self.base_contact_indices,
            self.cfg.rewards.body_collision_force_threshold,
        ) * LeggedRobotBox._effective_motion_quality_level(
            self, "body_collision_floor"
        )

    def _reward_thigh_collision(self):
        """Gradually discourage light thigh contacts."""
        return self._contact_count(
            self.thigh_contact_indices,
            self.cfg.rewards.leg_contact_force_threshold,
        ) * LeggedRobotBox._effective_motion_quality_level(
            self, "leg_collision_floor"
        )

    def _reward_calf_collision(self):
        """Gradually discourage light calf contacts."""
        return self._contact_count(
            self.calf_contact_indices,
            self.cfg.rewards.leg_contact_force_threshold,
        ) * LeggedRobotBox._effective_motion_quality_level(
            self, "leg_collision_floor"
        )

    def _reward_rear_support_missing(self):
        """Penalize a rear-support ratio below the configured flat target."""
        target = float(self.cfg.rewards.rear_support_target_ratio)
        deficit = torch.relu(
            (target - self.rear_support_window_ratio) / max(target, 1e-6)
        )
        valid = (
            (self._box_speed_blend() <= 0.0)
            & (self.next_box_idx < self.box_progress.required_boxes)
            & (
                self.episode_length_buf
                > int(self.cfg.rewards.rear_support_window_steps)
            )
        )
        return torch.square(deficit) * valid.float()

    def _reward_flat_airborne(self):
        """Penalize excessive flat-ground flight without suppressing running."""
        free_ratio = float(self.cfg.rewards.flat_airborne_free_ratio)
        excess = torch.relu(
            (self.flat_airborne_window_ratio - free_ratio)
            / max(1.0 - free_ratio, 1e-6)
        )
        valid = (
            (self._box_speed_blend() <= 0.0)
            & (self.next_box_idx < self.box_progress.required_boxes)
            & (
                self.episode_length_buf
                > int(self.cfg.rewards.rear_support_window_steps)
            )
        )
        return torch.square(excess) * valid.float()

    def _reward_rear_upper_joint_excursion(self):
        """Return monitored rear-joint excess; Stage A gives it zero weight."""
        return self._rear_upper_joint_excursion_raw(
            self._near_box_for_speed_control()
        )

    def _reward_exceed_torque_limits_l1norm(self):
        """Average normalized torque excess across substeps and joints."""
        soft_limit = (
            self.torque_limits * self.cfg.rewards.soft_torque_limit
        ).clamp_min(1e-6)
        normalized_excess = torch.relu(
            torch.abs(self.substep_torques) - soft_limit
        ) / soft_limit
        return normalized_excess.mean(dim=(1, 2)).clamp(max=1.0)

    def _reward_lin_pos_y(self):
        """Penalize lateral displacement from the course centerline."""
        return torch.abs((self.root_states[:, :3] - self.env_origins)[:, 1])

    def _reward_yaw_abs(self):
        """Penalize absolute yaw away from the positive world x direction."""
        yaw = get_euler_xyz(self.root_states[:, 3:7])[2]
        yaw = torch.where(yaw > np.pi, yaw - 2.0 * np.pi, yaw)
        yaw = torch.where(yaw < -np.pi, yaw + 2.0 * np.pi, yaw)
        return torch.abs(yaw)

    def _reward_box_passed(self):
        """Emit one event when the current box is passed."""
        return self.box_passed_buf.float()

    def _reward_box_front_foot_contact(self):
        """Emit once after enough front-foot contact steps on the box top."""
        return self.front_foot_contact_buf.float()

    def _reward_box_rear_foot_contact(self):
        """Emit once after enough rear-foot contact steps on the box top."""
        return self.rear_foot_contact_buf.float()

    def _reward_success(self):
        """Emit one event after the required boxes and a stable landing."""
        return self.success_buf.float()
