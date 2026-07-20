"""Go2 environment support for ordered random-box parkour courses."""

import numpy as np
import torch
from isaacgym.torch_utils import get_euler_xyz

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

        progress_cfg = self.cfg.box_progress
        self.box_progress = BoxProgressTracker(
            num_envs=self.num_envs,
            num_feet=len(self.feet_indices),
            num_boxes=self.terrain.num_boxes,
            device=self.device,
            required_boxes=progress_cfg.required_boxes,
            pass_margin=progress_cfg.pass_margin,
            top_contact_tolerance=progress_cfg.top_contact_tolerance,
            contact_force_threshold=progress_cfg.contact_force_threshold,
            required_distinct_feet=progress_cfg.required_distinct_feet,
            landing_steps=progress_cfg.landing_steps,
            body_contact_steps=progress_cfg.body_contact_steps,
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
        self.abs_roll_sum = torch.zeros_like(self.forward_speed_sum)
        self.max_abs_roll = torch.zeros_like(self.forward_speed_sum)
        self.abs_pitch_sum = torch.zeros_like(self.forward_speed_sum)
        self.max_abs_pitch = torch.zeros_like(self.forward_speed_sum)
        self.action_rate_l2_sum = torch.zeros_like(self.forward_speed_sum)
        self.max_action_rate_l2 = torch.zeros_like(self.forward_speed_sum)

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

    def _bind_progress_buffers(self):
        tracker = self.box_progress
        self.next_box_idx = tracker.next_box_idx
        self.passed_box_count = tracker.passed_box_count
        self.foot_contact_mask = tracker.foot_contact_mask
        self.landing_foot_contact_mask = tracker.landing_foot_contact_mask
        self.box_passed_buf = tracker.box_passed_buf
        self.first_foot_contact_buf = tracker.first_foot_contact_buf
        self.second_foot_contact_buf = tracker.second_foot_contact_buf
        self.success_buf = tracker.success_buf
        self.missed_box_buf = tracker.missed_box_buf
        self.out_of_track_buf = tracker.out_of_track_buf
        self.landing_overrun_buf = tracker.landing_overrun_buf
        self.fall_buf = tracker.fall_buf
        self.body_contact_counter = tracker.body_contact_counter
        self.landing_counter = tracker.landing_counter
        self.episode_timeout_buf = tracker.episode_timeout_buf

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
        self._update_speed_statistics(forward_speed, box_mask)
        self._update_motion_quality_statistics()

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

    def _update_speed_statistics(self, forward_speed, box_mask):
        """Accumulate finite whole-course and mutually exclusive region stats."""
        flat_mask = ~box_mask
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
        return {
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
            "early_failure_rate": early_failure.float().mean(),
            "mean_failure_time_s": (
                failure_steps.float().sum()
                * self.dt
                / failure_count.clamp_min(1)
            ),
        }

    def _reset_motion_quality_statistics(self, env_ids):
        """Clear posture and action-change accumulators."""
        self.abs_roll_sum[env_ids] = 0.0
        self.max_abs_roll[env_ids] = 0.0
        self.abs_pitch_sum[env_ids] = 0.0
        self.max_abs_pitch[env_ids] = 0.0
        self.action_rate_l2_sum[env_ids] = 0.0
        self.max_action_rate_l2[env_ids] = 0.0

    def check_termination(self):
        super().check_termination()
        natural_timeout = self.time_out_buf.clone()

        roll, pitch, _ = get_euler_xyz(self.base_quat)
        roll = torch.where(roll > np.pi, roll - 2.0 * np.pi, roll)
        pitch = torch.where(pitch > np.pi, pitch - 2.0 * np.pi, pitch)
        body_states = self.all_rigid_body_states.view(self.num_envs, -1, 13)
        feet_positions = body_states[:, self.feet_indices, :3]
        feet_contact_forces = self.contact_forces[:, self.feet_indices, :]
        feet_terrain_heights = self.terrain.get_terrain_heights(feet_positions)
        body_contact = torch.any(
            torch.norm(
                self.contact_forces[:, self.base_contact_indices, :], dim=-1
            )
            > self.cfg.box_progress.contact_force_threshold,
            dim=1,
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
            natural_timeout=natural_timeout,
            landing_end_x=self.course_landing_end_x,
        )
        self.box_progress.apply_termination(
            self.reset_buf, self.time_out_buf
        )

    def _fill_extras(self, env_ids):
        super()._fill_extras(env_ids)
        episode = self.extras["episode"]
        episode["success_rate"] = self.success_buf[env_ids].float().mean()
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
        episode["fall_rate"] = self.fall_buf[env_ids].float().mean()
        episode["required_boxes"] = torch.tensor(
            float(self.box_progress.required_boxes), device=self.device
        )
        episode["mean_progress_ratio"] = (
            self.passed_box_count[env_ids].float().mean()
            / self.box_progress.required_boxes
        )
        episode.update(self._get_speed_statistics(env_ids))
        episode.update(self._get_motion_quality_statistics(env_ids))
        for box_idx in range(self.box_progress.required_boxes):
            episode[f"box_{box_idx + 1}_pass_rate"] = (
                (self.passed_box_count[env_ids] > box_idx).float().mean()
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
        self.extras["episode_timeouts"] = self.episode_timeout_buf.clone()

    def _reset_buffers(self, env_ids):
        super()._reset_buffers(env_ids)
        if hasattr(self, "box_progress"):
            self.box_progress.reset(env_ids)
        if hasattr(self, "forward_speed_sum"):
            self._reset_speed_statistics(env_ids)
        if hasattr(self, "abs_roll_sum"):
            self._reset_motion_quality_statistics(env_ids)

    def _near_box_for_speed_control(self):
        """Return environments inside the current target box speed window."""
        cfg = self.cfg.rewards
        num_envs = self.root_states.shape[0]
        env_ids = torch.arange(num_envs, device=self.root_states.device)
        active = self.next_box_idx < self.box_progress.required_boxes
        target_indices = self.next_box_idx.clamp(
            max=self.box_progress.required_boxes - 1
        )
        target_bounds = self.env_box_bounds[env_ids, target_indices]
        base_x = self.root_states[:, 0]
        base_y = self.root_states[:, 1]
        near_x = (
            (base_x >= target_bounds[:, 0] - cfg.box_approach_distance)
            & (base_x <= target_bounds[:, 1] + cfg.box_exit_distance)
        )
        near_y = (
            (base_y >= target_bounds[:, 2] - cfg.box_lateral_margin)
            & (base_y <= target_bounds[:, 3] + cfg.box_lateral_margin)
        )
        return active & near_x & near_y

    def _positive_speed_allowance(self):
        """Allow a small positive speed error only near a box."""
        return self._near_box_for_speed_control().to(
            self.base_lin_vel.dtype
        ) * self.cfg.rewards.box_speed_allowance

    def _reward_speed_error_square(self):
        """Penalize command error while preserving a short box-speed allowance."""
        speed_error = self.base_lin_vel[:, 0] - self.commands[:, 0]
        allowance = self._positive_speed_allowance()
        adjusted_error = torch.where(
            speed_error > 0.0,
            torch.relu(speed_error - allowance),
            speed_error,
        )
        return torch.square(adjusted_error)

    def _reward_forward_speed_tracking(self):
        """Reward the commanded speed without rewarding faster motion."""
        speed_error = self.base_lin_vel[:, 0] - self.commands[:, 0]
        return torch.exp(
            -torch.square(speed_error)
            / self.cfg.rewards.forward_speed_tracking_sigma
        )

    def _reward_action_rate(self):
        """Penalize action changes except across an episode reset boundary."""
        action_rate = torch.sum(
            torch.square(self.last_actions - self.actions), dim=1
        )
        return action_rate * (self.episode_length_buf > 1).float()

    def _reward_overspeed(self):
        """Quadratically penalize speed above the local absolute limit."""
        near_box = self._near_box_for_speed_control()
        speed_limit = torch.where(
            near_box,
            torch.full_like(
                self.base_lin_vel[:, 0],
                self.cfg.rewards.box_speed_limit,
            ),
            torch.full_like(
                self.base_lin_vel[:, 0],
                self.cfg.rewards.flat_speed_limit,
            ),
        )
        excess_speed = torch.relu(self.base_lin_vel[:, 0] - speed_limit)
        return torch.square(excess_speed)

    def _reward_termination(self):
        """Make an early task failure costlier than a late failed attempt."""
        remaining_fraction = 1.0 - (
            self.episode_length_buf.float() / float(self.max_episode_length)
        )
        remaining_fraction = remaining_fraction.clamp(0.0, 1.0)
        return self.box_progress.failure_buf.float() * (
            1.0 + remaining_fraction
        )

    def _reward_flat_orientation(self):
        """Penalize base tilt only outside the active box maneuver window."""
        tilt_square = torch.sum(
            torch.square(self.projected_gravity[:, :2]), dim=1
        )
        return tilt_square * (~self._near_box_for_speed_control()).float()

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

    def _reward_box_first_foot_contact(self):
        """Emit one event when the first distinct foot reaches a box top."""
        return self.first_foot_contact_buf.float()

    def _reward_box_second_foot_contact(self):
        """Emit one event when the second distinct foot reaches a box top."""
        return self.second_foot_contact_buf.float()

    def _reward_success(self):
        """Emit one event after the required boxes and a stable landing."""
        return self.success_buf.float()

    def _reward_episode_timeout(self):
        """Emit one event only for the true episode time limit."""
        return self.episode_timeout_buf.float()
