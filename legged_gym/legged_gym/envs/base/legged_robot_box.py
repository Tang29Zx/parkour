"""Go2 environment support for ordered random-box parkour courses."""

import numpy as np
import torch
from isaacgym.torch_utils import get_euler_xyz

from .box_progress import BoxProgressTracker
from .legged_robot import LeggedRobot


class LeggedRobotBox(LeggedRobot):
    """Add five-box progress and course termination to ``LeggedRobot``."""

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
        self.max_forward_speed = torch.full(
            (self.num_envs,), -torch.inf, dtype=torch.float, device=self.device
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
        self.fall_buf = tracker.fall_buf
        self.body_contact_counter = tracker.body_contact_counter
        self.landing_counter = tracker.landing_counter
        self.episode_timeout_buf = tracker.episode_timeout_buf

    def _refresh_box_course_data(self):
        track_indices = torch.stack(
            (self.terrain_levels, self.terrain_types), dim=1
        )
        self.env_box_bounds = self.terrain.get_box_bounds(track_indices)
        spawn_margin = float(self.terrain.track_kwargs["spawn_margin"])
        self.track_start_x = self.env_origins[:, 0] - spawn_margin
        self.track_end_x = self.track_start_x + float(self.terrain.env_length)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        forward_speed = self.base_lin_vel[:, 0]
        self.forward_speed_sum += forward_speed
        self.max_forward_speed = torch.maximum(
            self.max_forward_speed, forward_speed
        )

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
        episode["fall_rate"] = self.fall_buf[env_ids].float().mean()
        episode_lengths = self.episode_length_buf[env_ids].clamp_min(1)
        episode["mean_forward_speed_mps"] = torch.mean(
            self.forward_speed_sum[env_ids] / episode_lengths
        )
        episode["max_forward_speed_mps"] = torch.max(
            self.max_forward_speed[env_ids]
        )
        for box_idx in range(self.terrain.num_boxes):
            episode[f"box_{box_idx + 1}_pass_rate"] = (
                (self.passed_box_count[env_ids] > box_idx).float().mean()
            )

        self.extras["successes"] = self.success_buf.clone()
        self.extras["episode_timeouts"] = self.episode_timeout_buf.clone()

    def _reset_buffers(self, env_ids):
        super()._reset_buffers(env_ids)
        if hasattr(self, "box_progress"):
            self.box_progress.reset(env_ids)
        if hasattr(self, "forward_speed_sum"):
            self.forward_speed_sum[env_ids] = 0.0
            self.max_forward_speed[env_ids] = -torch.inf

    def _near_box_for_speed_control(self):
        """Return environments inside a short approach/exit window of any box."""
        cfg = self.cfg.rewards
        base_x = self.root_states[:, 0].unsqueeze(1)
        base_y = self.root_states[:, 1].unsqueeze(1)
        near_x = (
            (base_x >= self.env_box_bounds[:, :, 0] - cfg.box_approach_distance)
            & (base_x <= self.env_box_bounds[:, :, 1] + cfg.box_exit_distance)
        )
        near_y = (
            (base_y >= self.env_box_bounds[:, :, 2] - cfg.box_lateral_margin)
            & (base_y <= self.env_box_bounds[:, :, 3] + cfg.box_lateral_margin)
        )
        return torch.any(near_x & near_y, dim=1)

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

    def _reward_overspeed(self):
        """Quadratically penalize speed above the local flat/box margin."""
        near_box = self._near_box_for_speed_control()
        margin = torch.where(
            near_box,
            torch.full_like(
                self.base_lin_vel[:, 0],
                self.cfg.rewards.box_overspeed_margin,
            ),
            torch.full_like(
                self.base_lin_vel[:, 0],
                self.cfg.rewards.flat_overspeed_margin,
            ),
        )
        excess_speed = torch.relu(
            self.base_lin_vel[:, 0] - self.commands[:, 0] - margin
        )
        return torch.square(excess_speed)

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
        """Emit one event after all five boxes and a stable landing."""
        return self.success_buf.float()

    def _reward_episode_timeout(self):
        """Emit one event only for the true episode time limit."""
        return self.episode_timeout_buf.float()
