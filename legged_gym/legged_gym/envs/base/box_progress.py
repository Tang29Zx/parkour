"""Tensor-only progress state for sequential box parkour courses."""

from numbers import Integral

import torch


class BoxProgressTracker:
    """Track ordered box contacts, course completion, and failure events."""

    def __init__(
        self,
        num_envs,
        num_feet,
        num_boxes,
        device,
        required_boxes=None,
        pass_margin=0.15,
        top_contact_tolerance=0.06,
        contact_force_threshold=1.0,
        front_foot_indices=(0, 1),
        rear_foot_indices=(2, 3),
        front_contact_required_steps=2,
        rear_contact_required_steps=2,
        landing_steps=10,
        landing_min_current_feet=2,
        landing_require_rear_foot=True,
        landing_roll_threshold=0.35,
        landing_pitch_threshold=0.45,
        landing_base_height_threshold=0.22,
        landing_vertical_speed_threshold=0.5,
        body_contact_window_steps=25,
        body_contact_failure_steps=8,
        severe_body_impact_force=80.0,
        roll_threshold=1.4,
        pitch_threshold=1.6,
        base_height_threshold=0.15,
        lateral_limit=0.8,
    ):
        front_foot_indices = tuple(int(index) for index in front_foot_indices)
        rear_foot_indices = tuple(int(index) for index in rear_foot_indices)
        all_group_indices = front_foot_indices + rear_foot_indices
        if not front_foot_indices or not rear_foot_indices:
            raise ValueError("Front and rear foot groups must both be non-empty.")
        if len(set(all_group_indices)) != len(all_group_indices):
            raise ValueError("Front and rear foot groups must be disjoint.")
        if min(all_group_indices) < 0 or max(all_group_indices) >= num_feet:
            raise ValueError("Foot group index is outside the available feet.")
        if min(front_contact_required_steps, rear_contact_required_steps) <= 0:
            raise ValueError("Box contact requirements must be positive.")
        if landing_steps <= 0:
            raise ValueError("landing_steps must be positive.")
        if not 1 <= landing_min_current_feet <= num_feet:
            raise ValueError(
                "landing_min_current_feet must be between 1 and num_feet."
            )
        if body_contact_window_steps <= 0:
            raise ValueError("body_contact_window_steps must be positive.")
        if not 1 <= body_contact_failure_steps <= body_contact_window_steps:
            raise ValueError(
                "body_contact_failure_steps must be within the contact window."
            )
        if severe_body_impact_force <= 0.0:
            raise ValueError("severe_body_impact_force must be positive.")

        self.num_envs = int(num_envs)
        self.num_feet = int(num_feet)
        self.num_boxes = int(num_boxes)
        if required_boxes is None:
            required_boxes = self.num_boxes
        if isinstance(required_boxes, bool) or not isinstance(
            required_boxes, Integral
        ):
            raise TypeError("required_boxes must be an integer.")
        required_boxes = int(required_boxes)
        if not 1 <= required_boxes <= self.num_boxes:
            raise ValueError("required_boxes must be between 1 and num_boxes.")
        self.required_boxes = required_boxes
        self.device = device
        self.pass_margin = float(pass_margin)
        self.top_contact_tolerance = float(top_contact_tolerance)
        self.contact_force_threshold = float(contact_force_threshold)
        self.front_foot_indices = torch.tensor(
            front_foot_indices, dtype=torch.long, device=device
        )
        self.rear_foot_indices = torch.tensor(
            rear_foot_indices, dtype=torch.long, device=device
        )
        self.front_contact_required_steps = int(front_contact_required_steps)
        self.rear_contact_required_steps = int(rear_contact_required_steps)
        self.landing_steps = int(landing_steps)
        self.landing_min_current_feet = int(landing_min_current_feet)
        self.landing_require_rear_foot = bool(landing_require_rear_foot)
        self.landing_roll_threshold = float(landing_roll_threshold)
        self.landing_pitch_threshold = float(landing_pitch_threshold)
        self.landing_base_height_threshold = float(
            landing_base_height_threshold
        )
        self.landing_vertical_speed_threshold = float(
            landing_vertical_speed_threshold
        )
        self.body_contact_window_steps = int(body_contact_window_steps)
        self.body_contact_failure_steps = int(body_contact_failure_steps)
        self.severe_body_impact_force = float(severe_body_impact_force)
        self.roll_threshold = float(roll_threshold)
        self.pitch_threshold = float(pitch_threshold)
        self.base_height_threshold = float(base_height_threshold)
        self.lateral_limit = float(lateral_limit)

        self.next_box_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=device
        )
        self.passed_box_count = torch.zeros_like(self.next_box_idx)
        self.foot_contact_mask = torch.zeros(
            self.num_envs, self.num_feet, dtype=torch.bool, device=device
        )
        self.front_contact_counter = torch.zeros_like(self.next_box_idx)
        self.rear_contact_counter = torch.zeros_like(self.next_box_idx)
        self.body_contact_history = torch.zeros(
            self.num_envs,
            self.body_contact_window_steps,
            dtype=torch.bool,
            device=device,
        )
        self.body_contact_history_index = 0
        self.body_contact_window_count = torch.zeros_like(self.next_box_idx)
        self.max_body_contact_window_count = torch.zeros_like(
            self.next_box_idx
        )
        self.landing_counter = torch.zeros_like(self.next_box_idx)

        self.box_passed_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self.front_foot_contact_buf = torch.zeros_like(self.box_passed_buf)
        self.rear_foot_contact_buf = torch.zeros_like(self.box_passed_buf)
        self.body_contact_window_failure_buf = torch.zeros_like(
            self.box_passed_buf
        )
        self.severe_body_impact_buf = torch.zeros_like(self.box_passed_buf)
        self.success_buf = torch.zeros_like(self.box_passed_buf)
        self.missed_box_buf = torch.zeros_like(self.box_passed_buf)
        self.out_of_track_buf = torch.zeros_like(self.box_passed_buf)
        self.landing_overrun_buf = torch.zeros_like(self.box_passed_buf)
        self.fall_buf = torch.zeros_like(self.box_passed_buf)
        self.incomplete_buf = torch.zeros_like(self.box_passed_buf)
        self.episode_timeout_buf = torch.zeros_like(self.box_passed_buf)
        self.task_progress_buf = torch.zeros(
            self.num_envs, dtype=torch.float, device=device
        )

    def reset(self, env_ids):
        """Clear all episode state for selected environments."""
        self.next_box_idx[env_ids] = 0
        self.passed_box_count[env_ids] = 0
        self.foot_contact_mask[env_ids] = False
        self.front_contact_counter[env_ids] = 0
        self.rear_contact_counter[env_ids] = 0
        self.body_contact_history[env_ids] = False
        self.body_contact_window_count[env_ids] = 0
        self.max_body_contact_window_count[env_ids] = 0
        self.landing_counter[env_ids] = 0
        self._clear_events(env_ids)

    def _clear_events(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.box_passed_buf[env_ids] = False
        self.front_foot_contact_buf[env_ids] = False
        self.rear_foot_contact_buf[env_ids] = False
        self.body_contact_window_failure_buf[env_ids] = False
        self.severe_body_impact_buf[env_ids] = False
        self.success_buf[env_ids] = False
        self.missed_box_buf[env_ids] = False
        self.out_of_track_buf[env_ids] = False
        self.landing_overrun_buf[env_ids] = False
        self.fall_buf[env_ids] = False
        self.incomplete_buf[env_ids] = False
        self.episode_timeout_buf[env_ids] = False
        self.task_progress_buf[env_ids] = 0.0

    def update(
        self,
        box_bounds,
        env_origins,
        track_start_x,
        track_end_x,
        feet_positions,
        feet_terrain_heights,
        feet_contact_forces,
        base_positions,
        roll,
        pitch,
        body_contact,
        body_contact_force,
        base_vertical_velocity,
        natural_timeout,
        landing_end_x=None,
        external_timeout=None,
    ):
        """Advance progress by one control step and update event buffers."""
        self._clear_events()
        if landing_end_x is None:
            landing_end_x = track_end_x
        env_ids = torch.arange(self.num_envs, device=self.device)
        active = self.next_box_idx < self.required_boxes
        target_indices = self.next_box_idx.clamp(max=self.required_boxes - 1)
        target_bounds = box_bounds[env_ids, target_indices]

        top_contact = self._get_top_contacts(
            active,
            target_bounds,
            feet_positions,
            feet_terrain_heights,
            feet_contact_forces,
        )
        self.foot_contact_mask |= top_contact
        front_contact = top_contact[:, self.front_foot_indices].any(dim=1)
        rear_contact = top_contact[:, self.rear_foot_indices].any(dim=1)
        previous_front_count = self.front_contact_counter.clone()
        previous_rear_count = self.rear_contact_counter.clone()
        self.front_contact_counter += active & front_contact
        self.rear_contact_counter += active & rear_contact
        self.front_contact_counter.clamp_(
            max=self.front_contact_required_steps
        )
        self.rear_contact_counter.clamp_(max=self.rear_contact_required_steps)
        self.front_foot_contact_buf[:] = (
            active
            & (previous_front_count < self.front_contact_required_steps)
            & (self.front_contact_counter >= self.front_contact_required_steps)
        )
        self.rear_foot_contact_buf[:] = (
            active
            & (previous_rear_count < self.rear_contact_required_steps)
            & (self.rear_contact_counter >= self.rear_contact_required_steps)
        )
        contact_requirements_met = (
            self.front_contact_counter >= self.front_contact_required_steps
        ) & (self.rear_contact_counter >= self.rear_contact_required_steps)
        crossed_target = active & (
            base_positions[:, 0] > target_bounds[:, 1] + self.pass_margin
        )
        self.missed_box_buf[:] = crossed_target & ~contact_requirements_met
        self.box_passed_buf[:] = crossed_target & contact_requirements_met

        passed_envs = self.box_passed_buf.nonzero(as_tuple=False).flatten()
        self.next_box_idx[passed_envs] += 1
        self.passed_box_count[passed_envs] += 1
        self.foot_contact_mask[passed_envs] = False
        self.front_contact_counter[passed_envs] = 0
        self.rear_contact_counter[passed_envs] = 0

        force_contact = feet_contact_forces.norm(dim=-1) > self.contact_force_threshold
        course_rear = box_bounds[:, self.required_boxes - 1, 1]
        post_box_ground = (
            (feet_positions[:, :, 0] > course_rear.unsqueeze(1))
            & (feet_positions[:, :, 0] < landing_end_x.unsqueeze(1))
            & (
                torch.abs(feet_positions[:, :, 1] - env_origins[:, 1].unsqueeze(1))
                <= self.lateral_limit
            )
            & (feet_terrain_heights.abs() <= 1e-6)
            & (
                torch.abs(feet_positions[:, :, 2] - feet_terrain_heights)
                <= self.top_contact_tolerance
            )
            & force_contact
        )
        course_complete = self.next_box_idx == self.required_boxes
        current_landing_feet = post_box_ground.sum(dim=1)
        rear_foot_support = post_box_ground[:, self.rear_foot_indices].any(
            dim=1
        )
        base_height = base_positions[:, 2] - env_origins[:, 2]
        stable_landing = (
            course_complete
            & (base_positions[:, 0] > course_rear)
            & (base_positions[:, 0] < landing_end_x)
            & (
                torch.abs(base_positions[:, 1] - env_origins[:, 1])
                <= self.lateral_limit
            )
            & (current_landing_feet >= self.landing_min_current_feet)
            & (
                rear_foot_support
                if self.landing_require_rear_foot
                else torch.ones_like(rear_foot_support)
            )
            & ~body_contact
            & (roll.abs() <= self.landing_roll_threshold)
            & (pitch.abs() <= self.landing_pitch_threshold)
            & (base_height >= self.landing_base_height_threshold)
            & (
                base_vertical_velocity.abs()
                <= self.landing_vertical_speed_threshold
            )
        )
        self.landing_counter[:] = torch.where(
            stable_landing,
            self.landing_counter + 1,
            torch.zeros_like(self.landing_counter),
        )
        landing_success = self.landing_counter >= self.landing_steps

        self.body_contact_history[:, self.body_contact_history_index] = (
            body_contact
        )
        self.body_contact_history_index = (
            self.body_contact_history_index + 1
        ) % self.body_contact_window_steps
        self.body_contact_window_count[:] = self.body_contact_history.sum(dim=1)
        self.max_body_contact_window_count[:] = torch.maximum(
            self.max_body_contact_window_count,
            self.body_contact_window_count,
        )
        self.body_contact_window_failure_buf[:] = (
            self.body_contact_window_count >= self.body_contact_failure_steps
        )
        self.severe_body_impact_buf[:] = (
            body_contact_force >= self.severe_body_impact_force
        )
        raw_fall = (
            (roll.abs() > self.roll_threshold)
            | (pitch.abs() > self.pitch_threshold)
            | (base_height < self.base_height_threshold)
            | self.body_contact_window_failure_buf
            | self.severe_body_impact_buf
        )

        lateral_offset = (base_positions[:, 1] - env_origins[:, 1]).abs()
        raw_out_of_track = (
            (lateral_offset > self.lateral_limit)
            | (base_positions[:, 0] < track_start_x)
            | ((base_positions[:, 0] > track_end_x) & ~landing_success)
        )
        raw_landing_overrun = torch.zeros_like(course_complete)
        if self.required_boxes < self.num_boxes:
            raw_landing_overrun = (
                course_complete
                & (base_positions[:, 0] >= landing_end_x)
                & ~landing_success
            )

        # Terminal causes are mutually exclusive. Unsafe task failures take
        # precedence over success, and natural timeout is considered only when
        # neither a failure nor success happened on the same control step.
        self.fall_buf[:] = raw_fall & ~self.missed_box_buf
        self.landing_overrun_buf[:] = (
            raw_landing_overrun & ~self.missed_box_buf & ~raw_fall
        )
        self.out_of_track_buf[:] = (
            raw_out_of_track
            & ~self.missed_box_buf
            & ~raw_fall
            & ~raw_landing_overrun
        )
        self.success_buf[:] = landing_success & ~self.failure_buf
        self.incomplete_buf[:] = (
            natural_timeout & ~self.failure_buf & ~self.success_buf
        )
        if external_timeout is None:
            external_timeout = torch.zeros_like(natural_timeout)
        self.episode_timeout_buf[:] = (
            external_timeout
            & ~self.failure_buf
            & ~self.success_buf
            & ~self.incomplete_buf
        )
        self.task_progress_buf[:] = self.compute_task_progress(
            box_bounds=box_bounds,
            track_start_x=track_start_x,
            base_x=base_positions[:, 0],
        )

    def compute_task_progress(self, box_bounds, track_start_x, base_x):
        """Return ordered course progress in ``[0, 1]`` without using time."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        completed = self.passed_box_count.clamp(max=self.required_boxes)
        course_complete = completed >= self.required_boxes
        target_indices = completed.clamp(max=self.required_boxes - 1)
        target_rear = box_bounds[env_ids, target_indices, 1] + self.pass_margin

        previous_indices = (target_indices - 1).clamp_min(0)
        previous_rear = box_bounds[env_ids, previous_indices, 1] + self.pass_margin
        segment_start = torch.where(
            target_indices > 0,
            previous_rear,
            track_start_x,
        )
        local_progress = (
            (base_x - segment_start)
            / (target_rear - segment_start).clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        progress = (
            completed.float() + local_progress
        ) / float(self.required_boxes)
        return torch.where(
            course_complete,
            torch.ones_like(progress),
            progress.clamp(0.0, 1.0),
        )

    def _get_top_contacts(
        self,
        active,
        target_bounds,
        feet_positions,
        feet_terrain_heights,
        feet_contact_forces,
    ):
        top_height = target_bounds[:, 4].unsqueeze(1)
        return (
            active.unsqueeze(1)
            & (feet_positions[:, :, 0] >= target_bounds[:, 0].unsqueeze(1))
            & (feet_positions[:, :, 0] <= target_bounds[:, 1].unsqueeze(1))
            & (feet_positions[:, :, 1] >= target_bounds[:, 2].unsqueeze(1))
            & (feet_positions[:, :, 1] <= target_bounds[:, 3].unsqueeze(1))
            & ((feet_terrain_heights - top_height).abs() <= 1e-6)
            & (
                (feet_positions[:, :, 2] - top_height).abs()
                <= self.top_contact_tolerance
            )
            & (
                feet_contact_forces.norm(dim=-1)
                > self.contact_force_threshold
            )
        )

    @property
    def failure_buf(self):
        return (
            self.missed_box_buf
            | self.out_of_track_buf
            | self.landing_overrun_buf
            | self.fall_buf
        )

    def apply_termination(self, reset_buf, time_out_buf):
        """Apply exclusive failure, success, incomplete, and timeout semantics."""
        time_out_buf.copy_(self.episode_timeout_buf)
        reset_buf |= (
            self.failure_buf
            | self.success_buf
            | self.incomplete_buf
            | time_out_buf
        )
