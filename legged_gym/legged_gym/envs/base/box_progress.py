"""Tensor-only progress state for sequential box parkour courses."""

import torch


class BoxProgressTracker:
    """Track ordered box contacts, course completion, and failure events."""

    def __init__(
        self,
        num_envs,
        num_feet,
        num_boxes,
        device,
        pass_margin=0.15,
        top_contact_tolerance=0.06,
        contact_force_threshold=1.0,
        required_distinct_feet=2,
        landing_steps=10,
        body_contact_steps=15,
        roll_threshold=1.4,
        pitch_threshold=1.6,
        base_height_threshold=0.15,
        lateral_limit=0.8,
    ):
        if num_feet < required_distinct_feet:
            raise ValueError("required_distinct_feet exceeds the available feet.")
        if landing_steps <= 0 or body_contact_steps <= 0:
            raise ValueError("Contact counters must contain at least one step.")

        self.num_envs = int(num_envs)
        self.num_feet = int(num_feet)
        self.num_boxes = int(num_boxes)
        self.device = device
        self.pass_margin = float(pass_margin)
        self.top_contact_tolerance = float(top_contact_tolerance)
        self.contact_force_threshold = float(contact_force_threshold)
        self.required_distinct_feet = int(required_distinct_feet)
        self.landing_steps = int(landing_steps)
        self.body_contact_steps = int(body_contact_steps)
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
        self.landing_foot_contact_mask = torch.zeros_like(
            self.foot_contact_mask
        )
        self.body_contact_counter = torch.zeros_like(self.next_box_idx)
        self.landing_counter = torch.zeros_like(self.next_box_idx)

        self.box_passed_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self.first_foot_contact_buf = torch.zeros_like(self.box_passed_buf)
        self.second_foot_contact_buf = torch.zeros_like(self.box_passed_buf)
        self.success_buf = torch.zeros_like(self.box_passed_buf)
        self.missed_box_buf = torch.zeros_like(self.box_passed_buf)
        self.out_of_track_buf = torch.zeros_like(self.box_passed_buf)
        self.fall_buf = torch.zeros_like(self.box_passed_buf)
        self.episode_timeout_buf = torch.zeros_like(self.box_passed_buf)

    def reset(self, env_ids):
        """Clear all episode state for selected environments."""
        self.next_box_idx[env_ids] = 0
        self.passed_box_count[env_ids] = 0
        self.foot_contact_mask[env_ids] = False
        self.landing_foot_contact_mask[env_ids] = False
        self.body_contact_counter[env_ids] = 0
        self.landing_counter[env_ids] = 0
        self._clear_events(env_ids)

    def _clear_events(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.box_passed_buf[env_ids] = False
        self.first_foot_contact_buf[env_ids] = False
        self.second_foot_contact_buf[env_ids] = False
        self.success_buf[env_ids] = False
        self.missed_box_buf[env_ids] = False
        self.out_of_track_buf[env_ids] = False
        self.fall_buf[env_ids] = False
        self.episode_timeout_buf[env_ids] = False

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
        natural_timeout,
    ):
        """Advance progress by one control step and update event buffers."""
        self._clear_events()
        env_ids = torch.arange(self.num_envs, device=self.device)
        active = self.next_box_idx < self.num_boxes
        target_indices = self.next_box_idx.clamp(max=self.num_boxes - 1)
        target_bounds = box_bounds[env_ids, target_indices]

        previous_contact_count = self.foot_contact_mask.sum(dim=1)
        top_contact = self._get_top_contacts(
            active,
            target_bounds,
            feet_positions,
            feet_terrain_heights,
            feet_contact_forces,
        )
        self.foot_contact_mask |= top_contact
        current_contact_count = self.foot_contact_mask.sum(dim=1)
        self.first_foot_contact_buf[:] = (
            active & (previous_contact_count < 1) & (current_contact_count >= 1)
        )
        self.second_foot_contact_buf[:] = (
            active & (previous_contact_count < 2) & (current_contact_count >= 2)
        )
        enough_feet = current_contact_count >= self.required_distinct_feet
        crossed_target = active & (
            base_positions[:, 0] > target_bounds[:, 1] + self.pass_margin
        )
        self.missed_box_buf[:] = crossed_target & ~enough_feet
        self.box_passed_buf[:] = crossed_target & enough_feet

        passed_envs = self.box_passed_buf.nonzero(as_tuple=False).flatten()
        self.next_box_idx[passed_envs] += 1
        self.passed_box_count[passed_envs] += 1
        self.foot_contact_mask[passed_envs] = False

        force_contact = feet_contact_forces.norm(dim=-1) > self.contact_force_threshold
        final_rear = box_bounds[:, -1, 1]
        post_box_ground = (
            (feet_positions[:, :, 0] > final_rear.unsqueeze(1))
            & (feet_positions[:, :, 0] < track_end_x.unsqueeze(1))
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
        course_complete = self.next_box_idx == self.num_boxes
        self.landing_foot_contact_mask |= (
            course_complete.unsqueeze(1) & post_box_ground
        )
        all_feet_landed = self.landing_foot_contact_mask.all(dim=1)
        base_height = base_positions[:, 2] - env_origins[:, 2]
        stable_landing = (
            course_complete
            & (base_positions[:, 0] > final_rear)
            & (base_positions[:, 0] < track_end_x)
            & (
                torch.abs(base_positions[:, 1] - env_origins[:, 1])
                <= self.lateral_limit
            )
            & (roll.abs() <= self.roll_threshold)
            & (pitch.abs() <= self.pitch_threshold)
            & (base_height >= self.base_height_threshold)
        )
        self.landing_counter[:] = torch.where(
            stable_landing,
            self.landing_counter + 1,
            torch.zeros_like(self.landing_counter),
        )
        landing_success = all_feet_landed & (
            self.landing_counter >= self.landing_steps
        )

        self.body_contact_counter[:] = torch.where(
            body_contact,
            self.body_contact_counter + 1,
            torch.zeros_like(self.body_contact_counter),
        )
        raw_fall = (
            (roll.abs() > self.roll_threshold)
            | (pitch.abs() > self.pitch_threshold)
            | (base_height < self.base_height_threshold)
            | (self.body_contact_counter >= self.body_contact_steps)
        )

        lateral_offset = (base_positions[:, 1] - env_origins[:, 1]).abs()
        raw_out_of_track = (
            (lateral_offset > self.lateral_limit)
            | (base_positions[:, 0] < track_start_x)
            | ((base_positions[:, 0] > track_end_x) & ~landing_success)
        )

        # Terminal causes are mutually exclusive. Unsafe task failures take
        # precedence over success, and natural timeout is considered only when
        # neither a failure nor success happened on the same control step.
        self.fall_buf[:] = raw_fall & ~self.missed_box_buf
        self.out_of_track_buf[:] = (
            raw_out_of_track & ~self.missed_box_buf & ~raw_fall
        )
        self.success_buf[:] = landing_success & ~self.failure_buf
        self.episode_timeout_buf[:] = (
            natural_timeout & ~self.failure_buf & ~self.success_buf
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
        return self.missed_box_buf | self.out_of_track_buf | self.fall_buf

    def apply_termination(self, reset_buf, time_out_buf):
        """Apply exclusive failure, success, and natural-timeout semantics."""
        time_out_buf.copy_(self.episode_timeout_buf)
        reset_buf |= self.failure_buf | self.success_buf | time_out_buf
