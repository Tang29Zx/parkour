"""Unit tests for tensor-only sequential box-course progress."""

import importlib.util
from pathlib import Path
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "Box progress tests require PyTorch.")
class BoxProgressTrackerTest(unittest.TestCase):
    def setUp(self):
        import torch

        tracker_path = (
            Path(__file__).resolve().parents[1] / "envs" / "base" / "box_progress.py"
        )
        spec = importlib.util.spec_from_file_location("box_progress", tracker_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.torch = torch
        self.BoxProgressTracker = module.BoxProgressTracker
        self.tracker = self.BoxProgressTracker(
            num_envs=2,
            num_feet=4,
            num_boxes=5,
            device="cpu",
        )
        single_bounds = torch.tensor(
            [
                [1.0, 2.0, -0.6, 0.6, 0.2],
                [3.0, 4.0, -0.6, 0.6, 0.3],
                [5.0, 6.0, -0.6, 0.6, 0.4],
                [7.0, 8.0, -0.6, 0.6, 0.4],
                [9.0, 10.0, -0.6, 0.6, 0.5],
            ],
            dtype=torch.float32,
        )
        self.box_bounds = single_bounds.unsqueeze(0).repeat(2, 1, 1)
        self.env_origins = torch.zeros(2, 3)
        self.track_start_x = torch.full((2,), -0.6)
        self.track_end_x = torch.full((2,), 14.9)
        self.landing_end_x = self.track_end_x.clone()
        self.feet_positions = torch.tensor(
            [
                [[0.0, -0.2, 0.0], [0.0, 0.2, 0.0], [0.0, -0.2, 0.0], [0.0, 0.2, 0.0]],
                [[0.0, -0.2, 0.0], [0.0, 0.2, 0.0], [0.0, -0.2, 0.0], [0.0, 0.2, 0.0]],
            ],
            dtype=torch.float32,
        )
        self.terrain_heights = torch.zeros(2, 4)
        self.contact_forces = torch.zeros(2, 4, 3)
        self.base_positions = torch.tensor(
            [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]], dtype=torch.float32
        )
        self.roll = torch.zeros(2)
        self.pitch = torch.zeros(2)
        self.body_contact = torch.zeros(2, dtype=torch.bool)
        self.body_contact_force = torch.zeros(2)
        self.base_vertical_velocity = torch.zeros(2)
        self.natural_timeout = torch.zeros(2, dtype=torch.bool)

    def update(self):
        self.tracker.update(
            box_bounds=self.box_bounds,
            env_origins=self.env_origins,
            track_start_x=self.track_start_x,
            track_end_x=self.track_end_x,
            feet_positions=self.feet_positions,
            feet_terrain_heights=self.terrain_heights,
            feet_contact_forces=self.contact_forces,
            base_positions=self.base_positions,
            roll=self.roll,
            pitch=self.pitch,
            body_contact=self.body_contact,
            body_contact_force=self.body_contact_force,
            base_vertical_velocity=self.base_vertical_velocity,
            natural_timeout=self.natural_timeout,
            landing_end_x=self.landing_end_x,
        )

    def put_foot_on_box(self, env_idx, foot_idx, box_idx):
        bounds = self.box_bounds[env_idx, box_idx]
        self.feet_positions[env_idx, foot_idx] = self.torch.tensor(
            [
                (bounds[0] + bounds[1]) / 2.0,
                (bounds[2] + bounds[3]) / 2.0,
                bounds[4],
            ]
        )
        self.terrain_heights[env_idx, foot_idx] = bounds[4]
        self.contact_forces[env_idx, foot_idx, 2] = 2.0

    def test_front_and_rear_contacts_may_qualify_sequentially(self):
        self.put_foot_on_box(0, 0, 0)
        self.update()
        self.update()
        self.contact_forces.zero_()
        self.put_foot_on_box(0, 2, 0)
        self.update()
        self.update()
        self.contact_forces.zero_()
        self.base_positions[0, 0] = 2.16
        self.update()

        self.assertTrue(self.tracker.box_passed_buf[0])
        self.assertEqual(self.tracker.passed_box_count[0].item(), 1)
        self.assertEqual(self.tracker.next_box_idx[0].item(), 1)

    def test_front_and_rear_events_fire_after_two_steps_once(self):
        self.put_foot_on_box(0, 0, 0)
        self.update()
        self.assertFalse(self.tracker.front_foot_contact_buf[0])
        self.assertFalse(self.tracker.rear_foot_contact_buf[0])

        self.update()
        self.assertTrue(self.tracker.front_foot_contact_buf[0])
        self.assertFalse(self.tracker.rear_foot_contact_buf[0])

        self.update()
        self.assertFalse(self.tracker.front_foot_contact_buf[0])
        self.assertFalse(self.tracker.rear_foot_contact_buf[0])

        self.contact_forces.zero_()
        self.put_foot_on_box(0, 2, 0)
        self.update()
        self.assertFalse(self.tracker.front_foot_contact_buf[0])
        self.assertFalse(self.tracker.rear_foot_contact_buf[0])

        self.update()
        self.assertFalse(self.tracker.front_foot_contact_buf[0])
        self.assertTrue(self.tracker.rear_foot_contact_buf[0])

    def test_front_and_rear_on_same_steps_emit_both_events_once(self):
        self.put_foot_on_box(0, 0, 0)
        self.put_foot_on_box(0, 2, 0)
        self.update()
        self.assertFalse(self.tracker.front_foot_contact_buf[0])
        self.assertFalse(self.tracker.rear_foot_contact_buf[0])

        self.update()
        self.assertTrue(self.tracker.front_foot_contact_buf[0])
        self.assertTrue(self.tracker.rear_foot_contact_buf[0])
        self.update()
        self.assertFalse(self.tracker.front_foot_contact_buf[0])
        self.assertFalse(self.tracker.rear_foot_contact_buf[0])

    def test_two_front_feet_do_not_satisfy_rear_contact_rule(self):
        self.put_foot_on_box(0, 0, 0)
        self.put_foot_on_box(0, 1, 0)
        self.update()
        self.update()
        self.base_positions[0, 0] = 2.16
        self.update()

        self.assertTrue(self.tracker.missed_box_buf[0])
        self.assertEqual(self.tracker.passed_box_count[0].item(), 0)

    def test_non_target_box_contact_does_not_change_progress(self):
        self.put_foot_on_box(0, 0, 1)
        self.put_foot_on_box(0, 2, 1)
        self.update()

        self.assertFalse(self.tracker.foot_contact_mask[0].any())
        self.assertEqual(self.tracker.front_contact_counter[0].item(), 0)
        self.assertEqual(self.tracker.rear_contact_counter[0].item(), 0)
        self.assertEqual(self.tracker.passed_box_count[0].item(), 0)

    def test_pass_event_is_one_step_and_backtracking_cannot_repeat_it(self):
        self.put_foot_on_box(0, 0, 0)
        self.put_foot_on_box(0, 2, 0)
        self.update()
        self.update()
        self.base_positions[0, 0] = 2.16
        self.update()
        self.assertTrue(self.tracker.box_passed_buf[0])

        self.base_positions[0, 0] = 1.0
        self.contact_forces.zero_()
        self.update()
        self.assertFalse(self.tracker.box_passed_buf[0])
        self.assertEqual(self.tracker.passed_box_count[0].item(), 1)

    def test_landing_requires_current_rear_support_for_ten_steps(self):
        self.tracker.next_box_idx[0] = 5
        self.tracker.passed_box_count[0] = 5
        self.feet_positions[0, :, 0] = 10.5
        self.feet_positions[0, :, 1] = self.torch.tensor([-0.3, 0.3, -0.3, 0.3])
        self.feet_positions[0, :, 2] = 0.0
        self.base_positions[0, 0] = 10.5
        self.terrain_heights[0] = 0.0
        # One front and one rear foot are enough, but both must support the
        # robot on every stable landing step.
        self.contact_forces[0, [0, 2], 2] = 2.0
        for _ in range(9):
            self.update()
        self.assertFalse(self.tracker.success_buf[0])
        self.assertEqual(self.tracker.landing_counter[0].item(), 9)

        self.natural_timeout[0] = True
        self.update()
        self.assertTrue(self.tracker.success_buf[0])
        self.assertFalse(self.tracker.episode_timeout_buf[0])

        reset_buf = self.torch.zeros(2, dtype=self.torch.bool)
        time_out_buf = self.torch.zeros(2, dtype=self.torch.bool)
        self.tracker.apply_termination(reset_buf, time_out_buf)
        self.assertFalse(time_out_buf[0])
        self.assertTrue(reset_buf[0])

    def test_landing_stability_failure_resets_the_counter(self):
        self.tracker.next_box_idx[0] = 5
        self.tracker.passed_box_count[0] = 5
        self.feet_positions[0, :, 0] = 10.5
        self.feet_positions[0, :, 1] = self.torch.tensor(
            [-0.3, 0.3, -0.3, 0.3]
        )
        self.feet_positions[0, :, 2] = 0.0
        self.base_positions[0, 0] = 10.5
        self.contact_forces[0, [0, 2], 2] = 2.0
        for _ in range(5):
            self.update()
        self.assertEqual(self.tracker.landing_counter[0].item(), 5)

        self.contact_forces[0, 2, 2] = 0.0
        self.update()
        self.assertEqual(self.tracker.landing_counter[0].item(), 0)

    def test_landing_uses_stricter_posture_height_and_body_contact_limits(self):
        self.tracker.next_box_idx[0] = 5
        self.tracker.passed_box_count[0] = 5
        self.feet_positions[0, :, 0] = 10.5
        self.feet_positions[0, :, 1] = self.torch.tensor(
            [-0.3, 0.3, -0.3, 0.3]
        )
        self.feet_positions[0, :, 2] = 0.0
        self.base_positions[0, 0] = 10.5
        self.contact_forces[0, [0, 2], 2] = 2.0

        invalid_cases = (
            (self.roll, 0.36),
            (self.pitch, 0.46),
        )
        for tensor, value in invalid_cases:
            tensor[0] = value
            self.update()
            self.assertEqual(self.tracker.landing_counter[0].item(), 0)
            tensor[0] = 0.0

        self.base_positions[0, 2] = 0.21
        self.update()
        self.assertEqual(self.tracker.landing_counter[0].item(), 0)
        self.base_positions[0, 2] = 0.5

        self.body_contact[0] = True
        self.body_contact_force[0] = 2.0
        self.update()
        self.assertEqual(self.tracker.landing_counter[0].item(), 0)

        self.body_contact[0] = False
        self.body_contact_force[0] = 0.0
        self.contact_forces[0, 2, 2] = 2.0
        self.base_vertical_velocity[0] = 0.51
        self.update()
        self.assertEqual(self.tracker.landing_counter[0].item(), 0)

    def test_failure_causes_and_true_timeout_are_separate(self):
        self.base_positions[0, 1] = 0.81
        self.roll[1] = 1.41
        self.natural_timeout[:] = True
        self.update()

        self.assertTrue(self.tracker.out_of_track_buf[0])
        self.assertTrue(self.tracker.fall_buf[1])
        self.assertEqual(self.tracker.episode_timeout_buf.tolist(), [False, False])

        self.roll.zero_()
        self.base_positions[:] = self.torch.tensor([0.0, 0.0, 0.5])
        self.natural_timeout.zero_()
        # Intermittent contact must accumulate inside the 25-step window.
        for step in range(14):
            self.body_contact[:] = step % 2 == 0
            self.body_contact_force[:] = self.body_contact.float() * 2.0
            self.update()
        self.assertFalse(self.tracker.fall_buf.any())

        self.body_contact[:] = True
        self.body_contact_force[:] = 2.0
        self.update()
        self.assertTrue(self.tracker.fall_buf.all())

    def test_single_severe_body_impact_fails_immediately(self):
        self.body_contact[0] = True
        self.body_contact_force[0] = 80.0
        self.update()

        self.assertTrue(self.tracker.severe_body_impact_buf[0])
        self.assertTrue(self.tracker.fall_buf[0])

    def test_task_deadline_is_incomplete_without_timeout_bootstrap(self):
        self.natural_timeout[0] = True
        self.update()

        reset_buf = self.torch.zeros(2, dtype=self.torch.bool)
        time_out_buf = self.natural_timeout.clone()
        self.tracker.apply_termination(reset_buf, time_out_buf)

        self.assertTrue(self.tracker.incomplete_buf[0])
        self.assertFalse(self.tracker.episode_timeout_buf[0])
        self.assertFalse(time_out_buf[0])
        self.assertTrue(reset_buf[0])
        self.assertFalse(reset_buf[1])

    def test_external_timeout_is_the_only_bootstrap_timeout(self):
        external_timeout = self.torch.tensor([True, False])
        self.tracker.update(
            box_bounds=self.box_bounds,
            env_origins=self.env_origins,
            track_start_x=self.track_start_x,
            track_end_x=self.track_end_x,
            feet_positions=self.feet_positions,
            feet_terrain_heights=self.terrain_heights,
            feet_contact_forces=self.contact_forces,
            base_positions=self.base_positions,
            roll=self.roll,
            pitch=self.pitch,
            body_contact=self.body_contact,
            body_contact_force=self.body_contact_force,
            base_vertical_velocity=self.base_vertical_velocity,
            natural_timeout=self.natural_timeout,
            landing_end_x=self.landing_end_x,
            external_timeout=external_timeout,
        )
        reset_buf = self.torch.zeros(2, dtype=self.torch.bool)
        time_out_buf = self.torch.zeros(2, dtype=self.torch.bool)
        self.tracker.apply_termination(reset_buf, time_out_buf)
        self.assertTrue(time_out_buf[0])
        self.assertTrue(reset_buf[0])
        self.assertFalse(self.tracker.incomplete_buf[0])

    def test_failure_takes_precedence_over_landing_success(self):
        self.tracker.next_box_idx[0] = 5
        self.tracker.passed_box_count[0] = 5
        self.tracker.landing_counter[0] = 9
        self.tracker.body_contact_history[0, 1:8] = True
        self.body_contact[0] = True
        self.body_contact_force[0] = 2.0
        self.feet_positions[0, :, 0] = 10.5
        self.feet_positions[0, :, 1] = self.torch.tensor(
            [-0.3, 0.3, -0.3, 0.3]
        )
        self.feet_positions[0, :, 2] = 0.0
        self.contact_forces[0, [0, 2], 2] = 2.0
        self.base_positions[0, 0] = 10.5
        self.update()

        self.assertTrue(self.tracker.fall_buf[0])
        self.assertFalse(self.tracker.success_buf[0])
        self.assertFalse(self.tracker.episode_timeout_buf[0])

    def test_reset_clears_all_progress_and_events(self):
        self.tracker.next_box_idx[:] = 3
        self.tracker.passed_box_count[:] = 3
        self.tracker.foot_contact_mask[:] = True
        self.tracker.front_contact_counter[:] = 2
        self.tracker.rear_contact_counter[:] = 2
        self.tracker.body_contact_history[:] = True
        self.tracker.body_contact_window_count[:] = 4
        self.tracker.max_body_contact_window_count[:] = 4
        self.tracker.landing_counter[:] = 5
        self.tracker.box_passed_buf[:] = True
        self.tracker.front_foot_contact_buf[:] = True
        self.tracker.rear_foot_contact_buf[:] = True
        self.tracker.body_contact_window_failure_buf[:] = True
        self.tracker.severe_body_impact_buf[:] = True
        self.tracker.success_buf[:] = True
        self.tracker.missed_box_buf[:] = True
        self.tracker.out_of_track_buf[:] = True
        self.tracker.landing_overrun_buf[:] = True
        self.tracker.fall_buf[:] = True
        self.tracker.incomplete_buf[:] = True
        self.tracker.episode_timeout_buf[:] = True
        self.tracker.task_progress_buf[:] = 1.0
        self.tracker.reset(self.torch.tensor([0, 1]))

        for value in (
            self.tracker.next_box_idx,
            self.tracker.passed_box_count,
            self.tracker.foot_contact_mask,
            self.tracker.front_contact_counter,
            self.tracker.rear_contact_counter,
            self.tracker.body_contact_history,
            self.tracker.body_contact_window_count,
            self.tracker.max_body_contact_window_count,
            self.tracker.landing_counter,
            self.tracker.box_passed_buf,
            self.tracker.front_foot_contact_buf,
            self.tracker.rear_foot_contact_buf,
            self.tracker.body_contact_window_failure_buf,
            self.tracker.severe_body_impact_buf,
            self.tracker.success_buf,
            self.tracker.missed_box_buf,
            self.tracker.out_of_track_buf,
            self.tracker.landing_overrun_buf,
            self.tracker.fall_buf,
            self.tracker.incomplete_buf,
            self.tracker.episode_timeout_buf,
            self.tracker.task_progress_buf,
        ):
            self.assertFalse(value.any())

    def test_4096_environment_buffer_shapes(self):
        for required_boxes in (1, 3, 5):
            tracker = self.BoxProgressTracker(
                4096, 4, 5, "cpu", required_boxes=required_boxes
            )
            self.assertEqual(tuple(tracker.next_box_idx.shape), (4096,))
            self.assertEqual(tuple(tracker.foot_contact_mask.shape), (4096, 4))
            self.assertEqual(
                tuple(tracker.body_contact_history.shape), (4096, 25)
            )
            self.assertEqual(
                tuple(tracker.front_foot_contact_buf.shape), (4096,)
            )
            self.assertEqual(
                tuple(tracker.rear_foot_contact_buf.shape), (4096,)
            )
            self.assertEqual(tuple(tracker.success_buf.shape), (4096,))
            self.assertEqual(tuple(tracker.incomplete_buf.shape), (4096,))
            self.assertEqual(tuple(tracker.landing_overrun_buf.shape), (4096,))

    def test_task_progress_depends_on_space_not_elapsed_time(self):
        self.base_positions[:, 0] = self.torch.tensor([0.0, 1.0])
        self.update()
        first = self.tracker.task_progress_buf.clone()
        for _ in range(20):
            self.update()
        self.assertTrue(self.torch.equal(first, self.tracker.task_progress_buf))
        self.assertGreater(first[1].item(), first[0].item())

    def test_required_boxes_is_validated(self):
        with self.assertRaises(TypeError):
            self.BoxProgressTracker(1, 4, 5, "cpu", required_boxes=1.0)
        with self.assertRaises(ValueError):
            self.BoxProgressTracker(1, 4, 5, "cpu", required_boxes=0)
        with self.assertRaises(ValueError):
            self.BoxProgressTracker(1, 4, 5, "cpu", required_boxes=6)

    def test_one_box_course_ignores_future_boxes_and_lands_before_box_two(self):
        self.tracker = self.BoxProgressTracker(
            2, 4, 5, "cpu", required_boxes=1
        )
        self.landing_end_x = self.box_bounds[:, 1, 0].clone()

        self.put_foot_on_box(0, 0, 0)
        self.put_foot_on_box(0, 2, 0)
        self.update()
        self.update()
        self.base_positions[0, 0] = 2.16
        self.update()
        self.assertTrue(self.tracker.box_passed_buf[0])
        self.assertEqual(self.tracker.next_box_idx[0].item(), 1)

        self.contact_forces.zero_()
        self.put_foot_on_box(0, 0, 1)
        self.put_foot_on_box(0, 2, 1)
        self.update()
        self.assertFalse(self.tracker.front_foot_contact_buf[0])
        self.assertFalse(self.tracker.rear_foot_contact_buf[0])
        self.assertEqual(self.tracker.passed_box_count[0].item(), 1)

        self.feet_positions[0, :, 0] = 2.5
        self.feet_positions[0, :, 1] = self.torch.tensor(
            [-0.3, 0.3, -0.3, 0.3]
        )
        self.feet_positions[0, :, 2] = 0.0
        self.terrain_heights[0] = 0.0
        self.contact_forces[0].zero_()
        self.contact_forces[0, :, 2] = 2.0
        self.base_positions[0, 0] = 2.5
        for _ in range(10):
            self.update()
        self.assertTrue(self.tracker.success_buf[0])

    def test_intermediate_course_landing_overrun_is_a_failure(self):
        self.tracker = self.BoxProgressTracker(
            2, 4, 5, "cpu", required_boxes=1
        )
        self.landing_end_x = self.box_bounds[:, 1, 0].clone()
        self.tracker.next_box_idx[0] = 1
        self.tracker.passed_box_count[0] = 1
        self.base_positions[0, 0] = self.landing_end_x[0]
        self.update()

        self.assertTrue(self.tracker.landing_overrun_buf[0])
        self.assertTrue(self.tracker.failure_buf[0])
        self.assertFalse(self.tracker.success_buf[0])
        self.assertFalse(self.tracker.episode_timeout_buf[0])

    def test_three_box_course_lands_between_boxes_three_and_four(self):
        self.tracker = self.BoxProgressTracker(
            2, 4, 5, "cpu", required_boxes=3
        )
        self.landing_end_x = self.box_bounds[:, 3, 0].clone()
        self.tracker.next_box_idx[0] = 3
        self.tracker.passed_box_count[0] = 3
        landing_x = 0.5 * (
            self.box_bounds[0, 2, 1] + self.landing_end_x[0]
        )
        self.feet_positions[0, :, 0] = landing_x
        self.feet_positions[0, :, 1] = self.torch.tensor(
            [-0.3, 0.3, -0.3, 0.3]
        )
        self.feet_positions[0, :, 2] = 0.0
        self.terrain_heights[0] = 0.0
        self.contact_forces[0, :, 2] = 2.0
        self.base_positions[0, 0] = landing_x

        for _ in range(10):
            self.update()

        self.assertTrue(self.tracker.success_buf[0])
        self.assertFalse(self.tracker.landing_overrun_buf[0])


if __name__ == "__main__":
    unittest.main()
