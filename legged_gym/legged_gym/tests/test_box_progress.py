"""Unit tests for the tensor-only five-box progress state."""

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
            natural_timeout=self.natural_timeout,
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

    def test_two_distinct_feet_may_contact_sequentially(self):
        self.put_foot_on_box(0, 0, 0)
        self.update()
        self.contact_forces.zero_()
        self.put_foot_on_box(0, 1, 0)
        self.update()
        self.contact_forces.zero_()
        self.base_positions[0, 0] = 2.16
        self.update()

        self.assertTrue(self.tracker.box_passed_buf[0])
        self.assertEqual(self.tracker.passed_box_count[0].item(), 1)
        self.assertEqual(self.tracker.next_box_idx[0].item(), 1)

    def test_first_and_second_foot_events_are_distinct_and_one_shot(self):
        self.put_foot_on_box(0, 0, 0)
        self.update()
        self.assertTrue(self.tracker.first_foot_contact_buf[0])
        self.assertFalse(self.tracker.second_foot_contact_buf[0])

        self.update()
        self.assertFalse(self.tracker.first_foot_contact_buf[0])
        self.assertFalse(self.tracker.second_foot_contact_buf[0])

        self.put_foot_on_box(0, 1, 0)
        self.update()
        self.assertFalse(self.tracker.first_foot_contact_buf[0])
        self.assertTrue(self.tracker.second_foot_contact_buf[0])

        self.put_foot_on_box(0, 2, 0)
        self.update()
        self.assertFalse(self.tracker.first_foot_contact_buf[0])
        self.assertFalse(self.tracker.second_foot_contact_buf[0])

    def test_two_feet_on_the_same_step_emit_both_events_once(self):
        self.put_foot_on_box(0, 0, 0)
        self.put_foot_on_box(0, 1, 0)
        self.update()

        self.assertTrue(self.tracker.first_foot_contact_buf[0])
        self.assertTrue(self.tracker.second_foot_contact_buf[0])
        self.update()
        self.assertFalse(self.tracker.first_foot_contact_buf[0])
        self.assertFalse(self.tracker.second_foot_contact_buf[0])

    def test_repeating_one_foot_does_not_satisfy_two_foot_rule(self):
        self.put_foot_on_box(0, 0, 0)
        self.update()
        self.update()
        self.base_positions[0, 0] = 2.16
        self.update()

        self.assertTrue(self.tracker.missed_box_buf[0])
        self.assertEqual(self.tracker.passed_box_count[0].item(), 0)

    def test_non_target_box_contact_does_not_change_progress(self):
        self.put_foot_on_box(0, 0, 1)
        self.put_foot_on_box(0, 1, 1)
        self.update()

        self.assertFalse(self.tracker.foot_contact_mask[0].any())
        self.assertEqual(self.tracker.passed_box_count[0].item(), 0)

    def test_pass_event_is_one_step_and_backtracking_cannot_repeat_it(self):
        self.put_foot_on_box(0, 0, 0)
        self.put_foot_on_box(0, 1, 0)
        self.update()
        self.base_positions[0, 0] = 2.16
        self.update()
        self.assertTrue(self.tracker.box_passed_buf[0])

        self.base_positions[0, 0] = 1.0
        self.contact_forces.zero_()
        self.update()
        self.assertFalse(self.tracker.box_passed_buf[0])
        self.assertEqual(self.tracker.passed_box_count[0].item(), 1)

    def test_landing_requires_ten_uninterrupted_steps(self):
        self.tracker.next_box_idx[0] = 5
        self.tracker.passed_box_count[0] = 5
        self.feet_positions[0, :, 0] = 10.5
        self.feet_positions[0, :, 1] = self.torch.tensor([-0.3, 0.3, -0.3, 0.3])
        self.feet_positions[0, :, 2] = 0.0
        self.terrain_heights[0] = 0.0
        self.contact_forces[0, :, 2] = 2.0

        for _ in range(9):
            self.update()
        self.assertFalse(self.tracker.success_buf[0])
        self.contact_forces[0, 0] = 0.0
        self.update()
        self.assertEqual(self.tracker.landing_counter[0].item(), 0)

        self.contact_forces[0, 0, 2] = 2.0
        self.natural_timeout[0] = True
        for _ in range(10):
            self.update()
        self.assertTrue(self.tracker.success_buf[0])
        self.assertFalse(self.tracker.episode_timeout_buf[0])

        reset_buf = self.torch.zeros(2, dtype=self.torch.bool)
        time_out_buf = self.torch.zeros(2, dtype=self.torch.bool)
        self.tracker.apply_termination(reset_buf, time_out_buf)
        self.assertTrue(time_out_buf[0])
        self.assertTrue(reset_buf[0])

    def test_failure_causes_and_true_timeout_are_separate(self):
        self.base_positions[0, 1] = 0.81
        self.roll[1] = 1.41
        self.natural_timeout[:] = True
        self.update()

        self.assertTrue(self.tracker.out_of_track_buf[0])
        self.assertTrue(self.tracker.fall_buf[1])
        self.assertEqual(self.tracker.episode_timeout_buf.tolist(), [True, True])

        self.roll.zero_()
        self.base_positions[:] = self.torch.tensor([0.0, 0.0, 0.5])
        self.natural_timeout.zero_()
        self.body_contact[:] = True
        for _ in range(15):
            self.update()
        self.assertTrue(self.tracker.fall_buf.all())

    def test_body_contact_can_be_non_terminal(self):
        self.tracker.reset_on_body_contact = False
        self.body_contact[:] = True
        for _ in range(20):
            self.update()

        self.assertEqual(self.tracker.body_contact_counter.tolist(), [20, 20])
        self.assertFalse(self.tracker.fall_buf.any())

    def test_reset_clears_all_progress_and_events(self):
        self.tracker.next_box_idx[:] = 3
        self.tracker.passed_box_count[:] = 3
        self.tracker.foot_contact_mask[:] = True
        self.tracker.body_contact_counter[:] = 4
        self.tracker.landing_counter[:] = 5
        self.tracker.box_passed_buf[:] = True
        self.tracker.first_foot_contact_buf[:] = True
        self.tracker.second_foot_contact_buf[:] = True
        self.tracker.success_buf[:] = True
        self.tracker.missed_box_buf[:] = True
        self.tracker.out_of_track_buf[:] = True
        self.tracker.fall_buf[:] = True
        self.tracker.episode_timeout_buf[:] = True
        self.tracker.reset(self.torch.tensor([0, 1]))

        for value in (
            self.tracker.next_box_idx,
            self.tracker.passed_box_count,
            self.tracker.foot_contact_mask,
            self.tracker.body_contact_counter,
            self.tracker.landing_counter,
            self.tracker.box_passed_buf,
            self.tracker.first_foot_contact_buf,
            self.tracker.second_foot_contact_buf,
            self.tracker.success_buf,
            self.tracker.missed_box_buf,
            self.tracker.out_of_track_buf,
            self.tracker.fall_buf,
            self.tracker.episode_timeout_buf,
        ):
            self.assertFalse(value.any())

    def test_4096_environment_buffer_shapes(self):
        tracker = self.BoxProgressTracker(4096, 4, 5, "cpu")
        self.assertEqual(tuple(tracker.next_box_idx.shape), (4096,))
        self.assertEqual(tuple(tracker.foot_contact_mask.shape), (4096, 4))
        self.assertEqual(tuple(tracker.first_foot_contact_buf.shape), (4096,))
        self.assertEqual(tuple(tracker.second_foot_contact_buf.shape), (4096,))
        self.assertEqual(tuple(tracker.success_buf.shape), (4096,))


if __name__ == "__main__":
    unittest.main()
