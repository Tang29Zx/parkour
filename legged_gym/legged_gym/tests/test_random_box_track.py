"""Geometry contract tests for the fixed RandomBoxTrack layout."""

import importlib.util
import unittest
from copy import deepcopy


REQUIRED_MODULES = ("isaacgym", "numpy", "torch")
DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in REQUIRED_MODULES
)
if DEPENDENCIES_AVAILABLE:
    # Isaac Gym must initialize its native bindings before PyTorch is imported.
    import isaacgym  # noqa: F401


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    "RandomBoxTrack tests require Isaac Gym, NumPy, and PyTorch.",
)
class RandomBoxTrackTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from types import SimpleNamespace

        from legged_gym.utils.terrain.random_box_track import RandomBoxTrack

        self.np = np
        self.RandomBoxTrack = RandomBoxTrack
        self.cfg = SimpleNamespace(
            mesh_type=None,
            num_rows=1,
            num_cols=4,
            horizontal_scale=0.025,
            vertical_scale=0.005,
            border_size=5.0,
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            RandomBoxTrack_kwargs=dict(
                randomize=False,
                seed=0,
                track_length=10.7,
                track_width=2.0,
                spawn_margin=0.6,
                boxes=[
                    dict(gap=0.6, length=1.2, width=1.2, height=0.20, lateral_offset=0.0),
                    dict(gap=0.7, length=1.2, width=1.2, height=0.30, lateral_offset=0.0),
                    dict(gap=0.8, length=1.2, width=1.2, height=0.40, lateral_offset=0.0),
                    dict(gap=0.6, length=1.2, width=1.2, height=0.40, lateral_offset=0.0),
                    dict(gap=0.8, length=1.2, width=1.2, height=0.50, lateral_offset=0.0),
                ],
            ),
        )

    def test_fixed_layout_geometry(self):
        terrain = self.RandomBoxTrack(self.cfg, num_robots=4)

        expected_origins = self.np.array(
            [
                [
                    [5.6, 6.0, 0.0],
                    [5.6, 8.0, 0.0],
                    [5.6, 10.0, 0.0],
                    [5.6, 12.0, 0.0],
                ]
            ],
            dtype=self.np.float32,
        )
        self.np.testing.assert_allclose(terrain.env_origins, expected_origins)
        self.assertEqual(terrain.heightfield_raw.shape, (828, 720))
        self.assertEqual(terrain.terrain_mesh[0].shape, (168, 3))
        self.assertEqual(terrain.terrain_mesh[1].shape, (252, 3))

        first_layout = terrain.layout_metadata[0][0]
        self.assertEqual(len(first_layout["boxes"]), 5)
        self.np.testing.assert_allclose(
            first_layout["boxes"][0]["box_min"], [6.2, 5.4, 0.0]
        )
        self.np.testing.assert_allclose(
            first_layout["boxes"][0]["box_max"], [7.4, 6.6, 0.2]
        )
        self.np.testing.assert_allclose(
            first_layout["boxes"][4]["box_min"], [13.9, 5.4, 0.0]
        )
        self.np.testing.assert_allclose(
            first_layout["boxes"][4]["box_max"], [15.1, 6.6, 0.5]
        )

    def test_height_range_and_goal_queries(self):
        import torch

        class FakeGym:
            def add_triangle_mesh(self, sim, vertices, triangles, params):
                self.sim = sim
                self.vertices = vertices
                self.triangles = triangles
                self.params = params

        terrain = self.RandomBoxTrack(self.cfg, num_robots=4)
        fake_gym = FakeGym()
        terrain.add_terrain_to_sim(fake_gym, sim=object(), device="cpu")

        sample_points = torch.tensor(
            [
                [5.6, 6.0, 0.5],
                [6.8, 6.0, 0.5],
                [8.7, 6.0, 0.5],
                [10.7, 6.0, 0.5],
                [12.5, 6.0, 0.5],
                [14.5, 6.0, 0.5],
                [-1.0, -1.0, 0.5],
            ],
            dtype=torch.float32,
        )
        heights = terrain.get_terrain_heights(sample_points)
        self.assertAlmostEqual(heights[0].item(), 0.0)
        self.assertAlmostEqual(heights[1].item(), 0.20)
        self.assertAlmostEqual(heights[2].item(), 0.30)
        self.assertAlmostEqual(heights[3].item(), 0.40)
        self.assertAlmostEqual(heights[4].item(), 0.40)
        self.assertAlmostEqual(heights[5].item(), 0.50)
        self.assertTrue(torch.isneginf(heights[6]))

        positions = torch.tensor(
            [[5.6, 6.0, 0.5], [4.9, 6.0, 0.5]], dtype=torch.float32
        )
        self.assertEqual(terrain.in_terrain_range(positions).tolist(), [True, False])
        goals = terrain.get_goal_position(positions)
        torch.testing.assert_close(
            goals[0], torch.tensor([6.8, 6.0, 0.2], dtype=torch.float32)
        )
        torch.testing.assert_close(goals[1], positions[1])

        next_position = torch.tensor([[6.9, 6.0, 0.6]], dtype=torch.float32)
        next_goal = terrain.get_goal_position(next_position)
        torch.testing.assert_close(
            next_goal[0], torch.tensor([8.7, 6.0, 0.3], dtype=torch.float32)
        )

        self.assertEqual(fake_gym.params.nb_vertices, 168)
        self.assertEqual(fake_gym.params.nb_triangles, 252)

    def test_build_is_deterministic(self):
        first = self.RandomBoxTrack(self.cfg, num_robots=4)
        second = self.RandomBoxTrack(self.cfg, num_robots=4)

        self.np.testing.assert_array_equal(
            first.heightfield_raw, second.heightfield_raw
        )
        self.np.testing.assert_array_equal(
            first.terrain_mesh[0], second.terrain_mesh[0]
        )
        self.np.testing.assert_array_equal(
            first.terrain_mesh[1], second.terrain_mesh[1]
        )

    def test_seeded_gap_randomization(self):
        random_cfg = deepcopy(self.cfg)
        random_cfg.RandomBoxTrack_kwargs.update(
            randomize=True,
            track_length=15.5,
            first_gap_range=(0.8, 1.2),
            gap_distributions=[
                dict(name="dense", range=(0.45, 0.75), weight=0.2),
                dict(name="normal", range=(0.75, 1.20), weight=0.6),
                dict(name="sparse", range=(1.20, 1.60), weight=0.2),
            ],
            high_box_threshold=0.4,
            post_high_min_gap=0.8,
        )

        first = self.RandomBoxTrack(random_cfg, num_robots=4)
        second = self.RandomBoxTrack(random_cfg, num_robots=4)
        self.assertEqual(first.heightfield_raw.shape, (1020, 720))
        self.np.testing.assert_array_equal(
            first.heightfield_raw, second.heightfield_raw
        )
        self.assertGreater(
            len(
                {
                    tuple(box["gap"] for box in layout["boxes"])
                    for layout in first.layout_metadata[0]
                }
            ),
            1,
        )

        gap_ranges = {
            "dense": (0.45, 0.75),
            "normal": (0.75, 1.20),
            "sparse": (1.20, 1.60),
        }
        heights = [0.20, 0.30, 0.40, 0.40, 0.50]
        for layout in first.layout_metadata[0]:
            self.assertGreaterEqual(layout["boxes"][0]["gap"], 0.8 - 1e-8)
            self.assertLessEqual(layout["boxes"][0]["gap"], 1.2 + 1e-8)
            for box_idx, box in enumerate(layout["boxes"][1:], start=1):
                lower, upper = gap_ranges[box["gap_type"]]
                self.assertGreaterEqual(box["gap"], lower - 1e-8)
                self.assertLessEqual(box["gap"], upper + 1e-8)
                if heights[box_idx - 1] >= 0.4:
                    self.assertGreaterEqual(box["gap"], 0.8 - 1e-8)

        expected_gaps = [
            [1.15, 1.05, 0.45, 1.20, 1.40],
            [1.175, 0.95, 1.025, 0.90, 0.85],
            [0.875, 1.025, 0.975, 1.175, 1.075],
            [1.10, 1.10, 1.125, 1.20, 1.30],
        ]
        actual_gaps = [
            [box["gap"] for box in layout["boxes"]]
            for layout in first.layout_metadata[0]
        ]
        self.np.testing.assert_allclose(actual_gaps, expected_gaps)

        spawn_margin = random_cfg.RandomBoxTrack_kwargs["spawn_margin"]
        for col_idx, layout in enumerate(first.layout_metadata[0]):
            track_end = (
                first.env_origins[0, col_idx, 0]
                - spawn_margin
                + first.env_length
            )
            final_box_rear = layout["boxes"][-1]["box_max"][0]
            self.assertGreaterEqual(track_end - final_box_rear, 1.3 - 1e-6)

    def test_four_layouts_repeat_across_separated_physical_tracks(self):
        repeat_cfg = deepcopy(self.cfg)
        repeat_cfg.num_rows = 2
        repeat_cfg.num_cols = 8
        repeat_cfg.RandomBoxTrack_kwargs.update(
            randomize=True,
            num_unique_layouts=4,
            track_length=15.5,
            first_gap_range=(0.8, 1.2),
            gap_distributions=[
                dict(name="dense", range=(0.45, 0.75), weight=0.2),
                dict(name="normal", range=(0.75, 1.20), weight=0.6),
                dict(name="sparse", range=(1.20, 1.60), weight=0.2),
            ],
            high_box_threshold=0.4,
            post_high_min_gap=0.8,
        )

        terrain = self.RandomBoxTrack(repeat_cfg, num_robots=128)
        signatures = {}
        origins = terrain.env_origins.reshape(-1, 3)
        self.assertEqual(len(self.np.unique(origins, axis=0)), 16)
        self.assertEqual(terrain.num_physical_tracks, 16)
        self.assertEqual(terrain.num_unique_layouts, 4)

        for metadata_row in terrain.layout_metadata:
            for layout in metadata_row:
                signature = tuple(box["gap"] for box in layout["boxes"])
                layout_idx = layout["layout_index"]
                if layout_idx in signatures:
                    self.assertEqual(signature, signatures[layout_idx])
                else:
                    signatures[layout_idx] = signature
        self.assertEqual(set(signatures), {0, 1, 2, 3})
        self.assertEqual(len(set(signatures.values())), 4)

    def test_repeated_layout_count_is_validated(self):
        invalid_cfg = deepcopy(self.cfg)
        invalid_cfg.RandomBoxTrack_kwargs["num_unique_layouts"] = 0
        with self.assertRaises(ValueError):
            self.RandomBoxTrack(invalid_cfg, num_robots=4)

        invalid_cfg.RandomBoxTrack_kwargs["num_unique_layouts"] = 5
        with self.assertRaises(ValueError):
            self.RandomBoxTrack(invalid_cfg, num_robots=4)

    def test_box_bounds_tensor_query(self):
        import torch

        class FakeGym:
            def add_triangle_mesh(self, sim, vertices, triangles, params):
                pass

        terrain = self.RandomBoxTrack(self.cfg, num_robots=4)
        terrain.add_terrain_to_sim(FakeGym(), sim=object(), device="cpu")
        bounds = terrain.get_box_bounds(torch.tensor([[0, 0], [0, 3]]))
        self.assertEqual(tuple(bounds.shape), (2, 5, 5))
        torch.testing.assert_close(
            bounds[0, 0],
            torch.tensor([6.2, 7.4, 5.4, 6.6, 0.2]),
        )


if __name__ == "__main__":
    unittest.main()
