"""Box terrain with matching triangle-mesh and heightfield geometry."""

from copy import deepcopy

import numpy as np
from isaacgym import gymapi
import torch

from legged_gym.utils import trimesh


class RandomBoxTrack:
    """Build a reproducible grid of fixed-height box layouts."""

    default_kwargs = dict(
        randomize=False,
        seed=0,
        num_unique_layouts=None,
        track_length=10.7,
        track_width=2.0,
        spawn_margin=0.6,
        first_gap_range=(0.8, 1.2),
        gap_distributions=[
            dict(name="dense", range=(0.45, 0.75), weight=0.2),
            dict(name="normal", range=(0.75, 1.20), weight=0.6),
            dict(name="sparse", range=(1.20, 1.60), weight=0.2),
        ],
        high_box_threshold=0.4,
        post_high_min_gap=0.8,
        boxes=[
            dict(gap=0.6, length=1.2, width=1.2, height=0.20, lateral_offset=0.0),
            dict(gap=0.7, length=1.2, width=1.2, height=0.30, lateral_offset=0.0),
            dict(gap=0.8, length=1.2, width=1.2, height=0.40, lateral_offset=0.0),
            dict(gap=0.6, length=1.2, width=1.2, height=0.40, lateral_offset=0.0),
            dict(gap=0.8, length=1.2, width=1.2, height=0.50, lateral_offset=0.0),
        ],
    )

    def __init__(self, cfg, num_robots: int) -> None:
        self.cfg = cfg
        self.num_robots = num_robots

        if self.cfg.mesh_type is not None:
            raise ValueError(
                "RandomBoxTrack requires cfg.terrain.mesh_type to be None."
            )
        if not hasattr(self.cfg, "RandomBoxTrack_kwargs"):
            raise ValueError(
                "RandomBoxTrack requires cfg.terrain.RandomBoxTrack_kwargs."
            )

        self.track_kwargs = deepcopy(self.default_kwargs)
        for key, value in self.cfg.RandomBoxTrack_kwargs.items():
            self.track_kwargs[key] = deepcopy(value)

        self.seed = int(self.track_kwargs["seed"])
        self.env_length = float(self.track_kwargs["track_length"])
        self.env_width = float(self.track_kwargs["track_width"])
        self.num_physical_tracks = self.cfg.num_rows * self.cfg.num_cols
        configured_layouts = self.track_kwargs["num_unique_layouts"]
        if configured_layouts is None:
            self.num_unique_layouts = self.num_physical_tracks
            self.repeat_layouts = False
        else:
            if not isinstance(configured_layouts, (int, np.integer)):
                raise TypeError("num_unique_layouts must be an integer or None.")
            self.num_unique_layouts = int(configured_layouts)
            self.repeat_layouts = True
        if not 1 <= self.num_unique_layouts <= self.num_physical_tracks:
            raise ValueError(
                "num_unique_layouts must be between 1 and the number of "
                "physical tracks."
            )
        self.env_origins = np.zeros(
            (self.cfg.num_rows, self.cfg.num_cols, 3), dtype=np.float32
        )

        self._initialize_geometry()

    def _as_cells(self, value, scale, name):
        cells = int(round(float(value) / float(scale)))
        if cells < 0 or not np.isclose(
            cells * float(scale), float(value), rtol=0.0, atol=1e-8
        ):
            raise ValueError(
                f"{name}={value} must be a non-negative multiple of {scale}."
            )
        return cells

    def _validate_layout(self):
        if self.env_length <= 0.0 or self.env_width <= 0.0:
            raise ValueError("Track length and width must be positive.")

        spawn_margin = float(self.track_kwargs["spawn_margin"])
        if spawn_margin < 0.0 or spawn_margin >= self.env_length:
            raise ValueError("spawn_margin must lie inside the track.")
        if not self.track_kwargs["boxes"]:
            raise ValueError("At least one fixed box must be configured.")

        if self.seed < 0:
            raise ValueError("seed must be non-negative.")

        max_random_gap = 0.0
        if self.track_kwargs["randomize"]:
            first_gap_range = self.track_kwargs["first_gap_range"]
            self._validate_range(first_gap_range, "first_gap_range")
            gap_distributions = self.track_kwargs["gap_distributions"]
            if not gap_distributions:
                raise ValueError("gap_distributions must not be empty.")
            total_weight = 0.0
            for distribution_idx, distribution in enumerate(gap_distributions):
                self._validate_range(
                    distribution["range"],
                    f"gap_distributions[{distribution_idx}].range",
                )
                weight = float(distribution["weight"])
                if weight <= 0.0:
                    raise ValueError("Gap distribution weights must be positive.")
                total_weight += weight
                max_random_gap = max(
                    max_random_gap, float(distribution["range"][1])
                )
            if total_weight <= 0.0:
                raise ValueError(
                    "Gap distribution weights must sum to a positive value."
                )
            if float(self.track_kwargs["post_high_min_gap"]) < 0.0:
                raise ValueError("post_high_min_gap must be non-negative.")

        forward_distance = 0.0
        for box_idx, box in enumerate(self.track_kwargs["boxes"]):
            positive_box_keys = ("length", "width", "height")
            if any(float(box[key]) <= 0.0 for key in positive_box_keys):
                raise ValueError(
                    f"Box {box_idx} length, width, and height must be positive."
                )
            if float(box["gap"]) < 0.0:
                raise ValueError(f"Box {box_idx} gap must be non-negative.")
            height_choices = box.get("height_choices")
            if height_choices is not None:
                if not isinstance(height_choices, (list, tuple)):
                    raise TypeError(
                        f"Box {box_idx} height_choices must be a list or tuple."
                    )
                if not height_choices:
                    raise ValueError(
                        f"Box {box_idx} height_choices must not be empty."
                    )
                for choice_idx, height in enumerate(height_choices):
                    if float(height) <= 0.0:
                        raise ValueError(
                            f"Box {box_idx} height_choices[{choice_idx}] "
                            "must be positive."
                        )
                    self._as_cells(
                        height,
                        self.cfg.vertical_scale,
                        f"boxes[{box_idx}].height_choices[{choice_idx}]",
                    )

            if self.track_kwargs["randomize"]:
                if box_idx == 0:
                    forward_distance += float(
                        self.track_kwargs["first_gap_range"][1]
                    )
                else:
                    forward_distance += max_random_gap
            else:
                forward_distance += float(box["gap"])
            box_end = forward_distance + float(box["length"])
            box_y_min = (
                self.env_width / 2.0
                + float(box["lateral_offset"])
                - float(box["width"]) / 2.0
            )
            box_y_max = box_y_min + float(box["width"])
            if spawn_margin + box_end > self.env_length:
                raise ValueError(f"Box {box_idx} extends beyond the track length.")
            if box_y_min < 0.0 or box_y_max > self.env_width:
                raise ValueError(f"Box {box_idx} extends beyond the track width.")
            forward_distance = box_end

    @staticmethod
    def _validate_range(value_range, name):
        if len(value_range) != 2:
            raise ValueError(f"{name} must contain exactly two values.")
        lower, upper = map(float, value_range)
        if lower < 0.0 or upper < lower:
            raise ValueError(f"{name} must be a non-negative ordered range.")

    def _sample_aligned(self, rng, value_range, name):
        scale = float(self.cfg.horizontal_scale)
        lower, upper = map(float, value_range)
        lower_cell = int(np.ceil(lower / scale - 1e-8))
        upper_cell = int(np.floor(upper / scale + 1e-8))
        if lower_cell > upper_cell:
            raise ValueError(f"{name} contains no value aligned to {scale}.")
        return float(rng.integers(lower_cell, upper_cell + 1) * scale)

    def _sample_gap(self, rng, box_idx, previous_height):
        if not self.track_kwargs["randomize"]:
            return float(self.track_kwargs["boxes"][box_idx]["gap"]), "fixed"
        if box_idx == 0:
            gap = self._sample_aligned(
                rng, self.track_kwargs["first_gap_range"], "first_gap_range"
            )
            return gap, "run_up"

        minimum_gap = 0.0
        if previous_height >= float(self.track_kwargs["high_box_threshold"]):
            minimum_gap = float(self.track_kwargs["post_high_min_gap"])
        candidates = []
        weights = []
        for distribution in self.track_kwargs["gap_distributions"]:
            lower, upper = map(float, distribution["range"])
            lower = max(lower, minimum_gap)
            if lower <= upper:
                candidates.append((distribution, (lower, upper)))
                weights.append(float(distribution["weight"]))
        if not candidates:
            raise ValueError("No gap distribution satisfies post_high_min_gap.")
        probabilities = np.asarray(weights, dtype=np.float64)
        probabilities /= probabilities.sum()
        distribution, value_range = candidates[
            rng.choice(len(candidates), p=probabilities)
        ]
        return self._sample_aligned(
            rng, value_range, f"{distribution['name']} gap range"
        ), str(distribution["name"])

    def _initialize_geometry(self):
        self._validate_layout()

        horizontal_scale = float(self.cfg.horizontal_scale)
        vertical_scale = float(self.cfg.vertical_scale)
        self.border = self._as_cells(
            self.cfg.border_size, horizontal_scale, "border_size"
        )
        self.track_length_px = self._as_cells(
            self.env_length, horizontal_scale, "track_length"
        )
        self.track_width_px = self._as_cells(
            self.env_width, horizontal_scale, "track_width"
        )
        self.track_center_y_px = self._as_cells(
            self.env_width / 2.0, horizontal_scale, "track_width / 2"
        )
        self.spawn_margin_px = self._as_cells(
            self.track_kwargs["spawn_margin"], horizontal_scale, "spawn_margin"
        )
        self.num_boxes = len(self.track_kwargs["boxes"])
        self.box_specs_by_track = [
            [None for _ in range(self.cfg.num_cols)]
            for _ in range(self.cfg.num_rows)
        ]
        self.box_goal_offsets = np.zeros(
            (self.cfg.num_rows, self.cfg.num_cols, self.num_boxes, 3),
            dtype=np.float32,
        )
        self.box_bounds = np.zeros(
            (self.cfg.num_rows, self.cfg.num_cols, self.num_boxes, 5),
            dtype=np.float32,
        )

        map_x_size = self.cfg.num_rows * self.track_length_px + 2 * self.border
        map_y_size = self.cfg.num_cols * self.track_width_px + 2 * self.border
        self.tot_rows = map_x_size
        self.tot_cols = map_y_size
        self.heightfield_raw = np.zeros(
            (map_x_size, map_y_size), dtype=np.float32
        )
        self.heightsamples = self.heightfield_raw
        self.layout_metadata = []

        ground_size = np.array(
            [
                map_x_size * horizontal_scale,
                map_y_size * horizontal_scale,
                0.02,
            ],
            dtype=np.float32,
        )
        meshes = [
            trimesh.box_trimesh(
                ground_size,
                np.array(
                    [ground_size[0] / 2.0, ground_size[1] / 2.0, -0.01],
                    dtype=np.float32,
                ),
            )
        ]

        for row_idx in range(self.cfg.num_rows):
            metadata_row = []
            for col_idx in range(self.cfg.num_cols):
                physical_track_idx = row_idx * self.cfg.num_cols + col_idx
                layout_idx = physical_track_idx % self.num_unique_layouts
                track_x0_px = self.border + row_idx * self.track_length_px
                track_y0_px = self.border + col_idx * self.track_width_px
                spawn_x_px = track_x0_px + self.spawn_margin_px
                spawn_y_px = track_y0_px + self.track_center_y_px

                spawn_world = np.array(
                    [
                        spawn_x_px * horizontal_scale,
                        spawn_y_px * horizontal_scale,
                        0.0,
                    ],
                    dtype=np.float32,
                )
                self.env_origins[row_idx, col_idx] = spawn_world
                if self.repeat_layouts:
                    seed_sequence = [self.seed, 0, layout_idx]
                else:
                    seed_sequence = [self.seed, row_idx, col_idx]
                rng = np.random.default_rng(
                    np.random.SeedSequence(seed_sequence)
                )
                box_specs = []
                forward_distance = 0.0
                previous_height = 0.0
                for box_idx, box in enumerate(self.track_kwargs["boxes"]):
                    height_choices = box.get("height_choices")
                    if height_choices is None:
                        box_height = float(box["height"])
                    else:
                        box_height = float(
                            height_choices[layout_idx % len(height_choices)]
                        )
                    gap, gap_type = self._sample_gap(
                        rng, box_idx, previous_height
                    )
                    forward_distance += gap
                    box_y_min_local = (
                        self.env_width / 2.0
                        + float(box["lateral_offset"])
                        - float(box["width"]) / 2.0
                    )
                    spec = dict(
                        index=box_idx,
                        front_distance=forward_distance,
                        gap=gap,
                        gap_type=gap_type,
                        length=float(box["length"]),
                        width=float(box["width"]),
                        height=box_height,
                        lateral_offset=float(box["lateral_offset"]),
                        front_distance_px=self._as_cells(
                            forward_distance,
                            horizontal_scale,
                            f"boxes[{box_idx}] front distance",
                        ),
                        length_px=self._as_cells(
                            box["length"],
                            horizontal_scale,
                            f"boxes[{box_idx}].length",
                        ),
                        width_px=self._as_cells(
                            box["width"],
                            horizontal_scale,
                            f"boxes[{box_idx}].width",
                        ),
                        height_px=self._as_cells(
                            box_height,
                            vertical_scale,
                            f"boxes[{box_idx}].height",
                        ),
                        y_min_local_px=self._as_cells(
                            box_y_min_local,
                            horizontal_scale,
                            f"boxes[{box_idx}] lateral boundary",
                        ),
                    )
                    box_specs.append(spec)
                    self.box_goal_offsets[row_idx, col_idx, box_idx] = [
                        spec["front_distance"] + spec["length"] / 2.0,
                        spec["lateral_offset"],
                        spec["height"],
                    ]
                    forward_distance += float(box["length"])
                    previous_height = box_height
                self.box_specs_by_track[row_idx][col_idx] = box_specs

                box_metadata = []
                for spec in box_specs:
                    box_x0_px = spawn_x_px + spec["front_distance_px"]
                    box_x1_px = box_x0_px + spec["length_px"]
                    box_y0_px = track_y0_px + spec["y_min_local_px"]
                    box_y1_px = box_y0_px + spec["width_px"]

                    self.heightfield_raw[
                        box_x0_px:box_x1_px, box_y0_px:box_y1_px
                    ] = spec["height_px"]

                    box_min_world = np.array(
                        [
                            box_x0_px * horizontal_scale,
                            box_y0_px * horizontal_scale,
                            0.0,
                        ],
                        dtype=np.float32,
                    )
                    box_max_world = np.array(
                        [
                            box_x1_px * horizontal_scale,
                            box_y1_px * horizontal_scale,
                            spec["height_px"] * vertical_scale,
                        ],
                        dtype=np.float32,
                    )
                    box_center_world = (box_min_world + box_max_world) / 2.0
                    self.box_bounds[row_idx, col_idx, spec["index"]] = [
                        box_min_world[0],
                        box_max_world[0],
                        box_min_world[1],
                        box_max_world[1],
                        box_max_world[2],
                    ]
                    meshes.append(
                        trimesh.box_trimesh(
                            box_max_world - box_min_world, box_center_world
                        )
                    )
                    box_metadata.append(
                        dict(
                            index=spec["index"],
                            gap=spec["gap"],
                            gap_type=spec["gap_type"],
                            box_min_local=[
                                spec["front_distance"],
                                spec["lateral_offset"] - spec["width"] / 2.0,
                                0.0,
                            ],
                            box_max_local=[
                                spec["front_distance"] + spec["length"],
                                spec["lateral_offset"] + spec["width"] / 2.0,
                                spec["height"],
                            ],
                            box_min=box_min_world.tolist(),
                            box_max=box_max_world.tolist(),
                            box_center=box_center_world.tolist(),
                        )
                    )
                metadata_row.append(
                    dict(
                        layout_id=(
                            f"{'random' if self.track_kwargs['randomize'] else 'fixed'}"
                            f"_{'five' if self.num_boxes == 5 else self.num_boxes}_box"
                            f"_v1_layout{layout_idx}"
                            f"_r{row_idx}_c{col_idx}"
                        ),
                        layout_index=layout_idx,
                        physical_track_index=physical_track_idx,
                        seed=self.seed,
                        row=row_idx,
                        col=col_idx,
                        spawn_position=spawn_world.tolist(),
                        boxes=box_metadata,
                    )
                )
            self.layout_metadata.append(metadata_row)

        self.terrain_mesh = self._combine_meshes(meshes)

    @staticmethod
    def _combine_meshes(meshes):
        vertices = []
        triangles = []
        vertex_offset = 0
        for mesh_vertices, mesh_triangles in meshes:
            vertices.append(mesh_vertices.astype(np.float32, copy=False))
            triangles.append(
                mesh_triangles.astype(np.uint32, copy=False) + vertex_offset
            )
            vertex_offset += mesh_vertices.shape[0]
        return np.concatenate(vertices, axis=0), np.concatenate(triangles, axis=0)

    def add_terrain_to_sim(self, gym, sim, device="cpu"):
        """Add the fixed terrain mesh and upload its height lookup buffer."""
        self.gym = gym
        self.sim = sim
        self.device = device

        mesh_vertices, mesh_triangles = self.terrain_mesh
        mesh_params = gymapi.TriangleMeshParams()
        mesh_params.nb_vertices = mesh_vertices.shape[0]
        mesh_params.nb_triangles = mesh_triangles.shape[0]
        mesh_params.static_friction = self.cfg.static_friction
        mesh_params.dynamic_friction = self.cfg.dynamic_friction
        mesh_params.restitution = self.cfg.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            mesh_vertices.flatten(order="C"),
            mesh_triangles.flatten(order="C"),
            mesh_params,
        )

        self.env_origins_pyt = torch.from_numpy(self.env_origins).to(self.device)
        self.heightfield_raw_pyt = torch.tensor(
            self.heightfield_raw, dtype=torch.float32, device=self.device
        )
        self.box_goal_offsets_pyt = torch.from_numpy(self.box_goal_offsets).to(
            self.device
        )
        self.box_bounds_pyt = torch.from_numpy(self.box_bounds).to(self.device)

    def get_box_bounds(self, track_indices):
        """Return world-frame box bounds for the requested ``(row, col)`` tracks."""
        if not hasattr(self, "box_bounds_pyt"):
            raise RuntimeError("add_terrain_to_sim() must run before tensor queries.")
        if track_indices.ndim != 2 or track_indices.shape[1] != 2:
            raise ValueError("track_indices must have shape (N, 2).")
        return self.box_bounds_pyt[track_indices[:, 0], track_indices[:, 1]]

    def get_track_idx(self, positions, clipped=True):
        """Return row and column indices for world-frame positions."""
        terrain_start = self.border * self.cfg.horizontal_scale
        track_size = torch.tensor(
            [self.env_length, self.env_width], device=positions.device
        )
        track_idx = torch.floor(
            (positions[:, :2] - terrain_start) / track_size
        ).to(torch.long)
        if clipped:
            track_idx = torch.minimum(
                torch.maximum(track_idx, torch.zeros_like(track_idx)),
                torch.tensor(
                    [self.cfg.num_rows - 1, self.cfg.num_cols - 1],
                    dtype=torch.long,
                    device=positions.device,
                ),
            )
        return track_idx

    def in_terrain_range(self, positions):
        """Check whether positions lie inside the track grid, excluding the border."""
        track_idx = self.get_track_idx(positions, clipped=False)
        return (
            (track_idx[:, 0] >= 0)
            & (track_idx[:, 0] < self.cfg.num_rows)
            & (track_idx[:, 1] >= 0)
            & (track_idx[:, 1] < self.cfg.num_cols)
        )

    @torch.no_grad()
    def get_terrain_heights(self, points):
        """Return terrain surface heights below world-frame sample points."""
        points_shape = points.shape
        flat_points = points.reshape(-1, 3)
        points_x_px = torch.floor(
            flat_points[:, 0] / self.cfg.horizontal_scale
        ).to(torch.long)
        points_y_px = torch.floor(
            flat_points[:, 1] / self.cfg.horizontal_scale
        ).to(torch.long)
        out_of_range = (
            (points_x_px < 0)
            | (points_x_px >= self.heightfield_raw_pyt.shape[0])
            | (points_y_px < 0)
            | (points_y_px >= self.heightfield_raw_pyt.shape[1])
        )
        points_x_px = torch.clamp(
            points_x_px, 0, self.heightfield_raw_pyt.shape[0] - 1
        )
        points_y_px = torch.clamp(
            points_y_px, 0, self.heightfield_raw_pyt.shape[1] - 1
        )
        heights = (
            self.heightfield_raw_pyt[points_x_px, points_y_px]
            * self.cfg.vertical_scale
        )
        heights[out_of_range] = float("-inf")
        return heights.reshape(points_shape[:-1])

    def get_goal_position(self, positions):
        """Return the fixed box-top center for each position's track."""
        in_range = self.in_terrain_range(positions)
        track_idx = self.get_track_idx(positions)
        origins = self.env_origins_pyt[track_idx[:, 0], track_idx[:, 1]]
        forward_distance = positions[:, 0] - origins[:, 0]
        track_goal_offsets = self.box_goal_offsets_pyt[
            track_idx[:, 0], track_idx[:, 1]
        ]
        passed_centers = (
            forward_distance.unsqueeze(1) > track_goal_offsets[:, :, 0]
        )
        goal_indices = torch.clamp(
            passed_centers.sum(dim=1), max=self.num_boxes - 1
        )
        goals = origins + track_goal_offsets[
            torch.arange(positions.shape[0], device=positions.device), goal_indices
        ]
        goals[~in_range] = positions[~in_range]
        return goals
