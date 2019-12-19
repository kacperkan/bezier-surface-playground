import math
import typing as t
import xml.etree.ElementTree as ET
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrr
import scipy as sp
import scipy.spatial.distance as sp_dist
import scipy.special
import tqdm

from scene_creator import SceneCreator, render

np.random.seed(0)

color_palette = (plt.cm.jet(np.arange(256))[:, :-1][::-1] * 255).astype(
    np.uint8
)

CUBE_SIZE = 5


def visualize_palette():
    width = 256
    palette = np.expand_dims(color_palette, axis=1)
    palette = np.repeat(palette, width, axis=1)

    plt.figure()
    plt.imshow(palette)
    plt.show()


class BezierSurface:
    def __init__(self, control_points: np.ndarray):
        self.control_points = control_points
        self.rng = np.random.RandomState(0)
        self.n_beziers = len(self.control_points)
        self._k_ij = control_points

        self._n = control_points.shape[1] - 1
        self._m = control_points.shape[2] - 1

        self._n_range = np.arange(0, self._n + 1).astype(np.float32)[
            np.newaxis
        ]
        self._m_range = np.arange(0, self._m + 1).astype(np.float32)[
            np.newaxis
        ]

        self._n_list = np.asarray(
            [self._n] * len(self._n_range), dtype=np.float32
        )
        self._m_list = np.asarray(
            [self._m] * len(self._m_range), dtype=np.float32
        )

        self._n_coeffs = sp.special.comb(self._n_list, self._n_range)
        self._m_coeffs = sp.special.comb(self._m_list, self._m_range)

    def sample_surface(self, n: int) -> np.ndarray:
        u_coordinates = self.rng.uniform(size=(self.n_beziers, n))
        v_coordinates = self.rng.uniform(size=(self.n_beziers, n))
        uv_coordinates = np.stack((u_coordinates, v_coordinates), axis=-1)
        return self.get_points_on_surface(uv_coordinates)

    def sample_surface_points_and_normals(
        self, n: int
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        u_coordinates = self.rng.uniform(size=(self.n_beziers, n))
        v_coordinates = self.rng.uniform(size=(self.n_beziers, n))
        uv_coordinates = np.stack((u_coordinates, v_coordinates), axis=-1)

        points = self.get_points_on_surface(uv_coordinates)
        normals = self.get_normals_for_points(uv_coordinates)

        return points, normals

    def get_normals_for_points(self, uv_coordinates: np.ndarray) -> np.ndarray:
        derivatives_u = self._derivatives_u(uv_coordinates)
        derivatives_v = self._derivatives_v(uv_coordinates)
        normals = np.cross(derivatives_v, derivatives_u)
        normals /= np.linalg.norm(normals, ord=2, axis=1, keepdims=True)
        return normals

    def get_points_on_surface(self, uv_coordinates: np.ndarray) -> np.ndarray:
        u, v = uv_coordinates[..., 0], uv_coordinates[..., 1]
        u = np.expand_dims(u, axis=-1)
        v = np.expand_dims(v, axis=-1)

        b_u = self._get_u_bernstein_polynomial_values(u)
        b_v = self._get_v_bernstein_polynomial_values(v)
        b_u = np.expand_dims(b_u, axis=3)
        b_v = np.expand_dims(b_v, axis=2)

        coeffs_mult = np.expand_dims(b_u * b_v, axis=-1)

        p_ij = coeffs_mult * np.expand_dims(self.control_points, 1)
        return p_ij.sum(axis=(2, 3))

    def _get_u_bernstein_polynomial_values(self, u: np.ndarray) -> np.ndarray:
        return (
            self._n_coeffs
            * (u ** self._n_range)
            * ((1 - u) ** (self._n - self._n_range))
        )

    def _get_v_bernstein_polynomial_values(self, v: np.ndarray) -> np.ndarray:
        return (
            self._m_coeffs
            * (v ** self._m_range)
            * ((1 - v) ** (self._m - self._m_range))
        )

    def _derivatives_u(self, uv_coordinates: np.ndarray) -> np.ndarray:
        u, v = uv_coordinates[..., 0], uv_coordinates[..., 1]
        u = np.expand_dims(u, axis=-1)
        v = np.expand_dims(v, axis=-1)
        b_v = self._get_v_bernstein_polynomial_values(v)
        coeffs = (
            b_v[:, :, None, :, None]
            * np.expand_dims(self.control_points, axis=1)
        ).sum(axis=3)

        return (
            -3 * (1 - u) ** 2 * coeffs[:, :, 0]
            + (3 * (1 - u) ** 2 - 6 * u * (1 - u)) * coeffs[:, :, 1]
            + (6 * u * (1 - u) - 3 * u ** 2) * coeffs[:, :, 2]
            + 3 * u ** 2 * coeffs[:, :, 3]
        )

    def _derivatives_v(self, uv_coordinates: np.ndarray) -> np.ndarray:
        u, v = uv_coordinates[..., 0], uv_coordinates[..., 1]
        u = np.expand_dims(u, axis=-1)
        v = np.expand_dims(v, axis=-1)
        b_u = self._get_u_bernstein_polynomial_values(v)
        coeffs = (
            b_u[:, :, :, None, None]
            * np.expand_dims(self.control_points, axis=1)
        ).sum(axis=2)
        return (
            -3 * (1 - v) ** 2 * coeffs[:, :, 0]
            + (3 * (1 - v) ** 2 - 6 * v * (1 - v)) * coeffs[:, :, 1]
            + (6 * v * (1 - v) - 3 * v ** 2) * coeffs[:, :, 2]
            + 3 * v ** 2 * coeffs[:, :, 3]
        )

    def get_distance_of_a_point(
        self, points: np.ndarray, num_iterations: int = 50
    ) -> np.ndarray:
        points = np.asarray(points)
        if len(points.shape) == 1:
            points = np.expand_dims(points, axis=0)

        if len(points.shape) == 2:
            points = np.expand_dims(points, axis=0)

        min_u, max_u = (
            np.zeros((self.n_beziers, points.shape[1])),
            np.ones((self.n_beziers, points.shape[1])),
        )
        min_v, max_v = (
            np.zeros((self.n_beziers, points.shape[1])),
            np.ones((self.n_beziers, points.shape[1])),
        )
        previous_u = None
        previous_v = None
        with tqdm.tqdm(total=num_iterations) as pbar:
            for i in range(num_iterations):
                mid_u = (min_u + max_u) / 2
                mid_v = (min_v + max_v) / 2

                if i > 0:
                    pbar.set_postfix(
                        OrderedDict(
                            {
                                "mid_u_diff": "{:.4f}".format(
                                    (mid_u - previous_u).sum()
                                ),
                                "mid_v_diff": "{:.4f}".format(
                                    (mid_v - previous_v).sum()
                                ),
                            }
                        )
                    )

                previous_u = min_u
                previous_v = min_v

                points_to_evaluate = np.stack(
                    [
                        np.stack(
                            [(min_u + mid_u) / 2, (min_v + mid_v) / 2], axis=-1
                        ),
                        np.stack(
                            [(max_u + mid_u) / 2, (min_v + mid_v) / 2], axis=-1
                        ),
                        np.stack(
                            [(min_u + mid_u) / 2, (max_v + mid_v) / 2], axis=-1
                        ),
                        np.stack(
                            [(max_u + mid_u) / 2, (max_v + mid_v) / 2], axis=-1
                        ),
                    ],
                    axis=2,
                )

                surface_points_coordinates = self.get_points_on_surface(
                    points_to_evaluate.reshape((self.n_beziers, -1, 2))
                ).reshape((self.n_beziers, -1, 4, 3))

                distances = (
                    (
                        surface_points_coordinates
                        - np.expand_dims(points, axis=2)
                    )
                    ** 2
                ).sum(axis=-1)
                index_closest_point_on_surface = np.argmin(distances, axis=2)

                where_0 = np.where(index_closest_point_on_surface == 0)
                where_1 = np.where(index_closest_point_on_surface == 1)
                where_2 = np.where(index_closest_point_on_surface == 2)
                where_3 = np.where(index_closest_point_on_surface == 3)

                max_u[where_0] = mid_u[where_0]
                max_v[where_0] = mid_v[where_0]

                min_u[where_1] = mid_u[where_1]
                max_v[where_1] = mid_v[where_1]

                max_u[where_2] = mid_u[where_2]
                min_v[where_2] = mid_v[where_2]

                min_u[where_3] = mid_u[where_3]
                min_v[where_3] = mid_v[where_3]
                pbar.update(1)

        uv_coordinates = np.stack([mid_u, mid_v], axis=-1)
        points_on_surface = self.get_points_on_surface(uv_coordinates)
        normals = self.get_normals_for_points(uv_coordinates)

        vectors = points_on_surface - points

        cos_theta = (normals * vectors).sum(axis=-1) / np.linalg.norm(
            vectors, ord=2, axis=-1
        )

        sign = (np.abs(np.arccos(cos_theta)) > np.pi / 2).astype(
            np.float32
        ) * 2 - 1

        distances = np.sqrt(((points_on_surface - points) ** 2).sum(axis=-1))
        return distances * sign


def distance_to_color_tuple(distance: float) -> t.Tuple[int]:
    max_dist = math.sqrt(3) * CUBE_SIZE
    distance += max_dist / 2
    percentage = distance / max_dist
    num_colors = len(color_palette)
    closest_coordinate = int(num_colors * percentage)
    return tuple(color_palette[closest_coordinate])


def generate_point_view(
    last_view_coords: t.Tuple[int],
    control_points: np.ndarray,
    sampled_points: np.ndarray,
    points_coordinates_with_distances: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    num_surfaces = len(control_points)
    creator = SceneCreator()
    creator.add_perspective_camera(
        (0, 0, 0), "toWorld", last_view_coords, (0, 0, 1)
    )

    control_points_to_render = control_points.reshape((num_surfaces, -1, 3))
    for surface_index in range(num_surfaces):
        for point in control_points_to_render[surface_index]:
            creator.add_sphere(tuple(point), 0.03, (100, 100, 100))

        for point_with_distance in points_coordinates_with_distances[
            surface_index
        ]:
            point = point_with_distance[:-1]
            distance = point_with_distance[-1]
            creator.add_sphere(
                tuple(point), 0.04, distance_to_color_tuple(distance)
            )

        normals = normals + sampled_points
        for normal, point in zip(
            normals[surface_index], sampled_points[surface_index]
        ):
            creator.add_sphere(tuple(point), 0.03, (0, 0, 255))
            creator.add_cylinder(
                tuple(normal), tuple(point), 0.03, (255, 0, 0)
            )

        for y in range(control_points[surface_index].shape[0]):
            for x in range(control_points[surface_index].shape[1]):
                if x < control_points[surface_index].shape[1] - 1:
                    creator.add_cylinder(
                        control_points[surface_index][y][x],
                        control_points[surface_index][y][x + 1],
                        radius=0.01,
                        color=(200, 200, 200),
                    )
                if y < control_points[surface_index].shape[0] - 1:
                    creator.add_cylinder(
                        control_points[surface_index][y][x],
                        control_points[surface_index][y + 1][x],
                        radius=0.01,
                        color=(200, 200, 200),
                    )

    img = render(creator.to_str())
    return img


def main():
    x = np.repeat(
        np.linspace(-2, 2, num=4, dtype=np.float32)[np.newaxis], 4, axis=0
    )
    y = np.repeat(
        np.linspace(-2, 2, num=4, dtype=np.float32)[:, np.newaxis], 4, axis=1
    )
    z = np.random.normal(0, 0.4, (4, 4)).astype(np.float32)

    z = np.asarray(
        [[0, 0, -1, -1], [0, 1.5, 1.3, 0], [0, 0.5, 2, 0], [0, 0, 0, 0]],
        dtype=np.float32,
    )
    control_points = np.stack((x, y, z), axis=-1)
    rotation_matrix = pyrr.Matrix33.from_y_rotation(30)
    rotated_control_points = control_points.dot(rotation_matrix.T)

    control_points = np.stack([control_points, rotated_control_points], axis=0)

    surface = BezierSurface(control_points)

    (
        sampled_points,
        sampled_normals,
    ) = surface.sample_surface_points_and_normals(300)

    num_points = 2000
    points_to_calculate_distance = np.random.uniform(
        -CUBE_SIZE / 2, CUBE_SIZE / 2, size=(num_points, 3)
    )
    points_to_calculate_distance[:, -1] = np.random.uniform(
        -CUBE_SIZE / 2, CUBE_SIZE / 2, size=(num_points,)
    )
    distances = surface.get_distance_of_a_point(points_to_calculate_distance)

    points_with_distances = np.concatenate(
        [
            np.repeat(points_to_calculate_distance[None], repeats=2, axis=0),
            np.expand_dims(distances, axis=-1),
        ],
        axis=-1,
    )
    control_points = control_points[:1]
    sampled_points = sampled_points[:1]
    points_with_distances = points_with_distances[:1]
    sampled_normals = sampled_normals[:1]

    top_view = generate_point_view(
        (0, 1, 16),
        control_points,
        sampled_points,
        points_with_distances,
        sampled_normals,
    )
    left_view = generate_point_view(
        (0, 16, 0),
        control_points,
        sampled_points,
        points_with_distances,
        sampled_normals,
    )
    front_view = generate_point_view(
        (16, 0, 0),
        control_points,
        sampled_points,
        points_with_distances,
        sampled_normals,
    )
    any_view = generate_point_view(
        (16, 13, 7),
        control_points,
        sampled_points,
        points_with_distances,
        sampled_normals,
    )

    final_img = np.zeros(
        (
            top_view.shape[0] + left_view.shape[0],
            top_view.shape[1] + left_view.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    final_img[: top_view.shape[0], : top_view.shape[1]] = top_view
    final_img[: top_view.shape[0], top_view.shape[1] :] = left_view
    final_img[top_view.shape[0] :, : top_view.shape[1]] = front_view
    final_img[top_view.shape[0] :, top_view.shape[1] :] = any_view

    cv2.imwrite("img.png", final_img[:, :, ::-1])


if __name__ == "__main__":
    main()
