from unittest import TestCase
from dshin import compgeom
import numpy as np


class TestRayTriangleIntersection(TestCase):
    def test_simple(self):
        v = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], ])
        f = np.array([[0, 1, 2], [1, 2, 3], ])
        ray = np.array([[0.25, 0.25, -2], [0.25, 0.25, -1], ])
        intersections, is_valid = compgeom.ray_triangle_intersection(f, v, ray)
        self.assertTrue((is_valid == [True, False]).all())
        self.assertTrue((intersections[0, :] == [0.25, 0.25, 0]).all())
        self.assertTrue(np.isnan(intersections[1, :]).all())

    def test_ray_direction_ignored(self):
        v = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], ])
        f = np.array([[0, 1, 2], [3, 4, 5], ])
        ray = np.array([[0.25, 0.25, 0.5], [0.25, 0.25, 0.4], ])
        intersections, is_valid = compgeom.ray_triangle_intersection(f, v, ray)
        self.assertTrue((is_valid == [True, True]).all())
        self.assertTrue((intersections[0, :] == [0.25, 0.25, 0]).all())
        self.assertTrue((intersections[1, :] == [0.25, 0.25, 1]).all())

    def test_on_line(self):
        v = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], ])
        f = np.array([[0, 1, 3], [0, 2, 3], ])
        ray = np.array([[0.25, 0.25, -2], [0.25, 0.25, -1], ])
        intersections, is_valid = compgeom.ray_triangle_intersection(f, v, ray)
        self.assertTrue((is_valid == [True, True]).all())
        self.assertTrue((intersections == [0.25, 0.25, 0]).all())

    def test_on_vertex(self):
        v = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], ])
        f = np.array([[0, 1, 3], [0, 2, 3], ])
        ray = np.array([[0, 0, -2], [0, 0, -1], ])
        intersections, is_valid = compgeom.ray_triangle_intersection(f, v, ray)
        self.assertTrue((is_valid == [True, True]).all())
        self.assertTrue((intersections == [0, 0, 0]).all())

    def test_degenerate(self):
        v = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0], ])
        f = np.array([[0, 1, 3], [0, 2, 3], ])
        ray = np.array([[0.25, 0.25, -2], [0.25, 0.25, -1], ])
        intersections, is_valid = compgeom.ray_triangle_intersection(f, v, ray)
        self.assertTrue((is_valid == [True, False]).all())
        self.assertTrue((intersections[0, :] == [0.25, 0.25, 0]).all())

    def test_parellel(self):
        v = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], ])
        f = np.array([[0, 1, 3], [0, 2, 3], ])
        ray = np.array([[0, 0, 0], [1,1,0], ])
        intersections, is_valid = compgeom.ray_triangle_intersection(f, v, ray)
        self.assertTrue((is_valid == [False, False]).all())
        self.assertTrue(np.isnan(intersections).all())

    def test_parellel2(self):
        v = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0], ])
        f = np.array([[0, 1, 3], [0, 2, 3], ])
        ray = np.array([[0, 0, 0], [1,1,0], ])
        intersections, is_valid = compgeom.ray_triangle_intersection(f, v, ray)
        self.assertTrue((is_valid == [False, False]).all())
        self.assertTrue(np.isnan(intersections).all())
