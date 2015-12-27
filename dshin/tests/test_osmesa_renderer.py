import unittest
import numpy as np
from dshin import osmesa_renderer


class MyTestCase(unittest.TestCase):
    def test_pixel_coordinates(self):
        vertices = np.array([[2.1, 1.9, 0], [7.9, 2.1, 0], [2.1, 8.1, 0],
                             [8.1, 8.1, 0], [7.9, 1.9, 0], [1.9, 7.9, 0]])
        P = np.eye(3, 4)
        lrtb = (0, 11 - 1, 0, 10 - 1)
        im_wh = (11, 10)
        depth = osmesa_renderer.render_depth(vertices, P, out_wh=im_wh,
                                             lrtb=lrtb, is_perspective=False)

        self.assertEqual(depth.shape, (10, 11))
        self.assertEqual((~depth.mask).sum(), 49)
        self.assertTrue((~depth.mask)[2:9, 2:9].all())
        self.assertTrue((depth[2, 2] == depth[2:9, 2:9]).all())

    def test_depth_precision(self):
        vertices = np.array([
            [1, 1, 0], [1, 8, 0], [8, 8, 0], [1, 1, 0.95], [8, 1, 0.95],
            [8, 8, 0.95], [11, 1, 0.5], [11, 8, 0.5], [18, 8, 0.5], [11, 1, -1],
            [18, 1, -1], [18, 8, -1], [11, 11, -0.5], [11, 18, -0.5],
            [18, 18, -0.5], [11, 11, 0.3], [18, 11, 0.3], [18, 18, 0.3],
            [1, 11, 0.8], [1, 18, 0.8], [8, 18, 0.8], [1, 11, -0.3],
            [8, 11, -0.3], [8, 18, -0.3],
        ])
        P = np.eye(3, 4)
        lrtb = (0, 20 - 1, 0, 20 - 1)
        im_wh = (20, 20)

        for scale in [1e-10, 1e-5, 1e-2, 1, 1e2, 1e5, 1e10]:
            scaled = vertices.copy()
            scaled[:, 2] *= scale

            depth = osmesa_renderer.render_depth(
                    scaled, P, out_wh=im_wh, lrtb=lrtb, is_perspective=False)

            expected = np.array(
                    [-1, -0.5, -0.3, 0, 0.3, 0.5, 0.8, 0.95]) * scale
            actual = np.sort(np.unique(depth.data[~np.isnan(depth.data)]))

            self.assertLess(np.power(actual - expected, 2).max(),
                            1e-14 * (scale ** 2))


if __name__ == '__main__':
    unittest.main()
