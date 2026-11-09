import unittest
import numpy as np
from seam_path import trace_path, depth_views, align_depth


class SeamPathTests(unittest.TestCase):
    def check_curve(self, truth, curved):
        rng = np.random.default_rng(4)
        cloud = np.repeat(truth, 3, axis=0)
        cloud += rng.normal(0, .001, cloud.shape)
        rng.shuffle(cloud)
        path = trace_path(cloud)
        self.assertIsNotNone(path)
        self.assertEqual(len(path) > 2, curved)
        actual = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        expected = np.linalg.norm(np.diff(truth, axis=0), axis=1).sum()
        self.assertLess(abs(actual - expected) / expected, .035)
        ends = min(np.linalg.norm(path[0]-truth[0])+np.linalg.norm(path[-1]-truth[-1]),
                   np.linalg.norm(path[-1]-truth[0])+np.linalg.norm(path[0]-truth[-1]))
        self.assertLess(ends, .04)

    def test_straight(self):
        t = np.linspace(0, 1, 600)
        self.check_curve(np.c_[t, t*0, t*0], False)

    def test_u_curve(self):
        t = np.linspace(0, np.pi, 600)
        self.check_curve(np.c_[np.cos(t), np.sin(t), t*0], True)

    def test_s_curve_3d(self):
        t = np.linspace(0, 1, 600)
        self.check_curve(np.c_[t, .2*np.sin(2*np.pi*t), .1*t], True)

    def test_broad_surface_is_not_seam(self):
        x, y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
        self.assertIsNone(trace_path(np.c_[x.ravel(), y.ravel(), x.ravel()*0]))

    def test_disconnected_components_are_not_bridged(self):
        t = np.linspace(0, 1, 200)
        line = np.c_[t, t*0, t*0]
        self.assertIsNone(trace_path(np.r_[line, line + [3, 0, 0]]))

    def test_depth_keeps_all_views(self):
        d = np.arange(3*8*12).reshape(1, 3, 8, 12, 1)
        out = depth_views(d, 3)
        self.assertEqual(out.shape, (3, 8, 12))
        np.testing.assert_array_equal(out[2], d[0, 2, :, :, 0])

    def test_portrait_depth_is_cropped_not_stretched(self):
        d = np.tile(np.arange(1036, dtype=float)[:, None], (1, 518))
        out = align_depth(d, (518, 518))
        self.assertEqual(out[0, 0], 259)
        self.assertEqual(out[-1, 0], 776)

    def test_batch_padding_is_invalid(self):
        out = align_depth(np.ones((252, 518)), (518, 518))
        self.assertTrue(np.isnan(out[0]).all())
        self.assertEqual(out[259, 259], 1)

    def test_vertical_t_joint_is_not_replaced_by_table_contact(self):
        import server
        server.np = np
        y, z = np.meshgrid(np.linspace(0, .25, 45), np.linspace(0, 1, 180))
        p = np.r_[np.c_[y.ravel()*0, y.ravel(), z.ravel()],
                  np.c_[y.ravel(), y.ravel()*0, z.ravel()]]
        p += np.random.default_rng(7).normal(0, .0005, p.shape)
        start, end, method, path = server._detect_seam_geometric(
            p, 1.1, table_n=np.array([0., 0., 1.]), table_d=0)
        self.assertTrue(method.startswith('t-joint:'))
        self.assertEqual(len(path), 2)
        self.assertGreater(abs(start[2] - end[2]), .98)
        self.assertLess(np.abs(path[:, :2]).max(), .002)


def _standing_wall_scene(plan, seed=0):
    """Flat base plate z=0 with a standing wall whose footprint is plan(t)."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(-.5, .5, 160), np.linspace(-.35, .35, 110))
    base = np.c_[x.ravel(), y.ravel(), x.ravel()*0]
    t = np.linspace(-.4, .4, 300)
    T, H = np.meshgrid(np.arange(len(t)), np.linspace(0, .25, 40))
    root = np.c_[t, plan(t), t*0]
    wall = np.c_[root[T.ravel(), :2], H.ravel()]
    p = np.r_[base, wall] + rng.normal(0, .001, (len(base) + len(wall), 3))
    return p, root


class CurvedJointTests(unittest.TestCase):
    def setUp(self):
        import server
        server.np = np
        self.server = server

    def check(self, p, root, closed=False):
        from scipy.spatial import cKDTree
        start, end, method, path = self.server._detect_seam_geometric(p, 1.4)
        self.assertIsNotNone(path, method)
        self.assertTrue(method.startswith('curved-joint:'), method)
        self.assertLess(cKDTree(root).query(path)[0].max(), .006)
        self.assertLess(np.percentile(cKDTree(path).query(root)[0], 95), .008)
        expected = np.linalg.norm(np.diff(root, axis=0), axis=1).sum()
        if closed:
            expected += np.linalg.norm(root[-1] - root[0])
        actual = self.server._path_arc_length(path)
        self.assertLess(abs(actual - expected) / expected, .02)

    def test_c_shaped_wall_on_plate(self):
        self.check(*_standing_wall_scene(lambda t: .15*np.cos(np.pi*t/.8) - .05))

    def test_s_shaped_wall_on_plate(self):
        self.check(*_standing_wall_scene(lambda t: .12*np.sin(2*np.pi*t/.8)))

    def test_pipe_on_plate_closed_loop(self):
        rng = np.random.default_rng(1)
        x, y = np.meshgrid(np.linspace(-.5, .5, 160), np.linspace(-.35, .35, 110))
        a, h = np.meshgrid(np.linspace(0, 2*np.pi, 400, endpoint=False),
                           np.linspace(0, .25, 40))
        p = np.r_[np.c_[x.ravel(), y.ravel(), x.ravel()*0],
                  np.c_[.2*np.cos(a.ravel()), .2*np.sin(a.ravel()), h.ravel()]]
        p += rng.normal(0, .001, p.shape)
        a1 = np.linspace(0, 2*np.pi, 400, endpoint=False)
        self.check(p, np.c_[.2*np.cos(a1), .2*np.sin(a1), a1*0], closed=True)

    def test_unevenly_seen_pipe_is_traced_as_full_loop(self):
        # Cameras see one side of the pipe far better than the other, and the
        # reconstructed wall is a few % thick. The open tracer still accepts
        # that footprint, but keeps only part of the ring.
        rng = np.random.default_rng(1)
        x, y = np.meshgrid(np.linspace(-.5, .5, 160), np.linspace(-.35, .35, 110))
        plate = np.c_[x.ravel(), y.ravel(), x.ravel()*0]
        a = rng.uniform(0, 2*np.pi, 30000)
        a = a[rng.random(len(a)) < .25 + .75*(1 + np.cos(a))/2]
        r = .2*(1 + rng.uniform(-.04, .06, len(a)))
        wall = np.c_[r*np.cos(a), r*np.sin(a), rng.uniform(0, .25, len(a))]
        p = np.r_[plate, wall] + rng.normal(0, .002, (len(plate) + len(a), 3))
        path = self.server._detect_seam_geometric(p, 1.4)[3]
        self.assertTrue(np.allclose(path[0], path[-1]))
        expected = 2*np.pi*.2
        self.assertLess(abs(self.server._path_arc_length(path) - expected) / expected, .03)

    def test_thin_base_plate_survives_table_crop(self):
        # A large background inflates the scene diagonal; the workpiece cut must
        # follow the table's measured noise, not that diagonal, or a 12 mm base
        # plate (and the weld root on it) is cropped away with the table.
        rng = np.random.default_rng(5)
        x, y = np.meshgrid(np.linspace(-.7, .7, 200), np.linspace(-.7, .7, 200))
        table = np.c_[x.ravel(), y.ravel(), rng.normal(0, .001, x.size)]
        far = rng.uniform(-3, 3, (3000, 3)) + [0, 0, 3.5]
        table_fit = self.server._fit_table(np.r_[table, far])
        self.assertGreater(table_fit['thresh'] * 3, .012)       # old cut: plate lost
        self.assertLess(table_fit['part_cut'], .012)
        self.assertGreater(table_fit['part_cut'], .003)          # table noise stays out

    def test_upright_t_on_plate_prefers_member_seam(self):
        # Upright T standing on a base plate: the seam between the two T plates
        # wins over the T-shaped footprint on the plate.
        rng = np.random.default_rng(6)
        x, y = np.meshgrid(np.linspace(-.5, .5, 160), np.linspace(-.35, .35, 110))
        s, h = rng.uniform(-.15, .15, 4000), rng.uniform(0, .15, 2000)
        p = np.r_[np.c_[x.ravel(), y.ravel(), x.ravel()*0],
                  np.c_[s, s*0, rng.uniform(0, .3, len(s))],
                  np.c_[h*0, h, rng.uniform(0, .3, len(h))]]
        p += rng.normal(0, .001, p.shape)
        start, end, method, path = self.server._detect_seam_geometric(
            p, 1.4, table_n=np.array([0., 0., 1.]), table_d=0)
        self.assertTrue(method.startswith('t-joint:members'), method)
        self.assertLess(np.abs(path[:, :2]).max(), .006)
        self.assertGreater(abs(start[2] - end[2]), .25)

    def test_straight_t_joint_stops_at_wall_ends(self):
        p, root = _standing_wall_scene(lambda t: t*0)
        start, end, method, path = self.server._detect_seam_geometric(p, 1.4)
        self.assertTrue(method.startswith('t-joint:'), method)
        self.assertEqual(len(path), 2)
        self.assertAlmostEqual(self.server._path_arc_length(path), .8, delta=.01)

    def test_flat_plate_alone_has_no_seam(self):
        x, y = np.meshgrid(np.linspace(-.5, .5, 160), np.linspace(-.35, .35, 110))
        p = np.c_[x.ravel(), y.ravel(), x.ravel()*0]
        p += np.random.default_rng(2).normal(0, .001, p.shape)
        self.assertIsNone(self.server._detect_seam_geometric(p, 1.4)[3])

    def _thick_capped(self, plan, thick=.02):
        """Wall with two faces AND end faces (the wrap-around seen on real T-joints)."""
        rng = np.random.default_rng(3)
        x, y = np.meshgrid(np.linspace(-.5, .5, 160), np.linspace(-.35, .35, 110))
        parts = [np.c_[x.ravel(), y.ravel(), x.ravel()*0]]
        t = np.linspace(-.4, .4, 300)
        c = np.c_[t, plan(t)]
        tg = np.gradient(c, axis=0); tg /= np.linalg.norm(tg, axis=1)[:, None]
        nrm = np.c_[-tg[:, 1], tg[:, 0]]
        h = np.linspace(0, .25, 40)
        for o in (-thick/2, thick/2):
            T, H = np.meshgrid(np.arange(len(t)), h)
            parts.append(np.c_[(c + o*nrm)[T.ravel()], H.ravel()])
        for e in (0, -1):
            o, H = np.meshgrid(np.linspace(-thick/2, thick/2, 8), h)
            parts.append(np.c_[c[e] + o.ravel()[:, None]*nrm[e], H.ravel()])
        p = np.vstack(parts)
        return p + rng.normal(0, .001, p.shape), np.c_[c, t*0]

    def test_thick_straight_plate_end_faces_stay_straight(self):
        p, root = self._thick_capped(lambda t: t*0)
        start, end, method, path = self.server._detect_seam_geometric(p, 1.4)
        self.assertTrue(method.startswith('t-joint:'), method)
        self.assertAlmostEqual(self.server._path_arc_length(path), .8, delta=.03)

    def test_thick_curved_plate_end_faces_trimmed(self):
        p, root = self._thick_capped(lambda t: .12*np.sin(2*np.pi*t/.8))
        start, end, method, path = self.server._detect_seam_geometric(p, 1.4)
        self.assertTrue(method.startswith('curved-joint:'), method)
        expected = np.linalg.norm(np.diff(root, axis=0), axis=1).sum()
        self.assertLess(abs(self.server._path_arc_length(path) - expected) / expected, .04)


if __name__ == '__main__':
    unittest.main()
