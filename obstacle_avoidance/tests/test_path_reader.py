from .. import geometric_primitives as gp
from .. import path_reader

import pytest
import numpy as np


@pytest.mark.parametrize(
    'filename, x, y, dist_to_rb, dist_to_lb, rb_raw, lb_raw',
    [
        (
            'MonzaSmall.csv',  # filename
            np.asarray([-0.320123, 0.168262, 0.656139, 1.143549, 1.630535, 2.117138, 2.603399, 3.089362, 3.575067]),  # x
            np.asarray([1.087714, 6.062191, 11.036647, 16.011082, 20.985493, 25.959881, 30.934243, 35.908579, 40.882887]),  # y
            np.asarray([5.739, 5.735, 5.731, 5.727, 5.723, 5.719, 5.715, 5.711, 5.707]),  # dist_to_rb
            np.asarray([5.932, 5.929, 5.926, 5.923, 5.920, 5.917, 5.914, 5.911, 5.908]),  # dist_to_lb
            np.asarray([
                [5.39141639, 0.52696557],
                [5.87587695, 5.50240837],
                [6.35982481, 10.47778283],
                [6.84330074, 15.45308643],
                [7.32634803, 20.42831899],
                [7.80900766, 25.40348116],
                [8.29132034, 30.37856696],
                [8.77333051, 35.35358081],
                [9.25505444, 40.32827753],
            ]),  # rb_raw
            np.asarray([
                [-6.2237389, 1.66732015],
                [-5.73242658, 6.64090961],
                [-5.24161743, 11.61452679],
                [-4.7512702, 16.58817433],
                [-4.26134219, 21.56184634],
                [-3.77179237, 26.5355442],
                [-3.28257944, 31.50926804],
                [-2.79365986, 36.4830133],
                [-2.30496906, 41.45702976],
            ]),  # lb_raw
        )
    ]
)
def test_path_reader(
    filename, x, y, dist_to_rb, dist_to_lb, rb_raw, lb_raw
):
    pr = path_reader.PathReader(filename)
    assert np.allclose(pr.x_coordinates, x, atol=gp.EPS)
    assert np.allclose(pr.y_coordinates, y, atol=gp.EPS)
    assert np.allclose(pr.dist_to_right_bound, dist_to_rb, atol=gp.EPS)
    assert np.allclose(pr.dist_to_left_bound, dist_to_lb, atol=gp.EPS)
    print(pr.right_bound_raw)
    print(pr.left_bound_raw)
    assert np.allclose(pr.right_bound_raw, rb_raw, atol=gp.EPS)
    assert np.allclose(pr.left_bound_raw, lb_raw, atol=gp.EPS)
