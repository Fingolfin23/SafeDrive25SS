from .. import geometric_primitives as gp
from .. import obstacles as obstcl
from .. import obstacle_aware_track_builder as oatb

import math
import numpy as np
import pytest


TEST_CENTRAL_PATH_1 = [
    [1, 1],
    [3, 2],
    [4, 4],
    [7, 5],
    [8, 8],
    [10, 8],
    [10, 6],
    [9, 3],
    [8, 1],
]


def add_extra_points(central_path, extra_points_number):
    result = [central_path[0]]
    for i in range(1, len(central_path)):
        coef = 1 / (extra_points_number + 1)
        for k in range(1, extra_points_number + 2):
            x_diff = central_path[i][0] - central_path[i - 1][0]
            y_diff = central_path[i][1] - central_path[i - 1][1]
            result.append(
                [
                    central_path[i - 1][0] + k * coef * x_diff,
                    central_path[i - 1][1] + k * coef * y_diff,
                ]
            )
    return result


@pytest.mark.parametrize(
    'central_path, track_builder, distance_to_path_bound, minimal_gap_width, extra_points_number, expected_result',
    [
        (TEST_CENTRAL_PATH_1, oatb.ObstacleAwareTrackBuilder([]), 1, 0.25, 0, TEST_CENTRAL_PATH_1),
        (TEST_CENTRAL_PATH_1, oatb.ObstacleAwareTrackBuilder([]), 1, 0.5, 0, TEST_CENTRAL_PATH_1),
        (TEST_CENTRAL_PATH_1, oatb.ObstacleAwareTrackBuilder([]), 1, 0.5, 1, add_extra_points(TEST_CENTRAL_PATH_1, 1)),
        (TEST_CENTRAL_PATH_1, oatb.ObstacleAwareTrackBuilder([]), 1, 0.5, 5, add_extra_points(TEST_CENTRAL_PATH_1, 5)),
    ]
)
def test_no_obstacles(
    central_path,
    track_builder,
    distance_to_path_bound,
    minimal_gap_width,
    extra_points_number,
    expected_result,
):
    new_central_path, new_distance_to_path_bound = track_builder.rebuild_central_path(
        central_path=central_path,
        distance_to_path_bound=distance_to_path_bound,
        minimal_gap_width=minimal_gap_width,
        extra_points_number=extra_points_number,
    )
    assert np.allclose(new_central_path, expected_result, atol=gp.EPS)
    assert math.fabs(new_distance_to_path_bound - distance_to_path_bound) < gp.EPS


@pytest.mark.parametrize(
    'central_path, track_builder, distance_to_path_bound, minimal_gap_width, extra_points_number',
    [
        (TEST_CENTRAL_PATH_1, oatb.ObstacleAwareTrackBuilder([]), 1, 3, 0),
        (TEST_CENTRAL_PATH_1, oatb.ObstacleAwareTrackBuilder([]), 1, 3, 5),
    ]
)
def test_infeasible_path_without_obstacles(
    central_path,
    track_builder,
    distance_to_path_bound,
    minimal_gap_width,
    extra_points_number,
):
    new_central_path, new_distance_to_path_bound = track_builder.rebuild_central_path(
        central_path=central_path,
        distance_to_path_bound=distance_to_path_bound,
        minimal_gap_width=minimal_gap_width,
        extra_points_number=extra_points_number,
    )
    assert np.all(np.isnan(new_central_path))
    assert math.fabs(new_distance_to_path_bound) < gp.EPS


@pytest.mark.parametrize(
    'central_path, track_builder, distance_to_path_bound, minimal_gap_width, extra_points_number, expected_result, expected_distance_to_bound',
    [
        (
            TEST_CENTRAL_PATH_1,
            oatb.ObstacleAwareTrackBuilder(
                [
                    obstcl.CircleObstacle(gp.Point(3, 2), 0.5),
                    obstcl.CircleObstacle(gp.Point(4, 4), 0.5),
                    obstcl.CircleObstacle(gp.Point(7, 5), 0.5),
                    obstcl.CircleObstacle(gp.Point(8, 8), 0.5),
                    obstcl.CircleObstacle(gp.Point(10, 8), 0.5),
                    obstcl.CircleObstacle(gp.Point(10, 6), 0.5),
                    obstcl.CircleObstacle(gp.Point(9, 3), 0.5),
                    obstcl.CircleObstacle(gp.Point(8, 1), 0.5),
                ]
            ),
            1,
            0.25,
            0,
            [
                [1, 1],
                [2.66459, 2.6708205],
                [3.3291795, 4.33541 ],
                [6.7628293, 5.7115126],
                [7.2884874, 8.237171],
                [10, 8.75],
                [10.75, 6],
                [9.711513, 2.762829 ],
                [8.67082, 0.6645898],
            ],
            0.25,
        ),
        (
            TEST_CENTRAL_PATH_1,
            oatb.ObstacleAwareTrackBuilder(
                [
                    obstcl.CircleObstacle(gp.Point(3, 2), 0.5),
                    obstcl.CircleObstacle(gp.Point(4, 4), 0.5),
                    obstcl.CircleObstacle(gp.Point(7, 5), 0.5),
                    obstcl.CircleObstacle(gp.Point(8, 8), 0.5),
                    obstcl.CircleObstacle(gp.Point(10, 8), 0.5),
                    obstcl.CircleObstacle(gp.Point(10, 6), 0.5),
                    obstcl.CircleObstacle(gp.Point(9, 3), 0.5),
                    obstcl.CircleObstacle(gp.Point(8, 1), 0.5),
                ]
            ),
            1,
            0.25,
            10,
            [
                [1, 1],
                [1.1818181, 1.0909091],
                [1.3636364, 1.1818181],
                [1.5454545, 1.2727273],
                [1.7272727, 1.3636364],
                [1.9090909, 1.4545455],
                [2.090909, 1.5454545],
                [2.2727273, 1.6363636],
                [2.4545455, 1.7272727],
                [2.3476758, 2.3955574],
                [2.4924285, 2.5605972],
                [2.66459, 2.6708205],
                [2.4394028, 2.5075715],
                [2.6044426, 2.6523242],
                [3.2727273, 2.5454545],
                [3.3636363, 2.7272727],
                [3.4545455, 2.909091],
                [3.5454545, 3.090909],
                [3.6363637, 3.2727273],
                [3.7272727, 3.4545455],
                [3.240806, 3.9250515],
                [3.2575846, 4.143935],
                [3.3291795, 4.33541],
                [4.0499306, 4.7592998],
                [4.5454545, 4.181818],
                [4.818182, 4.2727275],
                [5.090909,   4.3636365],
                [5.3636365, 4.4545455],
                [5.6363635, 4.5454545],
                [5.909091, 4.6363635],
                [6.181818, 4.7272725],
                [6.4545455, 4.818182],
                [6.504476, 5.5774813],
                [6.7628293, 5.7115126],
                [6.4225187, 5.495524],
                [7.181818, 5.5454545],
                [7.2727275, 5.818182],
                [7.3636365, 6.090909],
                [7.4545455, 6.3636365],
                [7.5454545, 6.6363635],
                [7.6363635, 6.909091],
                [7.7272725, 7.181818],
                [7.818182, 7.4545455],
                [7.2407002, 7.9500694],
                [7.2884874, 8.237171],
                [8.181818, 8.732885],
                [8.363636, 8.671587],
                [8.545455, 8],
                [8.727273, 8],
                [8.909091, 8],
                [9.090909, 8],
                [9.272727, 8],
                [9.454545, 8],
                [9.636364, 8.671587],
                [9.818182, 8.732885],
                [10, 8.75],
                [10.732885, 7.818182 ],
                [10.671587, 7.6363635],
                [10, 7.4545455],
                [10, 7.2727275],
                [10, 7.090909],
                [10, 6.909091],
                [10, 6.7272725],
                [10, 6.5454545],
                [10.671587, 6.3636365],
                [10.732885, 6.181818],
                [10.75, 6],
                [10.577481, 5.504476 ],
                [9.818182, 5.4545455],
                [9.727273, 5.181818],
                [9.636364, 4.909091],
                [9.545455, 4.6363635],
                [9.454545, 4.3636365],
                [9.363636, 4.090909],
                [9.272727, 3.8181818],
                [9.181818, 3.5454545],
                [9.759299, 3.0499303],
                [9.711513, 2.762829],
                [9.560597, 2.4924285],
                [9.395557, 2.3476758],
                [8.727273, 2.4545455],
                [8.636364, 2.2727273],
                [8.545455, 2.090909],
                [8.454545, 1.9090909],
                [8.363636, 1.7272727],
                [8.272727, 1.5454545],
                [8.759193, 1.0749485],
                [8.742415, 0.856065],
                [8.67082, 0.6645898],
            ],
            0.25,
        ),
        (
            TEST_CENTRAL_PATH_1,
            oatb.ObstacleAwareTrackBuilder(
                [
                    obstcl.CircleObstacle(gp.Point(3.5, 1.5), 0.5),
                ]
            ),
            1,
            0.25,
            0,
            [
                [1, 1],
                [2.8263931, 2.3472135],
                [4, 4],
                [7, 5],
                [8, 8],
                [10, 8],
                [10, 6],
                [9, 3],
                [8, 1],
            ],
            0.611803,
        ),
    ]
)
def test_feasible_with_obstacles(
    central_path,
    track_builder,
    distance_to_path_bound,
    minimal_gap_width,
    extra_points_number,
    expected_result,
    expected_distance_to_bound,
):
    new_central_path, new_distance_to_path_bound = track_builder.rebuild_central_path(
        central_path=central_path,
        distance_to_path_bound=distance_to_path_bound,
        minimal_gap_width=minimal_gap_width,
        extra_points_number=extra_points_number,
    )
    assert np.allclose(new_central_path, expected_result, atol=gp.EPS)
    assert math.fabs(new_distance_to_path_bound - expected_distance_to_bound) < gp.EPS
