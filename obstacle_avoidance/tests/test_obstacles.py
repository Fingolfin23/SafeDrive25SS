from .. import geometric_primitives as gp
from .. import obstacles as obstcl

import pytest


@pytest.mark.parametrize(
    'v, is_rectangularized, obstacle, expected_result',
    [
        (
            gp.Vector(5, 0, gp.Point(0, -5)),
            False,
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            None,
        ),  # vector is too far from obstacle
        (
            gp.Vector(-2, -2),
            False,
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            None,
        ),  # wrong direction
        (
            gp.Vector(6, 6, gp.Point(5, 5)),
            False,
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            None,
        ),  # wrong direction
        (
            gp.Vector(1, 1),
            False,
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            None,
        ),  # vector is too short
        (
            gp.Vector(3, 4, gp.Point(2, 3)),
            False,
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            None,
        ),  # vector is too short
        (
            gp.Vector(2, 2),
            False,
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            gp.Vector(1.17712434447, 1.17712434447),
        ),  # has intersection
        (
            gp.Vector(4, 5, gp.Point(2, 3)),
            False,
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            gp.Vector(3.41421356237, 4.41421356237, gp.Point(2, 3)),
        ),  # has intersection
        (
            gp.Vector(6, 5, gp.Point(-4, -7)),
            False,
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            gp.Vector(2.80916366965, 1.17099640358, gp.Point(-4, -7)),
        ),  # has intersection
        (
            gp.Vector(2, 2),
            True,
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            gp.Vector(1.08578643763, 1.08578643763),
        ),  # has intersection
        (
            gp.Vector(6, 5, gp.Point(-4, -7)),
            True,
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            gp.Vector(2.09668038100, 0.31601645720, gp.Point(-4, -7)),
        ),  # has intersection
    ]
)
def test_circle_obstacle_intersection_with_vector(
    v, is_rectangularized, obstacle, expected_result
):
    assert obstacle.get_intersection_with_vector(
        v,
        is_rectangularized=is_rectangularized,
    ) == expected_result


@pytest.mark.parametrize(
    'obstacle, new_origin, expected_result',
    [
        (
            obstcl.CircleObstacle(gp.Point(1, 1), 3),
            gp.Point(2, 3),
            obstcl.CircleObstacle(gp.Point(-1, -2), 3),
        ),
        (
            obstcl.CircleObstacle(gp.Point(3, 4), 5),
            gp.Point(2, 3),
            obstcl.CircleObstacle(gp.Point(1, 1), 5),
        ),
    ]
)
def test_circle_obstacle_shift_to_new_origin(
    obstacle, new_origin, expected_result
):
    assert obstacle.shift_to_new_origin(new_origin) == expected_result


@pytest.mark.parametrize(
    'obstacle, new_radius, expected_result',
    [
        (
            obstcl.CircleObstacle(gp.Point(1, 1), 3),
            5,
            obstcl.CircleObstacle(gp.Point(1, 1), 5),
        ),
        (
            obstcl.CircleObstacle(gp.Point(3, 4), 5),
            1,
            obstcl.CircleObstacle(gp.Point(3, 4), 1),
        ),
    ]
)
def test_circle_obstacle_update_radius(
    obstacle, new_radius, expected_result
):
    assert obstacle.update_radius(new_radius) == expected_result
