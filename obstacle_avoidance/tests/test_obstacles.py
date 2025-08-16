from .. import geometric_primitives as gp
from .. import obstacles as obstcl
from .. import path_reader

import pytest


@pytest.mark.parametrize(
    'v, obstacle, expected_result',
    [
        (
            gp.Vector(5, 0, gp.Point(0, -5)),
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            None,
        ),  # vector is too far from obstacle
        (
            gp.Vector(-2, -2),
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            None,
        ),  # wrong direction
        (
            gp.Vector(6, 6, gp.Point(5, 5)),
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            None,
        ),  # wrong direction
        (
            gp.Vector(1, 1),
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            None,
        ),  # vector is too short
        (
            gp.Vector(3, 4, gp.Point(2, 3)),
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            None,
        ),  # vector is too short
        (
            gp.Vector(2, 2),
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            gp.Vector(1.17712434447, 1.17712434447),
        ),  # has intersection
        (
            gp.Vector(4, 5, gp.Point(2, 3)),
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            gp.Vector(3.41421356237, 4.41421356237, gp.Point(2, 3)),
        ),  # has intersection
        (
            gp.Vector(6, 5, gp.Point(-4, -7)),
            obstcl.CircleObstacle(gp.Point(2, 3), 2),
            gp.Vector(2.80916366965, 1.17099640358, gp.Point(-4, -7)),
        ),  # has intersection
    ]
)
def test_circle_obstacle_intersection_with_vector(
    v, obstacle, expected_result
):
    assert obstacle.get_intersection_with_vector(
        v,
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


@pytest.mark.parametrize(
    'obstacle, p, length, expected_result',
    [
        (
            obstcl.CircleObstacle(gp.Point(7, 5), 3),
            gp.Point(1, 2),
            10,
            (
                gp.Vector(7.0, 10.0, gp.Point(1, 2)),
                gp.Vector(11.0, 2.0, gp.Point(1, 2)),
            )
        )
    ]
)
def test_circle_obstacle_update_radius(
    obstacle, p, length, expected_result
):
    assert obstacle.get_tangent_vectors(p, length) == expected_result


@pytest.mark.parametrize(
    'obstacle, new_origin, expected_result',
    [
        (
            obstcl.SegmentObstacle(gp.Point(1, 1), gp.Point(4, 5)),
            gp.Point(2, 3),
            obstcl.SegmentObstacle(gp.Point(-1, -2), gp.Point(2, 2)),
        ),
    ]
)
def test_segment_obstacle_shift_to_new_origin(
    obstacle, new_origin, expected_result
):
    assert obstacle.shift_to_new_origin(new_origin) == expected_result


@pytest.mark.parametrize(
    'obstacle, p, length, expected_result',
    [
        (
            obstcl.SegmentObstacle(gp.Point(1, 1), gp.Point(4, 5)),
            gp.Point(1, 2),
            10,
            (
                gp.Vector(1.0, -8.0, gp.Point(1, 2)),
                gp.Vector(8.07106781, 9.07106781, gp.Point(1, 2)),
            )
        )
    ]
)
def test_segment_obstacle_get_tangent_vectors(
    obstacle, p, length, expected_result
):
    assert obstacle.get_tangent_vectors(p, length) == expected_result


@pytest.mark.parametrize(
    'v, obstacle, expected_result',
    [
        (
            gp.Vector(4, 2, gp.Point(1, 2)),
            obstcl.SegmentObstacle(gp.Point(1, 1), gp.Point(4, 5)),
            gp.Vector(1.75, 2, gp.Point(1, 2)),
        ),
        (
            gp.Vector(4, 6, gp.Point(1, 2)),
            obstcl.SegmentObstacle(gp.Point(1, 1), gp.Point(4, 5)),
            None,
        ),
        (
            gp.Vector(2.5, 3, gp.Point(1, 1)),
            obstcl.SegmentObstacle(gp.Point(1, 1), gp.Point(4, 5)),
            None,
        ),
        (
            gp.Vector(2, 0),
            obstcl.SegmentObstacle(gp.Point(1, 1), gp.Point(4, 5)),
            None,
        ),
    ]
)
def test_segment_obstacle_intersection_with_vector(
    v, obstacle, expected_result
):
    assert obstacle.get_intersection_with_vector(
        v,
    ) == expected_result


@pytest.mark.parametrize(
    'filename, rb_segments, lb_segments',
    [
        (
            'MonzaSmall.csv',  # filename
            [
                obstcl.SegmentObstacle(
                    gp.Point(5.391416389598579, 0.5269655706095135),
                    gp.Point(5.87587694917424, 5.5024083707922475),
                ),
                obstcl.SegmentObstacle(
                    gp.Point(5.87587694917424, 5.5024083707922475),
                    gp.Point(6.359824811505467, 10.477782827207337),
                ),
                obstcl.SegmentObstacle(
                    gp.Point(6.359824811505467, 10.477782827207337),
                    gp.Point(6.8433007440265285, 15.453086429707053),
                ),
                obstcl.SegmentObstacle(
                    gp.Point(6.8433007440265285, 15.453086429707053),
                    gp.Point(7.3263480343632885, 20.428318989858397),
                ),
                obstcl.SegmentObstacle(
                    gp.Point(7.3263480343632885, 20.428318989858397),
                    gp.Point(7.809007659040329, 25.40348115771377),
                ),
                obstcl.SegmentObstacle(
                    gp.Point(7.809007659040329, 25.40348115771377),
                    gp.Point(8.291320337493506, 30.378566962649014),
                ),
                obstcl.SegmentObstacle(
                    gp.Point(8.291320337493506, 30.378566962649014),
                    gp.Point(8.773330509037885, 35.353580812376265),
                ),
                obstcl.SegmentObstacle(
                    gp.Point(8.773330509037885, 35.353580812376265),
                    gp.Point(9.255054441967994, 40.328277534622885),
                ),
            ],  # rb_segments
            [
                obstcl.SegmentObstacle(
                    gp.Point(-6.2237389015680025, 1.6673201479603366),
                    gp.Point(-5.732426584769671, 6.640909606551485),
                ),
                obstcl.SegmentObstacle(
                    gp.Point(-5.732426584769671, 6.640909606551485),
                    gp.Point(-5.241617433254476, 11.614526792003025),
                ),
                obstcl.SegmentObstacle(
                    gp.Point(-5.241617433254476, 11.614526792003025),
                    gp.Point(-4.751270203748756, 16.588174328067943),
                ),
                obstcl.SegmentObstacle(
                    gp.Point(-4.751270203748756, 16.588174328067943),
                    gp.Point(-4.261342190884268, 21.561846335669806),
                ),
                obstcl.SegmentObstacle(
                    gp.Point(-4.261342190884268, 21.561846335669806),
                    gp.Point(-3.771792367641479, 26.53554420454758),
                ),
                obstcl.SegmentObstacle(
                    gp.Point(-3.771792367641479, 26.53554420454758),
                    gp.Point(-3.282579440933786, 31.509268036726812),
                ),
                obstcl.SegmentObstacle(
                    gp.Point(-3.282579440933786, 31.509268036726812),
                    gp.Point(-2.7936598625324707, 36.48301329995516),
                ),
                obstcl.SegmentObstacle(
                    gp.Point(-2.7936598625324707, 36.48301329995516),
                    gp.Point(-2.3049690622300525, 41.45702975827019),
                ),
            ],  # lb_segments
        )
    ]
)
def test_track_bound_obstacle_init(
    filename, rb_segments, lb_segments
):
    pr = path_reader.PathReader(filename)
    tbo = obstcl.TrackBoundObstacle(pr)
    assert tbo.left_bound_segments == lb_segments
    assert tbo.right_bound_segments == rb_segments


@pytest.mark.parametrize(
    'filename, v, expected_result',
    [
        (
            'MonzaSmall.csv',
            gp.Vector(-40, 20),
            gp.Vector(-6.087756631948552, 3.043878315974276),
        ),
        (
            'MonzaSmall.csv',
            gp.Vector(40, 20),
            gp.Vector(5.613394646599156, 2.806697323299578),
        ),
        (
            'MonzaSmall.csv',
            gp.Vector(20, -20),
            gp.Vector(20, -20),
        ),
        (
            'MonzaSmall.csv',
            gp.Vector(20, 40),
            gp.Vector(6.630232378176908, 13.260464756353816),
        ),
        (
            'MonzaSmall.csv',
            gp.Vector(20, 15),
            gp.Vector(5.760804170252527, 4.320603127689395),
        ),
    ]
)
def test_track_bound_obstacle_intersection_with_vector(
    filename, v, expected_result
):
    pr = path_reader.PathReader(filename)
    tbo = obstcl.TrackBoundObstacle(pr)
    assert tbo.get_intersection_with_vector(v) == expected_result
