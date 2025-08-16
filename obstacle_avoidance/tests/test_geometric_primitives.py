from .. import geometric_primitives as gp

import math
import pytest


@pytest.mark.parametrize(
    'operation_type, p1, p2, expected_result',
    [
        ('add', gp.Point(1, 1), gp.Point(2, 8), gp.Point(3, 9)),
        ('sub', gp.Point(1, 1), gp.Point(2, 8), gp.Point(-1, -7)),
    ],
)
def test_point_operations(operation_type, p1, p2, expected_result):
    result = None

    if operation_type == 'add':
        result = p1 + p2
    elif operation_type == 'sub':
        result = p1 - p2
    else:
        pass

    assert result == expected_result


@pytest.mark.parametrize(
    'x, y, origin,expected_result',
    [
        (1, 1, gp.Point(0, 0), math.sqrt(1 + 1)),
        (1, 1, gp.Point(0, 1), 1),
        (-2, -5, gp.Point(2, -1), math.sqrt((-2 - 2) ** 2 + (-5 + 1) ** 2)),
    ]
)
def test_vector_norm(x, y, origin, expected_result):
    assert math.fabs(gp.Vector(x, y, origin).norm() - expected_result) < gp.EPS


@pytest.mark.parametrize(
    'v, expected_result',
    [
        (gp.Vector(2, 3, gp.Point(0, 0)), gp.Vector(2, 3)),
        (gp.Vector(2, 3, gp.Point(1, -1)), gp.Vector(1, 4)),
    ]
)
def test_position(v,expected_result):
    assert v.get_position() == expected_result


@pytest.mark.parametrize(
    'v, phi, expected_result',
    [
        (gp.Vector(2, 3, gp.Point(0, 0)), math.pi, gp.Vector(-2, -3)),
        (gp.Vector(2, 3, gp.Point(0, 0)), -math.pi, gp.Vector(-2, -3)),
        (gp.Vector(2, 3, gp.Point(0, 0)), math.pi / 2.0, gp.Vector(-3, 2)),
        (gp.Vector(2, 3, gp.Point(0, 0)), -math.pi / 2.0, gp.Vector(3, -2)),

        (gp.Vector(2, 3, gp.Point(-1, 1)), math.pi, gp.Vector(-4, -1, gp.Point(-1, 1))),
        (gp.Vector(2, 3, gp.Point(-1, 1)), -math.pi, gp.Vector(-4, -1, gp.Point(-1, 1))),
        (gp.Vector(2, 3, gp.Point(-1, 1)), math.pi / 2.0, gp.Vector(-3, 4, gp.Point(-1, 1))),
        (gp.Vector(2, 3, gp.Point(-1, 1)), -math.pi / 2.0, gp.Vector(1, -2, gp.Point(-1, 1))),

        (gp.Vector(1, 0, gp.Point(0, 0)), math.pi / 3.0, gp.Vector(0.5, math.sqrt(3.0) / 2.0)),
    ]
)
def test_rotate_by(v, phi, expected_result):
    assert v.rotate_by(phi) == expected_result


@pytest.mark.parametrize(
    'v, new_origin, expected_result',
    [
        (gp.Vector(2, 3, gp.Point(0, 0)), gp.Point(3, 2), gp.Vector(5, 5, gp.Point(3, 2))),
        (gp.Vector(2, 3, gp.Point(1, -1)), gp.Point(3, 2), gp.Vector(4, 6, gp.Point(3, 2))),
    ]
)
def test_shift_to_new_origin(v, new_origin, expected_result):
    assert v.shift_to_new_origin(new_origin) == expected_result


@pytest.mark.parametrize(
    'v, x, y, expected_result',
    [
        (gp.Vector(2, 3, gp.Point(0, 0)), 2, 2, gp.Vector(2, 2, gp.Point(0, -1))),
        (gp.Vector(2, 3, gp.Point(1, -1)), 0, 0, gp.Vector(0, 0, gp.Point(-1, -4))),
    ]
)
def test_shift_to_new_coordinates(v, x, y, expected_result):
    assert v.shift_to_new_coordinates(x, y) == expected_result


@pytest.mark.parametrize(
    'v, p, expected_result',
    [
        (
            gp.Vector(3.5, 2, gp.Point(1, -1)),
            gp.Point(2, 3),
            gp.Vector(
                2.1229508196721314,
                3.1475409836065573,
                gp.Point(-0.3770491803278688, 0.14754098360655732),
            )
        )
    ]
)
def test_shift_parallel(v, p, expected_result):
    assert v.shift_parallel(p) == expected_result


@pytest.mark.parametrize(
    'v, new_length, expected_result',
    [
        (gp.Vector(3, 4, gp.Point(0, 0)), 10, gp.Vector(6, 8)),
        (gp.Vector(4, 3, gp.Point(1, -1)), 10, gp.Vector(7, 7, gp.Point(1, -1))),
    ]
)
def test_update_length(v, new_length, expected_result):
    assert v.update_length(new_length) == expected_result


@pytest.mark.parametrize(
    'v, coef, expected_result',
    [
        (gp.Vector(3, 4), 10, gp.Vector(30, 40)),
        (gp.Vector(3, 4, gp.Point(1, -1)), -5, gp.Vector(-9, -26, gp.Point(1, -1))),
        (gp.Vector(3, 4, gp.Point(1, -1)), 0, gp.Vector(1, -1, gp.Point(1, -1))),
    ]
)
def test_vector_mul_by_coef(v, coef, expected_result):
    assert v * coef == expected_result


@pytest.mark.parametrize(
    'v, expected_result',
    [
        (gp.Vector(3, 4), gp.Vector(-3, -4)),
        (gp.Vector(3, 4, gp.Point(1, -1)), gp.Vector(-1, -6, gp.Point(1, -1))),
    ]
)
def test_vector_neg(v, expected_result):
    assert -v == expected_result


@pytest.mark.parametrize(
    'v1, v2, expected_result',
    [
        (gp.Vector(3, 4), gp.Vector(-5, 3), gp.Vector(-2, 7)),
        (
            gp.Vector(3, 4, gp.Point(1, -1)),
            gp.Vector(-5, 3),
            gp.Vector(-2, 7, gp.Point(1, -1)),
        ),
        (
            gp.Vector(3, 4, gp.Point(1, -1)),
            gp.Vector(-5, 3, gp.Point(8, 9)),
            gp.Vector(-10, -2, gp.Point(1, -1)),
        ),
        (
            gp.Vector(-5, 3, gp.Point(8, 9)),
            gp.Vector(3, 4, gp.Point(1, -1)),
            gp.Vector(-3, 8, gp.Point(8, 9)),
        ),
    ]
)
def test_vector_add(v1, v2, expected_result):
    assert v1 + v2 == expected_result


@pytest.mark.parametrize(
    'v1, v2, expected_result',
    [
        (gp.Vector(3, 4), gp.Vector(-5, 3), gp.Vector(8, 1)),
        (
            gp.Vector(3, 4, gp.Point(1, -1)),
            gp.Vector(-5, 3),
            gp.Vector(8, 1, gp.Point(1, -1)),
        ),
        (
            gp.Vector(3, 4, gp.Point(1, -1)),
            gp.Vector(-5, 3, gp.Point(8, 9)),
            gp.Vector(16, 10, gp.Point(1, -1)),
        ),
        (
            gp.Vector(-5, 3, gp.Point(8, 9)),
            gp.Vector(3, 4, gp.Point(1, -1)),
            gp.Vector(-7, -2, gp.Point(8, 9)),
        ),
    ]
)
def test_vector_sub(v1, v2, expected_result):
    assert v1 - v2 == expected_result


@pytest.mark.parametrize(
    'v1, v2, expected_result',
    [
        (
            gp.Vector(1, 0),
            gp.Vector(0, 1),
            math.pi / 2,
        ),
        (
            gp.Vector(0, 1),
            gp.Vector(1, 0),
            -math.pi / 2,
        ),
        (
            gp.Vector(1, -1),
            gp.Vector(1, 1),
            math.pi / 2,
        ),
        (   
            gp.Vector(1, 1),
            gp.Vector(1, -1),
            -math.pi / 2,
        ),

        (
            gp.Vector(2, 1, gp.Point(1, 1)),
            gp.Vector(-2, 4, gp.Point(-2, 3)),
            math.pi / 2,
        ),
        (
            gp.Vector(-2, 4, gp.Point(-2, 3)),
            gp.Vector(2, 1, gp.Point(1, 1)),
            -math.pi / 2,
        ),
        (
            gp.Vector(2, 0, gp.Point(1, 1)),
            gp.Vector(-1, 4, gp.Point(-2, 3)),
            math.pi / 2,
        ),
        (   
            gp.Vector(2, 2, gp.Point(1, 1)),
            gp.Vector(-1, 2, gp.Point(-2, 3)),
            -math.pi / 2,
        ),
    ]
)
def test_vector_angle_between(v1, v2, expected_result):
    assert math.fabs(v1.get_angle_between(v2) - expected_result) < gp.EPS
