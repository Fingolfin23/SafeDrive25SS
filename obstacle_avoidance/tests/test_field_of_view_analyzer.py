import obstacle_avoidance.geometric_primitives as gp
import obstacle_avoidance.obstacles as obstcl
import obstacle_avoidance.field_of_view_analyzer as fva

import math
import pytest


@pytest.mark.parametrize(
    'fva_sampler, n, distribuiton, expected_result',
    [
        (
            fva.FieldOfViewVectorSampler(gp.Point(0, 0), gp.Vector(1, 1), 1, math.pi / 2),
            3,
            'uniform',
            [
                gp.Vector(1, 0),
                gp.Vector(math.sqrt(2) / 2, math.sqrt(2) / 2),
                gp.Vector(0, 1),
            ]
        ),
        (
            fva.FieldOfViewVectorSampler(gp.Point(7, -2), gp.Vector(1, 0), 2, math.pi),
            3,
            'uniform',
            [
                gp.Vector(7, -4, gp.Point(7, -2)),
                gp.Vector(9, -2, gp.Point(7, -2)),
                gp.Vector(7, 0, gp.Point(7, -2)),
            ]
        ),
        (
            fva.FieldOfViewVectorSampler(gp.Point(0, 0), gp.Vector(1, 1), 1, 2 * math.pi / 3),
            10,
            'uniform',
            [
                gp.Vector(0.96592583, -0.25881905),
                gp.Vector(0.99957695, -0.02908472),
                gp.Vector(0.97934062, 0.20221757),
                gp.Vector(0.90630779, 0.42261826),
                gp.Vector(0.78441566, 0.62023549),
                gp.Vector(0.62023549, 0.78441566),
                gp.Vector(0.42261826, 0.90630779),
                gp.Vector(0.20221757, 0.97934062),
                gp.Vector(-0.02908472, 0.99957695),
                gp.Vector(-0.25881905, 0.96592583),
            ]
        ),
    ]
)
def test_field_of_view_vector_sampler(fva_sampler, n, distribuiton, expected_result):
    assert fva_sampler.sample_view_vectors(n, distribuiton) == expected_result


@pytest.mark.parametrize(
    'distance_detector, v, expected_result',
    [
        (
            fva.DistanceDetector(
                [
                    obstcl.CircleObstacle(gp.Point(1, 2), 2),
                    obstcl.CircleObstacle(gp.Point(2, 3), 2),
                    obstcl.CircleObstacle(gp.Point(2, 2), 2),
                    obstcl.CircleObstacle(gp.Point(3, 2), 2),
                    obstcl.CircleObstacle(gp.Point(10, 3), 2),
                ]
            ),
            gp.Vector(3, 2, gp.Point(0, 1)),
            gp.Vector(0.22650000, 1.07550000, gp.Point(0, 1)),
        ),
        (
            fva.DistanceDetector(
                [
                    obstcl.CircleObstacle(gp.Point(1, 2), 2),
                    obstcl.CircleObstacle(gp.Point(2, 3), 2),
                    obstcl.CircleObstacle(gp.Point(3, 2), 2),
                    obstcl.CircleObstacle(gp.Point(10, 3), 2),
                ]
            ),
            gp.Vector(3, 2, gp.Point(0, 1)),
            gp.Vector(0.93030615, 1.31010205, gp.Point(0, 1)),
        ),
        (
            fva.DistanceDetector(
                [
                    obstcl.CircleObstacle(gp.Point(1, 2), 2),
                    obstcl.CircleObstacle(gp.Point(3, 2), 2),
                    obstcl.CircleObstacle(gp.Point(10, 3), 2),
                ]
            ),
            gp.Vector(3, 2, gp.Point(0, 1)),
            gp.Vector(1.10263340, 1.36754446, gp.Point(0, 1)),
        ),
        (
            fva.DistanceDetector(
                [
                    obstcl.CircleObstacle(gp.Point(1, 2), 2),
                    obstcl.CircleObstacle(gp.Point(10, 3), 2),
                ]
            ),
            gp.Vector(3, 2, gp.Point(0, 1)),
            gp.Vector(3, 2, gp.Point(0, 1)),
        ),
        (
            fva.DistanceDetector(
                [
                    obstcl.CircleObstacle(gp.Point(10, 3), 2),
                ]
            ),
            gp.Vector(3, 2, gp.Point(0, 1)),
            gp.Vector(3, 2, gp.Point(0, 1)),
        ),
        (
            fva.DistanceDetector(
                [
                    obstcl.CircleObstacle(gp.Point(10, 3), 2),
                ]
            ),
            gp.Vector(12, 5, gp.Point(0, 1)),
            gp.Vector(8.13030615, 3.71010205, gp.Point(0, 1)),
        ),
    ]
)
def test_distance_detector(distance_detector, v, expected_result):
    assert distance_detector.get_min_distance_vector(v) == expected_result
