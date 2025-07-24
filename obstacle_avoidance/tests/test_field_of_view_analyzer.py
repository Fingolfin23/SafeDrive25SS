from .. import geometric_primitives as gp
from .. import obstacles as obstcl
from .. import field_of_view_analyzer as fva
from .. import path_reader

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
            (gp.Vector(0.22650000, 1.07550000, gp.Point(0, 1)), 2),
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
            (gp.Vector(0.93030615, 1.31010205, gp.Point(0, 1)), 1),
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
            (gp.Vector(1.10263340, 1.36754446, gp.Point(0, 1)), 1),
        ),
        (
            fva.DistanceDetector(
                [
                    obstcl.CircleObstacle(gp.Point(1, 2), 2),
                    obstcl.CircleObstacle(gp.Point(10, 3), 2),
                ]
            ),
            gp.Vector(3, 2, gp.Point(0, 1)),
            (gp.Vector(3, 2, gp.Point(0, 1)), 0),
        ),
        (
            fva.DistanceDetector(
                [
                    obstcl.CircleObstacle(gp.Point(10, 3), 2),
                ]
            ),
            gp.Vector(3, 2, gp.Point(0, 1)),
            (gp.Vector(3, 2, gp.Point(0, 1)), None),
        ),
        (
            fva.DistanceDetector(
                [
                    obstcl.CircleObstacle(gp.Point(10, 3), 2),
                ]
            ),
            gp.Vector(12, 5, gp.Point(0, 1)),
            (gp.Vector(8.13030615, 3.71010205, gp.Point(0, 1)), 0),
        ),
    ]
)
def test_distance_detector(distance_detector, v, expected_result):
    assert distance_detector.get_min_distance_vector(v) == expected_result


@pytest.mark.parametrize(
    'tangent_vectors_extractor, view_point, distance, expected_result',
    [
        (
            fva.FieldOfViewTangentVectorsExtractor(
                [
                    obstcl.CircleObstacle(gp.Point(1, 2), 2),
                    obstcl.CircleObstacle(gp.Point(2, 3), 2),
                    obstcl.CircleObstacle(gp.Point(2, 2), 2),
                    obstcl.CircleObstacle(gp.Point(3, 2), 2),
                    obstcl.CircleObstacle(gp.Point(10, 3), 2),
                ]
            ),
            gp.Point(6, 8),
            10,
            [ 
                (gp.Vector(1.77882423, -1.06541091, gp.Point(6, 8)), 0),
                (gp.Vector(-2.15560199, 2.21327760, gp.Point(6, 8)), 0),
                (gp.Vector(2.50462192, -1.36922259, gp.Point(6, 8)), 1),
                (gp.Vector(-2.37342685, 2.53321642, gp.Point(6, 8)), 1),
                (gp.Vector(2.97830520, -1.53254218, gp.Point(6, 8)), 2),
                (gp.Vector(-1.63707940, 1.54438088, gp.Point(6, 8)), 2),
                (gp.Vector(4.39791717, -1.87083231, gp.Point(6, 8)), 3),
                (gp.Vector(-0.93541615, 0.79583435, gp.Point(6, 8)), 3),
                (gp.Vector(14.37342685, 2.53321642, gp.Point(6, 8)), 4),
                (gp.Vector(9.49537807, -1.36922259, gp.Point(6, 8)), 4),
            ]
        ),
    ]
)
def test_fov_tangent_vectors_extractor(
    tangent_vectors_extractor, view_point, distance, expected_result
):
    assert tangent_vectors_extractor.extract_tangtent_vectors_list(
        view_point, distance
    ) == expected_result


@pytest.mark.parametrize(
    'v1, v2, expected_result',
    [
        (gp.Vector(0, 4), gp.Vector(3, 0), 5),
        (gp.Vector(0, 1), gp.Vector(1, 0), math.sqrt(2)),
        (gp.Vector(-1, 1), gp.Vector(1, 0), math.sqrt(2 ** 2 + 1 ** 2)),
    ]
)
def test_gap_params_gap_width(v1, v2, expected_result):
    assert math.fabs(fva.GapParams(v1, v2).get_gap_width() - expected_result) < gp.EPS


@pytest.mark.parametrize(
    'v1, v2, v1_, v2_',
    [
        (
            gp.Vector(0, 4),
            gp.Vector(3, 0),
            gp.Vector(0, 3),
            gp.Vector(3, 0),
        ),
        (
            gp.Vector(0, 1),
            gp.Vector(1, 0),
            gp.Vector(0, 1),
            gp.Vector(1, 0),
        ),
        (
            gp.Vector(-1, 1),
            gp.Vector(1, 0),
            gp.Vector(-math.sqrt(0.5), math.sqrt(0.5)),
            gp.Vector(1, 0)
        ),
    ]
)
def test_gap_params_shrink_extra_vector_length(v1, v2, v1_, v2_):
    gap_params = fva.GapParams(v1, v2).shrink_extra_vector_length()
    assert gap_params.left_bound == v1_
    assert gap_params.right_bound == v2_


@pytest.mark.parametrize(
    'path_filename, obstacles, fov_sampler_count, view_point, direction, distance, view_angle, minimal_gap_width, expected_result',
    [
        (
            'MonzaSmall.csv',  # path_filename
            [],  # obstacles
            100,  # fov_sampler_count
            gp.Point(0, 0),  # view_point
            gp.Vector(0, 1),  # direction
            50,  # distance
            math.pi / 2,  # view_angle
            2,  # minimal_gap_width
            fva.GapParams(
                gp.Vector(-5.81410214, 5.81410214),
                gp.Vector(5.81410214, 5.81410214),
            ),  # expected_result
        ),
        (
            'MonzaSmall.csv',  # path_filename
            [],  # obstacles
            100,  # fov_sampler_count
            gp.Point(2, 2),  # view_point
            gp.Vector(1, 1),  # direction
            50,  # distance
            math.pi / 3,  # view_angle
            2,  # minimal_gap_width
            fva.GapParams(
                gp.Vector(2.97253287, 5.62954211, gp.Point(2, 2)),
                gp.Vector(5.62954211, 2.97253287, gp.Point(2, 2)),
            ),  # expected_result
        ),
        (
            'MonzaSmall.csv',  # path_filename
            [
                obstcl.CircleObstacle(gp.Point(0, 5), 1),
                obstcl.CircleObstacle(gp.Point(5, 3), 1),
            ],  # obstacles
            100,  # fov_sampler_count
            gp.Point(2, 2),  # view_point
            gp.Vector(1, 1),  # direction
            50,  # distance
            math.pi / 3,  # view_angle
            2,  # minimal_gap_width
            fva.GapParams(
                gp.Vector(3.244768560380172, 6.645539511003168, gp.Point(2, 2)),
                gp.Vector(5.819434406313135, 4.922739627998688, gp.Point(2, 2)),
            )  # expected_result
        ),
    ]
)
def test_fov_analyzer_vector_sampler_get_available_gap(
    path_filename,
    obstacles,
    fov_sampler_count,
    view_point,
    direction,
    distance,
    view_angle,
    minimal_gap_width,
    expected_result,
):
    analyzer = fva.FieldOfViewAnalyzer(
        pr=path_reader.PathReader(path_filename),
        obstacles=obstacles,
        fov_sampler_count=fov_sampler_count,
    )
    assert analyzer.get_available_gap(
        view_point=view_point,
        direction=direction,
        distance=distance,
        view_angle=view_angle,
        minimal_gap_width=minimal_gap_width,
    ) == expected_result
