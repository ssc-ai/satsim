import math
import numpy as np
import pytest
from satsim.vecmath import Matrix2, Cartesian2, Matrix3, Cartesian3, Matrix4, Cartesian4, Quaternion
from satsim.math.const import PI_OVER_TWO, PI_OVER_FOUR, EPSILON6, EPSILON7, EPSILON11, EPSILON12, EPSILON14, EPSILON15, EPSILON20


def test_quaternion():

    quaternion = Quaternion()
    assert(quaternion.x == 0.0)
    assert(quaternion.y == 0.0)
    assert(quaternion.z == 0.0)
    assert(quaternion.w == 0.0)

    quaternion = Quaternion(1.0, 2.0, 3.0, 4.0)
    assert(quaternion.x == 1.0)
    assert(quaternion.y == 2.0)
    assert(quaternion.z == 3.0)
    assert(quaternion.w == 4.0)

    axis = Cartesian3(0.0, 0.0, 1.0)
    angle = PI_OVER_TWO
    s = math.sin(angle / 2.0)
    c = math.cos(angle / 2.0)
    a = Cartesian3.multiplyByScalar(axis, s, Cartesian3())
    expected = Quaternion(a.x, a.y, a.z, c)
    returnedResult = Quaternion.fromAxisAngle(axis, angle)
    assert(returnedResult == expected)

    axis = Cartesian3(0.0, 0.0, 1.0)
    angle = PI_OVER_TWO
    s = math.sin(angle / 2.0)
    c = math.cos(angle / 2.0)
    a = Cartesian3.multiplyByScalar(axis, s, Cartesian3())
    result = Quaternion()
    expected = Quaternion(a.x, a.y, a.z, c)
    returnedResult = Quaternion.fromAxisAngle(axis, angle, result)
    assert(result is returnedResult)
    assert(returnedResult == expected)

    q = Quaternion.fromAxisAngle(
        Cartesian3.negate(Cartesian3.UNIT_Z(), Cartesian3()),
        math.pi
    )
    rotation = Matrix3(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    assert(Quaternion.equalsEpsilon(Quaternion.fromRotationMatrix(rotation),
                                    q,
                                    EPSILON15
                                    ))

    q = Quaternion.fromAxisAngle(
        Cartesian3.negate(Cartesian3.UNIT_Y(), Cartesian3()),
        math.pi
    )
    rotation = Matrix3(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    assert(Quaternion.equalsEpsilon(Quaternion.fromRotationMatrix(rotation),
                                    q,
                                    EPSILON15
                                    ))

    q = Quaternion.fromAxisAngle(
        Cartesian3.negate(Cartesian3.UNIT_X(), Cartesian3()),
        math.pi
    )
    rotation = Matrix3(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0)
    assert(Quaternion.equalsEpsilon(Quaternion.fromRotationMatrix(rotation),
                                    q,
                                    EPSILON15
                                    ))

    rotation = Matrix3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    q = Quaternion(0.0, 0.0, 0.0, 1.0)
    assert(Quaternion.equalsEpsilon(Quaternion.fromRotationMatrix(rotation),
                                    q,
                                    EPSILON15
                                    ))

    rotation = Matrix3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    q = Quaternion(0.0, 0.0, 0.0, 1.0)
    result = Quaternion()
    returnedResult = Quaternion.fromRotationMatrix(rotation, result)
    assert(Quaternion.equalsEpsilon(returnedResult, q, EPSILON15))
    assert(returnedResult is result)

    direction = Cartesian3(
        -0.2349326833984488,
        0.8513513009480378,
        0.46904967396353314
    )
    up = Cartesian3(
        0.12477198625717335,
        -0.4521499177166376,
        0.8831717858696695
    )
    right = Cartesian3(
        0.9639702203483635,
        0.26601017702986895,
        6.456422901079747e-10
    )
    matrix = Matrix3(
        right.x,
        right.y,
        right.z,
        up.x,
        up.y,
        up.z,
        -direction.x,
        -direction.y,
        -direction.z
    )
    quaternion = Quaternion.fromRotationMatrix(matrix)
    assert(Matrix3.equalsEpsilon(Matrix3.fromQuaternion(quaternion),
                                 matrix,
                                 EPSILON12
                                 ))

    angle = np.radians(20.0)
    hpr = {
        'heading': angle,
        'pitch': 0.0,
        'roll': 0.0
    }
    quaternion = Quaternion.fromHeadingPitchRoll(hpr)
    assert(Matrix3.equalsEpsilon(Matrix3.fromQuaternion(quaternion),
                                 Matrix3.fromRotationZ(-angle),
                                 EPSILON11
                                 ))

    angle = np.radians(20.0)
    hpr = {
        'heading': 0.0,
        'pitch': angle,
        'roll': 0.0
    }
    quaternion = Quaternion.fromHeadingPitchRoll(hpr)
    assert(Matrix3.equalsEpsilon(Matrix3.fromQuaternion(quaternion),
                                 Matrix3.fromRotationY(-angle),
                                 EPSILON11
                                 ))

    angle = np.radians(20.0)
    hpr = {
        'heading': 0.0,
        'pitch': 0.0,
        'roll': angle
    }
    quaternion = Quaternion.fromHeadingPitchRoll(hpr)
    assert(Matrix3.equalsEpsilon(Matrix3.fromQuaternion(quaternion),
                                 Matrix3.fromRotationX(angle),
                                 EPSILON11
                                 ))

    angle = np.radians(20.0)
    hpr = {
        'heading': angle,
        'pitch': angle,
        'roll': angle
    }
    quaternion = Quaternion.fromHeadingPitchRoll(hpr)
    expected = Matrix3.fromRotationX(angle)
    Matrix3.multiply(Matrix3.fromRotationY(-angle), expected, expected)
    Matrix3.multiply(Matrix3.fromRotationZ(-angle), expected, expected)
    assert(Matrix3.equalsEpsilon(Matrix3.fromQuaternion(quaternion),
                                 expected,
                                 EPSILON11
                                 ))

    heading = np.radians(180.0)
    pitch = np.radians(-45.0)
    roll = np.radians(45.0)
    hpr = {
        'heading': heading,
        'pitch': pitch,
        'roll': roll
    }
    quaternion = Quaternion.fromHeadingPitchRoll(hpr)
    expected = Matrix3.fromRotationX(roll)
    Matrix3.multiply(Matrix3.fromRotationY(-pitch), expected, expected)
    Matrix3.multiply(Matrix3.fromRotationZ(-heading), expected, expected)
    assert(Matrix3.equalsEpsilon(Matrix3.fromQuaternion(quaternion),
                                 expected,
                                 EPSILON11
                                 ))

    angle = np.radians(20.0)
    hpr = {
        'heading': 0.0,
        'pitch': 0.0,
        'roll': angle
    }
    result = Quaternion()
    quaternion = Quaternion.fromHeadingPitchRoll(hpr, result)
    expected = Quaternion.fromRotationMatrix(
        Matrix3.fromRotationX(angle)
    )
    assert(quaternion is result)
    assert(Quaternion.equalsEpsilon(quaternion, expected, EPSILON11))

    quaternion = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = Quaternion.clone(quaternion)
    assert(quaternion is not result)
    assert(quaternion == result)

    quaternion = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = Quaternion()
    returnedResult = Quaternion.clone(quaternion, result)
    assert(quaternion is not result)
    assert(result is returnedResult)
    assert(quaternion == result)

    quaternion = Quaternion(1.0, 2.0, 3.0, 4.0)
    returnedResult = Quaternion.clone(quaternion, quaternion)
    assert(quaternion is returnedResult)

    expected = Quaternion(-1.0, -2.0, -3.0, 4.0)
    quaternion = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = Quaternion()
    returnedResult = Quaternion.conjugate(quaternion, result)
    assert(result is returnedResult)
    assert(returnedResult == expected)

    expected = Quaternion(-1.0, -2.0, -3.0, 4.0)
    quaternion = Quaternion(1.0, 2.0, 3.0, 4.0)
    returnedResult = Quaternion.conjugate(quaternion, quaternion)
    assert(quaternion is returnedResult)
    assert(quaternion == expected)

    expected = 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5
    quaternion = Quaternion(2.0, 3.0, 4.0, 5.0)
    result = Quaternion.magnitudeSquared(quaternion)
    assert(result == expected)

    expected = math.sqrt(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5)
    quaternion = Quaternion(2.0, 3.0, 4.0, 5.0)
    result = Quaternion.magnitude(quaternion)
    assert(result == expected)

    quaternion = Quaternion(2.0, 0.0, 0.0, 0.0)
    expectedResult = Quaternion(1.0, 0.0, 0.0, 0.0)
    result = Quaternion()
    returnedResult = Quaternion.normalize(quaternion, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    quaternion = Quaternion(2.0, 0.0, 0.0, 0.0)
    expectedResult = Quaternion(1.0, 0.0, 0.0, 0.0)
    returnedResult = Quaternion.normalize(quaternion, quaternion)
    assert(quaternion is returnedResult)
    assert(quaternion == expectedResult)

    quaternion = Quaternion(2.0, 3.0, 4.0, 5.0)
    magnitudeSquared = Quaternion.magnitudeSquared(quaternion)
    expected = Quaternion(
        -2.0 / magnitudeSquared,
        -3.0 / magnitudeSquared,
        -4.0 / magnitudeSquared,
        5.0 / magnitudeSquared
    )
    result = Quaternion()
    returnedResult = Quaternion.inverse(quaternion, result)
    assert(returnedResult == expected)
    assert(returnedResult is result)

    quaternion = Quaternion(2.0, 3.0, 4.0, 5.0)
    magnitudeSquared = Quaternion.magnitudeSquared(quaternion)
    expected = Quaternion(
        -2.0 / magnitudeSquared,
        -3.0 / magnitudeSquared,
        -4.0 / magnitudeSquared,
        5.0 / magnitudeSquared
    )
    returnedResult = Quaternion.inverse(quaternion, quaternion)
    assert(returnedResult == expected)
    assert(returnedResult is quaternion)

    left = Quaternion(2.0, 3.0, 6.0, 8.0)
    right = Quaternion(4.0, 5.0, 7.0, 9.0)
    expectedResult = 137.0
    result = Quaternion.dot(left, right)
    assert(result == expectedResult)

    left = Quaternion(1.0, 2.0, 3.0, 4.0)
    right = Quaternion(8.0, 7.0, 6.0, 5.0)

    expected = Quaternion(28.0, 56.0, 30.0, -20.0)
    result = Quaternion()
    returnedResult = Quaternion.multiply(left, right, result)
    assert(returnedResult == expected)
    assert(result is returnedResult)

    left = Quaternion(1.0, 2.0, 3.0, 4.0)
    right = Quaternion(8.0, 7.0, 6.0, 5.0)

    expected = Quaternion(28.0, 56.0, 30.0, -20.0)
    returnedResult = Quaternion.multiply(left, right, left)
    assert(returnedResult == expected)
    assert(left is returnedResult)

    left = Quaternion(2.0, 3.0, 6.0, 8.0)
    right = Quaternion(4.0, 5.0, 7.0, 9.0)
    result = Quaternion()
    expectedResult = Quaternion(6.0, 8.0, 13.0, 17.0)
    returnedResult = Quaternion.add(left, right, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    left = Quaternion(2.0, 3.0, 6.0, 8.0)
    right = Quaternion(4.0, 5.0, 7.0, 9.0)
    expectedResult = Quaternion(6.0, 8.0, 13.0, 17.0)
    returnedResult = Quaternion.add(left, right, left)
    assert(left is returnedResult)
    assert(left == expectedResult)

    left = Quaternion(2.0, 3.0, 4.0, 8.0)
    right = Quaternion(1.0, 5.0, 7.0, 9.0)
    result = Quaternion()
    expectedResult = Quaternion(1.0, -2.0, -3.0, -1.0)
    returnedResult = Quaternion.subtract(left, right, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    left = Quaternion(2.0, 3.0, 4.0, 8.0)
    right = Quaternion(1.0, 5.0, 7.0, 9.0)
    expectedResult = Quaternion(1.0, -2.0, -3.0, -1.0)
    returnedResult = Quaternion.subtract(left, right, left)
    assert(returnedResult is left)
    assert(left == expectedResult)

    quaternion = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = Quaternion()
    scalar = 2
    expectedResult = Quaternion(2.0, 4.0, 6.0, 8.0)
    returnedResult = Quaternion.multiplyByScalar(
        quaternion,
        scalar,
        result
    )
    assert(result is returnedResult)
    assert(result == expectedResult)

    quaternion = Quaternion(1.0, 2.0, 3.0, 4.0)
    scalar = 2
    expectedResult = Quaternion(2.0, 4.0, 6.0, 8.0)
    returnedResult = Quaternion.multiplyByScalar(
        quaternion,
        scalar,
        quaternion
    )
    assert(quaternion is returnedResult)
    assert(quaternion == expectedResult)

    quaternion = Quaternion(1.0, 2.0, 3.0, 4.0)
    result = Quaternion()
    scalar = 2
    expectedResult = Quaternion(0.5, 1.0, 1.5, 2.0)
    returnedResult = Quaternion.divideByScalar(
        quaternion,
        scalar,
        result
    )
    assert(result is returnedResult)
    assert(result == expectedResult)

    quaternion = Quaternion(1.0, 2.0, 3.0, 4.0)
    scalar = 2
    expectedResult = Quaternion(0.5, 1.0, 1.5, 2.0)
    returnedResult = Quaternion.divideByScalar(
        quaternion,
        scalar,
        quaternion
    )
    assert(quaternion is returnedResult)
    assert(quaternion == expectedResult)

    # 60 degrees is used here to ensure that the sine and cosine of the half angle are not equal.
    angle = math.pi / 3.0
    cos = math.cos(angle / 2.0)
    sin = math.sin(angle / 2.0)
    expected = Cartesian3.normalize(
        Cartesian3(2.0, 3.0, 6.0),
        Cartesian3()
    )
    quaternion = Quaternion(
        sin * expected.x,
        sin * expected.y,
        sin * expected.z,
        cos
    )
    result = Cartesian3()
    returnedResult = Quaternion.computeAxis(quaternion, result)
    assert(Cartesian3.equalsEpsilon(returnedResult, expected, EPSILON15))
    assert(result is returnedResult)

    expected = Cartesian3(0.0, 0.0, 0.0)
    quaternion = Quaternion(4.0, 2.0, 3.0, 1.0)
    result = Cartesian3(1, 2, 3)
    returnedResult = Quaternion.computeAxis(quaternion, result)
    assert(returnedResult == expected)
    assert(result is returnedResult)

    # 60 degrees is used here to ensure that the sine and cosine of the half angle are not equal.
    angle = math.pi / 3.0
    cos = math.cos(angle / 2.0)
    sin = math.sin(angle / 2.0)
    axis = Cartesian3.normalize(
        Cartesian3(2.0, 3.0, 6.0),
        Cartesian3()
    )
    quaternion = Quaternion(
        sin * axis.x,
        sin * axis.y,
        sin * axis.z,
        cos
    )
    result = Quaternion.computeAngle(quaternion)
    assert(np.isclose(result, angle, 0, EPSILON15))

    result = Quaternion.computeAngle(Quaternion(0,0,0,1.0 - EPSILON7))
    assert(result == 0)

    quaternion = Quaternion(1.0, -2.0, -5.0, 4.0)
    result = Quaternion()
    expectedResult = Quaternion(-1.0, 2.0, 5.0, -4.0)
    returnedResult = Quaternion.negate(quaternion, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    quaternion = Quaternion(1.0, -2.0, -5.0)
    expectedResult = Quaternion(-1.0, 2.0, 5.0)
    returnedResult = Quaternion.negate(quaternion, quaternion)
    assert(quaternion is returnedResult)
    assert(quaternion == expectedResult)

    start = Quaternion(4.0, 8.0, 10.0, 20.0)
    end = Quaternion(8.0, 20.0, 20.0, 30.0)
    t = 0.25
    result = Quaternion()
    expectedResult = Quaternion(5.0, 11.0, 12.5, 22.5)
    returnedResult = Quaternion.lerp(start, end, t, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    start = Quaternion(4.0, 8.0, 10.0, 20.0)
    end = Quaternion(8.0, 20.0, 20.0, 30.0)
    t = 0.25
    expectedResult = Quaternion(5.0, 11.0, 12.5, 22.5)
    returnedResult = Quaternion.lerp(start, end, t, start)
    assert(start is returnedResult)
    assert(start == expectedResult)

    start = Quaternion(4.0, 8.0, 10.0, 20.0)
    end = Quaternion(8.0, 20.0, 20.0, 30.0)
    t = 2.0
    expectedResult = Quaternion(12.0, 32.0, 30.0, 40.0)
    result = Quaternion.lerp(start, end, t, Quaternion())
    assert(result == expectedResult)

    start = Quaternion(4.0, 8.0, 10.0, 20.0)
    end = Quaternion(8.0, 20.0, 20.0, 30.0)
    t = -1.0
    expectedResult = Quaternion(0.0, -4.0, 0.0, 10.0)
    result = Quaternion.lerp(start, end, t, Quaternion())
    assert(result == expectedResult)

    start = Quaternion.normalize(
        Quaternion(0.0, 0.0, 0.0, 1.0),
        Quaternion()
    )
    end = Quaternion(
        0.0,
        0.0,
        math.sin(PI_OVER_FOUR),
        math.cos(PI_OVER_FOUR)
    )
    expected = Quaternion(
        0.0,
        0.0,
        math.sin(math.pi / 8.0),
        math.cos(math.pi / 8.0)
    )

    result = Quaternion()
    returnedResult = Quaternion.slerp(start, end, 0.5, result)
    assert(Quaternion.equalsEpsilon(result, expected, EPSILON15))
    assert(result is returnedResult)

    start = Quaternion.normalize(
        Quaternion(0.0, 0.0, 0.0, 1.0),
        Quaternion()
    )
    end = Quaternion(
        0.0,
        0.0,
        math.sin(PI_OVER_FOUR),
        math.cos(PI_OVER_FOUR)
    )
    expected = Quaternion(
        0.0,
        0.0,
        math.sin(math.pi / 8.0),
        math.cos(math.pi / 8.0)
    )

    returnedResult = Quaternion.slerp(start, end, 0.5, start)
    assert(Quaternion.equalsEpsilon(start, expected, EPSILON15))
    assert(start is returnedResult)

    start = Quaternion.normalize(
        Quaternion(0.0, 0.0, 0.0, -1.0),
        Quaternion()
    )
    end = Quaternion(
        0.0,
        0.0,
        math.sin(PI_OVER_FOUR),
        math.cos(PI_OVER_FOUR)
    )
    expected = Quaternion(
        0.0,
        0.0,
        -math.sin(math.pi / 8.0),
        -math.cos(math.pi / 8.0)
    )
    assert(Quaternion.equalsEpsilon(Quaternion.slerp(start, end, 0.5, Quaternion()),
                                    expected,
                                    EPSILON15
                                    ))

    start = Quaternion(0.0, 0.0, 0.0, 1.0)
    end = Quaternion(1.0, 2.0, 3.0, 1.0)
    expected = Quaternion(0.5, 1.0, 1.5, 1.0)
    result = Quaternion()
    assert(Quaternion.slerp(start, end, 0.0, result) == start)
    assert(Quaternion.slerp(start, end, 1.0, result) == end)
    assert(Quaternion.slerp(start, end, 0.5, result) == expected)

    start = Quaternion(0.0, 0.0, 0.0, 1.0)
    end = Quaternion(1.0, 2.0, 3.0, 1.0)

    result = Quaternion()
    actual = Quaternion.slerp(start, end, 0.0, result)
    assert(actual is result)
    assert(result == start)

    axis = Cartesian3.normalize(
        Cartesian3(1.0, -1.0, 1.0),
        Cartesian3()
    )
    angle = PI_OVER_FOUR
    quat = Quaternion.fromAxisAngle(axis, angle)

    result = Cartesian3()
    log = Quaternion.log(quat, result)
    expected = Cartesian3.multiplyByScalar(
        axis,
        angle * 0.5,
        Cartesian3()
    )
    assert(log is result)
    assert(Cartesian3.equalsEpsilon(log, expected, EPSILON15))

    axis = Cartesian3.normalize(
        Cartesian3(1.0, -1.0, 1.0),
        Cartesian3()
    )
    angle = PI_OVER_FOUR
    cartesian = Cartesian3.multiplyByScalar(
        axis,
        angle * 0.5,
        Cartesian3()
    )

    result = Quaternion()
    exp = Quaternion.exp(cartesian, result)
    expected = Quaternion.fromAxisAngle(axis, angle)
    assert(exp is result)
    assert(Quaternion.equalsEpsilon(exp, expected, EPSILON15))

    q0 = Quaternion.fromAxisAngle(Cartesian3.UNIT_X(), 0.0)
    q1 = Quaternion.fromAxisAngle(
        Cartesian3.UNIT_X(),
        PI_OVER_FOUR
    )
    q2 = Quaternion.fromAxisAngle(
        Cartesian3.UNIT_Z(),
        PI_OVER_FOUR
    )
    q3 = Quaternion.fromAxisAngle(
        Cartesian3.UNIT_X(),
        -PI_OVER_FOUR
    )

    s1Result = Quaternion()
    s1 = Quaternion.computeInnerQuadrangle(q0, q1, q2, s1Result)
    assert(s1 is s1Result)

    s2 = Quaternion.computeInnerQuadrangle(q1, q2, q3, Quaternion())

    squadResult = Quaternion()
    squad = Quaternion.squad(q1, q2, s1, s2, 0.0, squadResult)
    assert(squad is squadResult)
    assert(Quaternion.equalsEpsilon(squad, q1, EPSILON15))

    start = Quaternion.normalize(
        Quaternion(0.0, 0.0, 0.0, 1.0),
        Quaternion()
    )
    end = Quaternion(
        0.0,
        0.0,
        math.sin(PI_OVER_FOUR),
        math.cos(PI_OVER_FOUR)
    )
    expected = Quaternion(
        0.0,
        0.0,
        math.sin(math.pi / 8.0),
        math.cos(math.pi / 8.0)
    )

    result = Quaternion()
    returnedResult = Quaternion.fastSlerp(start, end, 0.5, result)
    assert(Quaternion.equalsEpsilon(result, expected, EPSILON6))
    assert(result is returnedResult)

    start = Quaternion.normalize(
        Quaternion(0.0, 0.0, 0.0, 1.0),
        Quaternion()
    )
    end = Quaternion(
        0.0,
        0.0,
        math.sin(PI_OVER_FOUR),
        math.cos(PI_OVER_FOUR)
    )
    expected = Quaternion(
        0.0,
        0.0,
        math.sin(math.pi / 8.0),
        math.cos(math.pi / 8.0)
    )

    returnedResult = Quaternion.fastSlerp(start, end, 0.5, start)
    assert(Quaternion.equalsEpsilon(start, expected, EPSILON6))
    assert(start is returnedResult)

    start = Quaternion.normalize(
        Quaternion(0.0, 0.0, 0.0, -1.0),
        Quaternion()
    )
    end = Quaternion(
        0.0,
        0.0,
        math.sin(PI_OVER_FOUR),
        math.cos(PI_OVER_FOUR)
    )
    expected = Quaternion(
        0.0,
        0.0,
        -math.sin(math.pi / 8.0),
        -math.cos(math.pi / 8.0)
    )
    assert(
        Quaternion.equalsEpsilon(Quaternion.fastSlerp(start, end, 0.5, Quaternion()), expected, EPSILON6)
    )

    start = Quaternion.normalize(
        Quaternion(0.0, 0.0, 0.0, 1.0),
        Quaternion()
    )
    end = Quaternion(
        0.0,
        0.0,
        math.sin(PI_OVER_FOUR),
        math.cos(PI_OVER_FOUR)
    )

    expected = Quaternion.slerp(start, end, 0.25, Quaternion())
    actual = Quaternion.fastSlerp(start, end, 0.25, Quaternion())
    assert(Quaternion.equalsEpsilon(actual, expected, EPSILON6))

    expected = Quaternion.slerp(start, end, 0.5, Quaternion())
    actual = Quaternion.fastSlerp(start, end, 0.5, Quaternion())
    assert(Quaternion.equalsEpsilon(actual, expected, EPSILON6))

    expected = Quaternion.slerp(start, end, 0.75, Quaternion())
    actual = Quaternion.fastSlerp(start, end, 0.75, Quaternion())
    assert(Quaternion.equalsEpsilon(actual, expected, EPSILON6))

    q0 = Quaternion.fromAxisAngle(Cartesian3.UNIT_X(), 0.0)
    q1 = Quaternion.fromAxisAngle(
        Cartesian3.UNIT_X(),
        PI_OVER_FOUR
    )
    q2 = Quaternion.fromAxisAngle(
        Cartesian3.UNIT_Z(),
        PI_OVER_FOUR
    )
    q3 = Quaternion.fromAxisAngle(
        Cartesian3.UNIT_X(),
        -PI_OVER_FOUR
    )

    s1 = Quaternion.computeInnerQuadrangle(q0, q1, q2, Quaternion())
    s2 = Quaternion.computeInnerQuadrangle(q1, q2, q3, Quaternion())

    squadResult = Quaternion()
    squad = Quaternion.fastSquad(q1, q2, s1, s2, 0.0, squadResult)
    assert(squad is squadResult)
    assert(Quaternion.equalsEpsilon(squad, q1, EPSILON6))

    q0 = Quaternion.fromAxisAngle(Cartesian3.UNIT_X(), 0.0)
    q1 = Quaternion.fromAxisAngle(
        Cartesian3.UNIT_X(),
        PI_OVER_FOUR
    )
    q2 = Quaternion.fromAxisAngle(
        Cartesian3.UNIT_Z(),
        PI_OVER_FOUR
    )
    q3 = Quaternion.fromAxisAngle(
        Cartesian3.UNIT_X(),
        -PI_OVER_FOUR
    )

    s1 = Quaternion.computeInnerQuadrangle(q0, q1, q2, Quaternion())
    s2 = Quaternion.computeInnerQuadrangle(q1, q2, q3, Quaternion())

    actual = Quaternion.fastSquad(q1, q2, s1, s2, 0.25, Quaternion())
    expected = Quaternion.squad(q1, q2, s1, s2, 0.25, Quaternion())
    assert(Quaternion.equalsEpsilon(actual, expected, EPSILON6))

    actual = Quaternion.fastSquad(q1, q2, s1, s2, 0.5, Quaternion())
    expected = Quaternion.squad(q1, q2, s1, s2, 0.5, Quaternion())
    assert(Quaternion.equalsEpsilon(actual, expected, EPSILON6))

    actual = Quaternion.fastSquad(q1, q2, s1, s2, 0.75, Quaternion())
    expected = Quaternion.squad(q1, q2, s1, s2, 0.75, Quaternion())
    assert(Quaternion.equalsEpsilon(actual, expected, EPSILON6))

    quaternion = Quaternion(1.0, 2.0, 3.0, 4.0)
    assert(
        Quaternion.equals(quaternion, Quaternion(1.0, 2.0, 3.0, 4.0))
        is True)
    assert(
        Quaternion.equals(quaternion, Quaternion(2.0, 2.0, 3.0, 4.0))
        is False)
    assert(
        Quaternion.equals(quaternion, Quaternion(2.0, 1.0, 3.0, 4.0))
        is False)
    assert(
        Quaternion.equals(quaternion, Quaternion(1.0, 2.0, 4.0, 4.0))
        is False)
    assert(
        Quaternion.equals(quaternion, Quaternion(1.0, 2.0, 3.0, 5.0))
        is False)
    assert(Quaternion.equals(quaternion, None) is False)

    quaternion = Quaternion(1.0, 2.0, 3.0, 4.0)
    assert(
        Quaternion.equalsEpsilon(
            quaternion,
            Quaternion(1.0, 2.0, 3.0, 4.0),
            0.0
        )
        is True)
    assert(
        Quaternion.equalsEpsilon(
            quaternion,
            Quaternion(1.0, 2.0, 3.0, 4.0),
            1.0
        )
        is True)
    assert(
        Quaternion.equalsEpsilon(
            quaternion,
            Quaternion(2.0, 2.0, 3.0, 4.0),
            1.0
        )
        is True)
    assert(
        Quaternion.equalsEpsilon(
            quaternion,
            Quaternion(1.0, 3.0, 3.0, 4.0),
            1.0
        )
        is True)
    assert(
        Quaternion.equalsEpsilon(
            quaternion,
            Quaternion(1.0, 2.0, 4.0, 4.0),
            1.0
        )
        is True)
    assert(
        Quaternion.equalsEpsilon(
            quaternion,
            Quaternion(1.0, 2.0, 3.0, 5.0),
            1.0
        )
        is True)
    assert(
        Quaternion.equalsEpsilon(
            quaternion,
            Quaternion(2.0, 2.0, 3.0, 4.0),
            0.99999
        )
        is False)
    assert(
        Quaternion.equalsEpsilon(
            quaternion,
            Quaternion(1.0, 3.0, 3.0, 4.0),
            0.99999
        )
        is False)
    assert(
        Quaternion.equalsEpsilon(
            quaternion,
            Quaternion(1.0, 2.0, 4.0, 4.0),
            0.99999
        )
        is False)
    assert(
        Quaternion.equalsEpsilon(
            quaternion,
            Quaternion(1.0, 2.0, 3.0, 5.0),
            0.99999
        )
        is False)
    assert(Quaternion.equalsEpsilon(quaternion, None, 1) is False)

    quaternion = Quaternion(1.123, 2.345, 6.789, 6.123)
    assert(str(quaternion) == "[1.123, 2.345, 6.789, 6.123]")

    assert(Quaternion.clone() is None)


def test_matrix2():

    matrix = Matrix2()
    assert(matrix[Matrix2.COLUMN0ROW0] == 0.0)
    assert(matrix[Matrix2.COLUMN1ROW0] == 0.0)
    assert(matrix[Matrix2.COLUMN0ROW1] == 0.0)
    assert(matrix[Matrix2.COLUMN1ROW1] == 0.0)

    matrix = Matrix2(1.0, 2.0, 3.0, 4.0)
    assert(matrix[Matrix2.COLUMN0ROW0] == 1.0)
    assert(matrix[Matrix2.COLUMN1ROW0] == 2.0)
    assert(matrix[Matrix2.COLUMN0ROW1] == 3.0)
    assert(matrix[Matrix2.COLUMN1ROW1] == 4.0)

    expected = Matrix2(1.0, 2.0, 3.0, 4.0)
    matrix = Matrix2.fromArray([1.0, 3.0, 2.0, 4.0])
    assert(matrix == expected)

    expected = Matrix2(1.0, 2.0, 3.0, 4.0)
    result = Matrix2()
    matrix = Matrix2.fromArray([1.0, 3.0, 2.0, 4.0], 0, result)
    assert(matrix is result)
    assert(matrix == expected)

    expected = Matrix2(1.0, 2.0, 3.0, 4.0)
    result = Matrix2()
    matrix = Matrix2.fromArray(
        [0.0, 0.0, 0.0, 1.0, 3.0, 2.0, 4.0],
        3,
        result
    )
    assert(matrix is result)
    assert(matrix == expected)

    expected = Matrix2(1.0, 2.0, 3.0, 4.0)
    matrix = Matrix2.fromRowMajorArray([1.0, 2.0, 3.0, 4.0])
    assert(matrix == expected)

    expected = Matrix2(1.0, 2.0, 3.0, 4.0)
    result = Matrix2()
    matrix = Matrix2.fromRowMajorArray([1.0, 2.0, 3.0, 4.0], result)
    assert(matrix is result)
    assert(matrix == expected)

    expected = Matrix2(1.0, 2.0, 3.0, 4.0)
    matrix = Matrix2.fromColumnMajorArray([1.0, 3.0, 2.0, 4.0])
    assert(matrix == expected)

    expected = Matrix2(1.0, 2.0, 3.0, 4.0)
    result = Matrix2()
    matrix = Matrix2.fromColumnMajorArray([1.0, 3.0, 2.0, 4.0], result)
    assert(matrix is result)
    assert(matrix == expected)

    expected = Matrix2(7.0, 0.0, 0.0, 8.0)
    returnedResult = Matrix2.fromScale(Cartesian2(7.0, 8.0))
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = Matrix2(7.0, 0.0, 0.0, 8.0)
    result = Matrix2()
    returnedResult = Matrix2.fromScale(Cartesian2(7.0, 8.0), result)
    assert(returnedResult is result)
    assert(returnedResult == expected)

    expected = Matrix2(2.0, 0.0, 0.0, 2.0)
    returnedResult = Matrix2.fromUniformScale(2.0)
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = Matrix2(2.0, 0.0, 0.0, 2.0)
    result = Matrix2()
    returnedResult = Matrix2.fromUniformScale(2.0, result)
    assert(returnedResult is result)
    assert(returnedResult == expected)

    matrix = Matrix2.fromRotation(0.0)
    assert(matrix == Matrix2.IDENTITY())

    expected = Matrix2(0.0, -1.0, 1.0, 0.0)
    result = Matrix2()
    matrix = Matrix2.fromRotation(np.radians(90.0), result)
    assert(matrix is result)
    assert(Matrix2.equalsEpsilon(matrix, expected, EPSILON15))

    expected = Matrix2(1.0, 2.0, 3.0, 4.0)
    returnedResult = Matrix2.clone(expected)
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = Matrix2(1.0, 2.0, 3.0, 4.0)
    result = Matrix2()
    returnedResult = Matrix2.clone(expected, result)
    assert(returnedResult is result)
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = [1.0, 2.0, 3.0, 4.0]
    returnedResult = Matrix2.toArray(
        Matrix2.fromColumnMajorArray(expected)
    )
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = [1.0, 2.0, 3.0, 4.0]
    result = [0] * 4
    returnedResult = Matrix2.toArray(
        Matrix2.fromColumnMajorArray(expected),
        result
    )
    assert(returnedResult is result)
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    i = 0
    for col in range(0, 2):
        for row in range(0, 2):
            index = Matrix2.getElementIndex(col, row)
            assert(index == i)
            i = i + 1

    matrix = Matrix2(1.0, 2.0, 3.0, 4.0)
    expectedColumn0 = Cartesian2(1.0, 3.0)
    expectedColumn1 = Cartesian2(2.0, 4.0)

    resultColumn0 = Cartesian2()
    resultColumn1 = Cartesian2()
    returnedResultColumn0 = Matrix2.getColumn(matrix, 0, resultColumn0)
    returnedResultColumn1 = Matrix2.getColumn(matrix, 1, resultColumn1)

    assert(resultColumn0 is returnedResultColumn0)
    assert(resultColumn0 == expectedColumn0)
    assert(resultColumn1 is returnedResultColumn1)
    assert(resultColumn1 == expectedColumn1)

    matrix = Matrix2(1.0, 2.0, 3.0, 4.0)
    result = Matrix2()

    expected = Matrix2(5.0, 2.0, 6.0, 4.0)
    returnedResult = Matrix2.setColumn(
        matrix,
        0,
        Cartesian2(5.0, 6.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    expected = Matrix2(1.0, 7.0, 3.0, 8.0)
    returnedResult = Matrix2.setColumn(
        matrix,
        1,
        Cartesian2(7.0, 8.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    matrix = Matrix2(1.0, 2.0, 3.0, 4.0)
    expectedRow0 = Cartesian2(1.0, 2.0)
    expectedRow1 = Cartesian2(3.0, 4.0)

    resultRow0 = Cartesian2()
    resultRow1 = Cartesian2()
    returnedResultRow0 = Matrix2.getRow(matrix, 0, resultRow0)
    returnedResultRow1 = Matrix2.getRow(matrix, 1, resultRow1)

    assert(resultRow0 is returnedResultRow0)
    assert(resultRow0 == expectedRow0)
    assert(resultRow1 is returnedResultRow1)
    assert(resultRow1 == expectedRow1)

    matrix = Matrix2(1.0, 2.0, 3.0, 4.0)
    result = Matrix2()

    expected = Matrix2(5.0, 6.0, 3.0, 4.0)
    returnedResult = Matrix2.setRow(
        matrix,
        0,
        Cartesian2(5.0, 6.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    expected = Matrix2(1.0, 2.0, 7.0, 8.0)
    returnedResult = Matrix2.setRow(
        matrix,
        1,
        Cartesian2(7.0, 8.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    oldScale = Cartesian2(2.0, 3.0)
    newScale = Cartesian2(4.0, 5.0)

    matrix = Matrix2.fromScale(oldScale, Matrix2())
    result = Matrix2()

    assert(Matrix2.getScale(matrix, Cartesian2()) == oldScale)

    returnedResult = Matrix2.setScale(matrix, newScale, result)

    assert(Matrix2.getScale(returnedResult, Cartesian2()) ==
           newScale
           )
    assert(result is returnedResult)

    oldScale = Cartesian2(2.0, 3.0)
    newScale = 4.0

    matrix = Matrix2.fromScale(oldScale, Matrix2())
    result = Matrix2()

    assert(Matrix2.getScale(matrix, Cartesian2()) == oldScale)

    returnedResult = Matrix2.setUniformScale(matrix, newScale, result)

    assert(Matrix2.getScale(returnedResult, Cartesian2()) ==
           Cartesian2(newScale, newScale)
           )
    assert(result is returnedResult)

    scale = Cartesian2(2.0, 3.0)
    result = Cartesian2()
    computedScale = Matrix2.getScale(Matrix2.fromScale(scale), result)

    assert(computedScale is result)
    assert(Cartesian2.equalsEpsilon(computedScale, scale, EPSILON14))

    m = Matrix2.fromScale(Cartesian2(2.0, 3.0))
    assert(np.isclose(Matrix2.getMaximumScale(m),
                      3.0,
                      EPSILON14
                      ))

    scaleVec = Cartesian2(2.0, 3.0)
    scale = Matrix2.fromScale(scaleVec, Matrix2())
    rotation = Matrix2.fromRotation(0.5, Matrix2())
    scaleRotation = Matrix2.setRotation(scale, rotation, Matrix2())

    extractedScale = Matrix2.getScale(scaleRotation, Cartesian2())
    extractedRotation = Matrix2.getRotation(scaleRotation, Matrix2())

    assert(Cartesian2.equalsEpsilon(extractedScale, scaleVec, EPSILON14))
    assert(Matrix2.equalsEpsilon(extractedRotation, rotation, EPSILON14))

    matrix = Matrix2.fromColumnMajorArray([1.0, 2.0, 3.0, 4.0])
    expectedRotation = Matrix2.fromArray([
        1.0 / math.sqrt(1.0 * 1.0 + 2.0 * 2.0),
        2.0 / math.sqrt(1.0 * 1.0 + 2.0 * 2.0),
        3.0 / math.sqrt(3.0 * 3.0 + 4.0 * 4.0),
        4.0 / math.sqrt(3.0 * 3.0 + 4.0 * 4.0),
    ])
    rotation = Matrix2.getRotation(matrix, Matrix2())
    assert(Matrix2.equalsEpsilon(rotation, expectedRotation, EPSILON14))

    matrix = Matrix2.fromColumnMajorArray([1.0, 2.0, 3.0, 4.0])
    duplicateMatrix = Matrix2.clone(matrix, Matrix2())
    expectedRotation = Matrix2.fromArray([
        1.0 / math.sqrt(1.0 * 1.0 + 2.0 * 2.0),
        2.0 / math.sqrt(1.0 * 1.0 + 2.0 * 2.0),
        3.0 / math.sqrt(3.0 * 3.0 + 4.0 * 4.0),
        4.0 / math.sqrt(3.0 * 3.0 + 4.0 * 4.0),
    ])
    result = Matrix2.getRotation(matrix, Matrix2())
    assert(Matrix2.equalsEpsilon(result, expectedRotation, EPSILON14))
    assert(matrix == duplicateMatrix)
    assert(matrix is not result)

    left = Matrix2(1, 2, 3, 4)
    right = Matrix2(5, 6, 7, 8)
    expected = Matrix2(19, 22, 43, 50)
    result = Matrix2()
    returnedResult = Matrix2.multiply(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    left = Matrix2(1, 2, 3, 4)
    right = Matrix2(5, 6, 7, 8)
    expected = Matrix2(19, 22, 43, 50)
    returnedResult = Matrix2.multiply(left, right, left)
    assert(returnedResult is left)
    assert(left == expected)

    left = Matrix2(1, 2, 3, 4)
    right = Matrix2(10, 11, 12, 13)
    expected = Matrix2(11, 13, 15, 17)
    result = Matrix2()
    returnedResult = Matrix2.add(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    left = Matrix2(1, 2, 3, 4)
    right = Matrix2(10, 11, 12, 13)
    expected = Matrix2(11, 13, 15, 17)
    returnedResult = Matrix2.add(left, right, left)
    assert(returnedResult is left)
    assert(left == expected)

    left = Matrix2(11, 13, 15, 17)
    right = Matrix2(10, 11, 12, 13)
    expected = Matrix2(1, 2, 3, 4)
    result = Matrix2()
    returnedResult = Matrix2.subtract(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    left = Matrix2(11, 13, 15, 17)
    right = Matrix2(10, 11, 12, 13)
    expected = Matrix2(1, 2, 3, 4)
    returnedResult = Matrix2.subtract(left, right, left)
    assert(returnedResult is left)
    assert(left == expected)

    m = Matrix2(2, 3, 6, 7)
    scale = Cartesian2(2.0, 3.0)
    expected = Matrix2.multiply(
        m,
        Matrix2.fromScale(scale),
        Matrix2()
    )
    result = Matrix2()
    returnedResult = Matrix2.multiplyByScale(m, scale, result)
    assert(returnedResult is result)
    assert(result == expected)

    m = Matrix2(1, 2, 5, 6)
    scale = Cartesian2(1.0, 2.0)
    expected = Matrix2.multiply(
        m,
        Matrix2.fromScale(scale),
        Matrix2()
    )
    returnedResult = Matrix2.multiplyByScale(m, scale, m)
    assert(returnedResult is m)
    assert(m == expected)

    m = Matrix2(2, 3, 4, 5)
    scale = 2.0
    expected = Matrix2.multiply(
        m,
        Matrix2.fromUniformScale(scale),
        Matrix2()
    )
    result = Matrix2()
    returnedResult = Matrix2.multiplyByUniformScale(m, scale, result)
    assert(returnedResult is result)
    assert(result == expected)

    m = Matrix2(2, 3, 4, 5)
    scale = 2.0
    expected = Matrix2.multiply(
        m,
        Matrix2.fromUniformScale(scale),
        Matrix2()
    )
    returnedResult = Matrix2.multiplyByUniformScale(m, scale, m)
    assert(returnedResult is m)
    assert(m == expected)

    left = Matrix2(1, 2, 3, 4)
    right = Cartesian2(5, 6)
    expected = Cartesian2(17, 39)
    result = Cartesian2()
    returnedResult = Matrix2.multiplyByVector(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    left = Matrix2(1, 2, 3, 4)
    right = 2
    expected = Matrix2(2, 4, 6, 8)
    result = Matrix2()
    returnedResult = Matrix2.multiplyByScalar(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    matrix = Matrix2(1, 2, 3, 4)
    expected = Matrix2(-1, -2, -3, -4)
    result = Matrix2()
    returnedResult = Matrix2.negate(matrix, result)
    assert(result is returnedResult)
    assert(result == expected)

    matrix = Matrix2(1, 2, 3, 4)
    expected = Matrix2(-1, -2, -3, -4)
    returnedResult = Matrix2.negate(matrix, matrix)
    assert(matrix is returnedResult)
    assert(matrix == expected)

    matrix = Matrix2(1, 2, 3, 4)
    expected = Matrix2(1, 3, 2, 4)
    result = Matrix2()
    returnedResult = Matrix2.transpose(matrix, result)
    assert(result is returnedResult)
    assert(result == expected)

    matrix = Matrix2(1, 2, 3, 4)
    expected = Matrix2(1, 3, 2, 4)
    returnedResult = Matrix2.transpose(matrix, matrix)
    assert(matrix is returnedResult)
    assert(matrix == expected)

    matrix = Matrix2(-1.0, -2.0, -3.0, -4.0)
    expected = Matrix2(1.0, 2.0, 3.0, 4.0)
    result = Matrix2()
    returnedResult = Matrix2.abs(matrix, result)
    assert(returnedResult == expected)

    matrix = Matrix2(1.0, 2.0, 3.0, 4.0)
    returnedResult = Matrix2.abs(matrix, result)
    assert(returnedResult == expected)

    matrix = Matrix2(1.0, -2.0, -3.0, 4.0)
    returnedResult = Matrix2.abs(matrix, result)
    assert(returnedResult == expected)

    matrix = Matrix2(-1.0, -2.0, -3.0, -4.0)
    expected = Matrix2(1.0, 2.0, 3.0, 4.0)
    returnedResult = Matrix2.abs(matrix, matrix)
    assert(matrix is returnedResult)
    assert(matrix == expected)

    left = Matrix2(1.0, 2.0, 3.0, 4.0)
    right = Matrix2(1.0, 2.0, 3.0, 4.0)
    assert(left == right)

    left = Matrix2(1.0, 2.0, 3.0, 4.0)
    right = Matrix2(5.0, 2.0, 3.0, 4.0)
    assert(left != right)

    left = Matrix2(1.0, 2.0, 3.0, 4.0)
    right = Matrix2(1.0, 6.0, 3.0, 4.0)
    assert(left != right)

    left = Matrix2(1.0, 2.0, 3.0, 4.0)
    right = Matrix2(1.0, 2.0, 7.0, 4.0)
    assert(left != right)

    left = Matrix2(1.0, 2.0, 3.0, 4.0)
    right = Matrix2(1.0, 2.0, 3.0, 8.0)
    assert(left != right)

    assert(Matrix2() is not None)
    assert(None is not Matrix2())

    left = Matrix2(1.0, 2.0, 3.0, 4.0)
    right = Matrix2(1.0, 2.0, 3.0, 4.0)
    assert(Matrix2.equalsEpsilon(left, right, 1.0) is True)

    left = Matrix2(1.0, 2.0, 3.0, 4.0)
    right = Matrix2(5.0, 2.0, 3.0, 4.0)
    assert(Matrix2.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix2.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix2(1.0, 2.0, 3.0, 4.0)
    right = Matrix2(1.0, 6.0, 3.0, 4.0)
    assert(Matrix2.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix2.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix2(1.0, 2.0, 3.0, 4.0)
    right = Matrix2(1.0, 2.0, 7.0, 4.0)
    assert(Matrix2.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix2.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix2(1.0, 2.0, 3.0, 4.0)
    right = Matrix2(1.0, 2.0, 3.0, 8.0)
    assert(Matrix2.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix2.equalsEpsilon(left, right, 4.0) is True)

    assert(Matrix2.equalsEpsilon(None, None, 1.0) is True)
    assert(Matrix2.equalsEpsilon(Matrix2(), None, 1.0) is False)
    assert(Matrix2.equalsEpsilon(None, Matrix2(), 1.0) is False)

    matrix = Matrix2(1, 2, 3, 4)
    assert(str(matrix) == "[1, 2]\n[3, 4]")

    assert(Matrix2.clone(None) is None)

    matrix = Matrix2(1, 3, 2, 4)
    assert(len(matrix) == 4)

    for index in range(0, len(matrix)):
        assert(matrix[index] == index + 1)

    matrix = Matrix2.IDENTITY()
    assert(matrix[Matrix2.COLUMN0ROW0] == 1.0)
    assert(matrix[Matrix2.COLUMN1ROW0] == 0.0)
    assert(matrix[Matrix2.COLUMN0ROW1] == 0.0)
    assert(matrix[Matrix2.COLUMN1ROW1] == 1.0)

    matrix = Matrix2.ZERO()
    assert(matrix[Matrix2.COLUMN0ROW0] == 0.0)
    assert(matrix[Matrix2.COLUMN1ROW0] == 0.0)
    assert(matrix[Matrix2.COLUMN0ROW1] == 0.0)
    assert(matrix[Matrix2.COLUMN1ROW1] == 0.0)


def test_matrix3():

    matrix = Matrix3()
    assert(matrix[Matrix3.COLUMN0ROW0] == 0.0)
    assert(matrix[Matrix3.COLUMN1ROW0] == 0.0)
    assert(matrix[Matrix3.COLUMN2ROW0] == 0.0)
    assert(matrix[Matrix3.COLUMN0ROW1] == 0.0)
    assert(matrix[Matrix3.COLUMN1ROW1] == 0.0)
    assert(matrix[Matrix3.COLUMN2ROW1] == 0.0)
    assert(matrix[Matrix3.COLUMN0ROW2] == 0.0)
    assert(matrix[Matrix3.COLUMN1ROW2] == 0.0)
    assert(matrix[Matrix3.COLUMN2ROW2] == 0.0)

    matrix = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    assert(matrix[Matrix3.COLUMN0ROW0] == 1.0)
    assert(matrix[Matrix3.COLUMN1ROW0] == 2.0)
    assert(matrix[Matrix3.COLUMN2ROW0] == 3.0)
    assert(matrix[Matrix3.COLUMN0ROW1] == 4.0)
    assert(matrix[Matrix3.COLUMN1ROW1] == 5.0)
    assert(matrix[Matrix3.COLUMN2ROW1] == 6.0)
    assert(matrix[Matrix3.COLUMN0ROW2] == 7.0)
    assert(matrix[Matrix3.COLUMN1ROW2] == 8.0)
    assert(matrix[Matrix3.COLUMN2ROW2] == 9.0)

    sPiOver4 = math.sin(PI_OVER_FOUR)
    cPiOver4 = math.cos(PI_OVER_FOUR)
    sPiOver2 = math.sin(PI_OVER_TWO)
    cPiOver2 = math.cos(PI_OVER_TWO)

    tmp = Cartesian3.multiplyByScalar(
        Cartesian3(0.0, 0.0, 1.0),
        sPiOver4,
        Cartesian3()
    )
    quaternion = Quaternion(tmp.x, tmp.y, tmp.z, cPiOver4)
    expected = Matrix3(
        cPiOver2,
        -sPiOver2,
        0.0,
        sPiOver2,
        cPiOver2,
        0.0,
        0.0,
        0.0,
        1.0
    )

    returnedResult = Matrix3.fromQuaternion(quaternion)
    assert(Matrix3.equalsEpsilon(returnedResult, expected, EPSILON15))

    sPiOver4 = math.sin(PI_OVER_FOUR)
    cPiOver4 = math.cos(PI_OVER_FOUR)
    sPiOver2 = math.sin(PI_OVER_TWO)
    cPiOver2 = math.cos(PI_OVER_TWO)

    tmp = Cartesian3.multiplyByScalar(
        Cartesian3(0.0, 0.0, 1.0),
        sPiOver4,
        Cartesian3()
    )
    quaternion = Quaternion(tmp.x, tmp.y, tmp.z, cPiOver4)
    expected = Matrix3(
        cPiOver2,
        -sPiOver2,
        0.0,
        sPiOver2,
        cPiOver2,
        0.0,
        0.0,
        0.0,
        1.0
    )
    result = Matrix3()
    returnedResult = Matrix3.fromQuaternion(quaternion, result)
    assert(result is returnedResult)
    assert(Matrix3.equalsEpsilon(returnedResult, expected, EPSILON15))

    sPiOver4 = math.sin(PI_OVER_FOUR)
    cPiOver4 = math.cos(PI_OVER_FOUR)
    sPiOver2 = math.sin(PI_OVER_TWO)
    cPiOver2 = math.cos(PI_OVER_TWO)

    tmp = Cartesian3.multiplyByScalar(
        Cartesian3(0.0, 0.0, 1.0),
        sPiOver4,
        Cartesian3()
    )
    quaternion = Quaternion(tmp.x, tmp.y, tmp.z, cPiOver4)
    headingPitchRoll = Quaternion.toHeadingPitchRoll(quaternion)
    expected = Matrix3(
        cPiOver2,
        -sPiOver2,
        0.0,
        sPiOver2,
        cPiOver2,
        0.0,
        0.0,
        0.0,
        1.0
    )

    returnedResult = Matrix3.fromHeadingPitchRoll(headingPitchRoll)
    assert(Matrix3.equalsEpsilon(returnedResult, expected, EPSILON15))

    sPiOver4 = math.sin(PI_OVER_FOUR)
    cPiOver4 = math.cos(PI_OVER_FOUR)
    sPiOver2 = math.sin(PI_OVER_TWO)
    cPiOver2 = math.cos(PI_OVER_TWO)

    tmp = Cartesian3.multiplyByScalar(
        Cartesian3(0.0, 0.0, 1.0),
        sPiOver4,
        Cartesian3()
    )
    quaternion = Quaternion(tmp.x, tmp.y, tmp.z, cPiOver4)
    headingPitchRoll = Quaternion.toHeadingPitchRoll(quaternion)
    expected = Matrix3(
        cPiOver2,
        -sPiOver2,
        0.0,
        sPiOver2,
        cPiOver2,
        0.0,
        0.0,
        0.0,
        1.0
    )
    result = Matrix3()
    returnedResult = Matrix3.fromHeadingPitchRoll(
        headingPitchRoll,
        result
    )
    assert(result is returnedResult)
    assert(Matrix3.equalsEpsilon(returnedResult, expected, EPSILON15))

    # Expected generated via STK Components
    expected = Matrix3(
        0.754406506735489,
        0.418940943945763,
        0.505330889696038,
        0.133022221559489,
        0.656295369162553,
        -0.742685314912828,
        -0.642787609686539,
        0.627506871597133,
        0.439385041770705
    )

    headingPitchRoll = {
        'heading': -np.radians(10),
        'pitch': -np.radians(40),
        'roll': np.radians(55)
    }
    result = Matrix3()
    returnedResult = Matrix3.fromHeadingPitchRoll(
        headingPitchRoll,
        result
    )
    assert(result is returnedResult)
    assert(Matrix3.equalsEpsilon(returnedResult, expected, EPSILON15))

    expected = Matrix3(7.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 9.0)
    returnedResult = Matrix3.fromScale(Cartesian3(7.0, 8.0, 9.0))
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = Matrix3(7.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 9.0)
    result = Matrix3()
    returnedResult = Matrix3.fromScale(
        Cartesian3(7.0, 8.0, 9.0),
        result
    )

    assert(returnedResult is result)
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = Matrix3(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0)
    returnedResult = Matrix3.fromUniformScale(2.0)
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = Matrix3(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0)
    result = Matrix3()
    returnedResult = Matrix3.fromUniformScale(2.0, result)
    assert(returnedResult is result)
    assert(returnedResult == expected)

    expected = Matrix3(
        0.0,
        -3.0,
        -2.0,
        3.0,
        0.0,
        -1.0,
        2.0,
        1.0,
        0.0
    )
    left = Cartesian3(1.0, -2.0, 3.0)
    returnedResult = Matrix3.fromCrossProduct(left)
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    right = Cartesian3(2.0, 3.0, 4.0)
    crossProductExpected = Cartesian3(-17.0, 2.0, 7.0)

    crossProductResult = Cartesian3()
    # Check Cartesian3 cross product.
    crossProductResult = Cartesian3.cross(left, right, crossProductResult)
    assert(crossProductResult == crossProductExpected)

    # Check Matrix3 cross product equivalent.
    crossProductResult = Matrix3.multiplyByVector(
        returnedResult,
        right,
        crossProductResult
    )
    assert(crossProductResult == crossProductExpected)

    expected = Matrix3(
        0.0,
        -3.0,
        -2.0,
        3.0,
        0.0,
        -1.0,
        2.0,
        1.0,
        0.0
    )
    left = Cartesian3(1.0, -2.0, 3.0)
    result = Matrix3()
    returnedResult = Matrix3.fromCrossProduct(left, result)
    assert(returnedResult is result)
    assert(returnedResult == expected)

    right = Cartesian3(2.0, 3.0, 4.0)
    crossProductExpected = Cartesian3(-17.0, 2.0, 7.0)

    crossProductResult = Cartesian3()
    # Check Cartesian3 cross product.
    crossProductResult = Cartesian3.cross(left, right, crossProductResult)
    assert(crossProductResult == crossProductExpected)

    # Check Matrix3 cross product equivalent.
    crossProductResult = Matrix3.multiplyByVector(
        returnedResult,
        right,
        crossProductResult
    )
    assert(crossProductResult == crossProductExpected)

    expected = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    matrix = Matrix3.fromArray([
        1.0,
        4.0,
        7.0,
        2.0,
        5.0,
        8.0,
        3.0,
        6.0,
        9.0,
    ])
    assert(matrix == expected)

    expected = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    result = Matrix3()
    matrix = Matrix3.fromArray(
        [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
        0,
        result
    )
    assert(matrix is result)
    assert(matrix == expected)

    expected = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    result = Matrix3()
    matrix = Matrix3.fromArray(
        [0.0, 0.0, 0.0, 1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
        3,
        result
    )
    assert(matrix is result)
    assert(matrix == expected)

    expected = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    matrix = Matrix3.fromRowMajorArray([
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
    ])
    assert(matrix == expected)

    expected = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    result = Matrix3()
    matrix = Matrix3.fromRowMajorArray(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        result
    )
    assert(matrix is result)
    assert(matrix == expected)

    expected = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    matrix = Matrix3.fromColumnMajorArray([
        1.0,
        4.0,
        7.0,
        2.0,
        5.0,
        8.0,
        3.0,
        6.0,
        9.0,
    ])
    assert(matrix == expected)

    expected = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    result = Matrix3()
    matrix = Matrix3.fromColumnMajorArray(
        [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
        result
    )
    assert(matrix is result)
    assert(matrix == expected)

    matrix = Matrix3.fromRotationX(0.0)
    assert(matrix == Matrix3.IDENTITY())

    expected = Matrix3(1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
    result = Matrix3()
    matrix = Matrix3.fromRotationX(np.radians(90.0), result)
    assert(matrix is result)
    assert(Matrix3.equalsEpsilon(matrix, expected, EPSILON15))

    matrix = Matrix3.fromRotationY(0.0)
    assert(matrix == Matrix3.IDENTITY())

    expected = Matrix3(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
    result = Matrix3()
    matrix = Matrix3.fromRotationY(np.radians(90.0), result)
    assert(matrix is result)
    assert(Matrix3.equalsEpsilon(matrix, expected, EPSILON15))

    matrix = Matrix3.fromRotationZ(0.0)
    assert(matrix == Matrix3.IDENTITY())

    expected = Matrix3(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    result = Matrix3()
    matrix = Matrix3.fromRotationZ(np.radians(90.0), result)
    assert(matrix is result)
    assert(Matrix3.equalsEpsilon(matrix, expected, EPSILON15))

    expected = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    returnedResult = Matrix3.clone(expected)
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    result = Matrix3()
    returnedResult = Matrix3.clone(expected, result)
    assert(returnedResult is result)
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    returnedResult = Matrix3.toArray(
        Matrix3.fromColumnMajorArray(expected)
    )
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    result = [0.0] * 9
    returnedResult = Matrix3.toArray(
        Matrix3.fromColumnMajorArray(expected),
        result
    )
    assert(returnedResult is result)
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    i = 0
    for col in range(0, 3):
        for row in range(0, 3):
            index = Matrix3.getElementIndex(col, row)
            assert(index == i)
            i = i + 1

    matrix = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    expectedColumn0 = Cartesian3(1.0, 4.0, 7.0)
    expectedColumn1 = Cartesian3(2.0, 5.0, 8.0)
    expectedColumn2 = Cartesian3(3.0, 6.0, 9.0)

    resultColumn0 = Cartesian3()
    resultColumn1 = Cartesian3()
    resultColumn2 = Cartesian3()
    returnedResultColumn0 = Matrix3.getColumn(matrix, 0, resultColumn0)
    returnedResultColumn1 = Matrix3.getColumn(matrix, 1, resultColumn1)
    returnedResultColumn2 = Matrix3.getColumn(matrix, 2, resultColumn2)

    assert(resultColumn0 is returnedResultColumn0)
    assert(resultColumn0 == expectedColumn0)
    assert(resultColumn1 is returnedResultColumn1)
    assert(resultColumn1 == expectedColumn1)
    assert(resultColumn2 is returnedResultColumn2)
    assert(resultColumn2 == expectedColumn2)

    matrix = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    result = Matrix3()

    expected = Matrix3(10.0, 2.0, 3.0, 11.0, 5.0, 6.0, 12.0, 8.0, 9.0)
    returnedResult = Matrix3.setColumn(
        matrix,
        0,
        Cartesian3(10.0, 11.0, 12.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    expected = Matrix3(1.0, 13.0, 3.0, 4.0, 14.0, 6.0, 7.0, 15.0, 9.0)
    returnedResult = Matrix3.setColumn(
        matrix,
        1,
        Cartesian3(13.0, 14.0, 15.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    expected = Matrix3(1.0, 2.0, 16.0, 4.0, 5.0, 17.0, 7.0, 8.0, 18.0)
    returnedResult = Matrix3.setColumn(
        matrix,
        2,
        Cartesian3(16.0, 17.0, 18.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    matrix = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    expectedRow0 = Cartesian3(1.0, 2.0, 3.0)
    expectedRow1 = Cartesian3(4.0, 5.0, 6.0)
    expectedRow2 = Cartesian3(7.0, 8.0, 9.0)

    resultRow0 = Cartesian3()
    resultRow1 = Cartesian3()
    resultRow2 = Cartesian3()
    returnedResultRow0 = Matrix3.getRow(matrix, 0, resultRow0)
    returnedResultRow1 = Matrix3.getRow(matrix, 1, resultRow1)
    returnedResultRow2 = Matrix3.getRow(matrix, 2, resultRow2)

    assert(resultRow0 is returnedResultRow0)
    assert(resultRow0 == expectedRow0)
    assert(resultRow1 is returnedResultRow1)
    assert(resultRow1 == expectedRow1)
    assert(resultRow2 is returnedResultRow2)
    assert(resultRow2 == expectedRow2)

    matrix = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    result = Matrix3()

    expected = Matrix3(10.0, 11.0, 12.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    returnedResult = Matrix3.setRow(
        matrix,
        0,
        Cartesian3(10.0, 11.0, 12.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    expected = Matrix3(1.0, 2.0, 3.0, 13.0, 14.0, 15.0, 7.0, 8.0, 9.0)
    returnedResult = Matrix3.setRow(
        matrix,
        1,
        Cartesian3(13.0, 14.0, 15.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    expected = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 16.0, 17.0, 18.0)
    returnedResult = Matrix3.setRow(
        matrix,
        2,
        Cartesian3(16.0, 17.0, 18.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    oldScale = Cartesian3(2.0, 3.0, 4.0)
    newScale = Cartesian3(5.0, 6.0, 7.0)

    matrix = Matrix3.fromScale(oldScale, Matrix3())
    result = Matrix3()

    assert(Matrix3.getScale(matrix, Cartesian3()) == oldScale)

    returnedResult = Matrix3.setScale(matrix, newScale, result)

    assert(Matrix3.getScale(returnedResult, Cartesian3()) ==
           newScale
           )
    assert(result is returnedResult)

    oldScale = Cartesian3(2.0, 3.0, 4.0)
    newScale = 5.0

    matrix = Matrix3.fromScale(oldScale, Matrix3())
    result = Matrix3()

    assert(Matrix3.getScale(matrix, Cartesian3()) == oldScale)

    returnedResult = Matrix3.setUniformScale(matrix, newScale, result)

    assert(Matrix3.getScale(returnedResult, Cartesian3()) ==
           Cartesian3(newScale, newScale, newScale)
           )
    assert(result is returnedResult)

    scale = Cartesian3(2.0, 3.0, 4.0)
    result = Cartesian3()
    computedScale = Matrix3.getScale(Matrix3.fromScale(scale), result)

    assert(computedScale is result)
    assert(Cartesian3.equalsEpsilon(computedScale, scale, EPSILON14))

    m = Matrix3.fromScale(Cartesian3(2.0, 3.0, 4.0))
    assert(np.isclose(Matrix3.getMaximumScale(m),
                      4.0,
                      EPSILON14
                      ))

    scaleVec = Cartesian3(2.0, 3.0, 4.0)
    scale = Matrix3.fromScale(scaleVec, Matrix3())
    rotation = Matrix3.fromRotationX(0.5, Matrix3())
    scaleRotation = Matrix3.setRotation(scale, rotation, Matrix3())

    extractedScale = Matrix3.getScale(scaleRotation, Cartesian3())
    extractedRotation = Matrix3.getRotation(scaleRotation, Matrix3())

    assert(Cartesian3.equalsEpsilon(extractedScale, scaleVec, EPSILON14))
    assert(Matrix3.equalsEpsilon(extractedRotation, rotation, EPSILON14))

    matrix = Matrix3.fromColumnMajorArray([
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
    ])
    expectedRotation = Matrix3.fromArray([
        1.0 / math.sqrt(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0),
        2.0 / math.sqrt(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0),
        3.0 / math.sqrt(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0),
        4.0 / math.sqrt(4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0),
        5.0 / math.sqrt(4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0),
        6.0 / math.sqrt(4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0),
        7.0 / math.sqrt(7.0 * 7.0 + 8.0 * 8.0 + 9.0 * 9.0),
        8.0 / math.sqrt(7.0 * 7.0 + 8.0 * 8.0 + 9.0 * 9.0),
        9.0 / math.sqrt(7.0 * 7.0 + 8.0 * 8.0 + 9.0 * 9.0),
    ])
    rotation = Matrix3.getRotation(matrix, Matrix3())
    assert(Matrix3.equalsEpsilon(rotation, expectedRotation, EPSILON14))

    matrix = Matrix3.fromColumnMajorArray([
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
    ])
    duplicateMatrix = Matrix3.clone(matrix, Matrix3())
    expectedRotation = Matrix3.fromArray([
        1.0 / math.sqrt(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0),
        2.0 / math.sqrt(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0),
        3.0 / math.sqrt(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0),
        4.0 / math.sqrt(4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0),
        5.0 / math.sqrt(4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0),
        6.0 / math.sqrt(4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0),
        7.0 / math.sqrt(7.0 * 7.0 + 8.0 * 8.0 + 9.0 * 9.0),
        8.0 / math.sqrt(7.0 * 7.0 + 8.0 * 8.0 + 9.0 * 9.0),
        9.0 / math.sqrt(7.0 * 7.0 + 8.0 * 8.0 + 9.0 * 9.0),
    ])
    result = Matrix3.getRotation(matrix, Matrix3())
    assert(Matrix3.equalsEpsilon(result, expectedRotation, EPSILON14))
    assert(matrix == duplicateMatrix)
    assert(matrix is not result)

    left = Matrix3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    right = Matrix3(10, 11, 12, 13, 14, 15, 16, 17, 18)
    expected = Matrix3(84, 90, 96, 201, 216, 231, 318, 342, 366)
    result = Matrix3()
    returnedResult = Matrix3.multiply(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    left = Matrix3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    right = Matrix3(10, 11, 12, 13, 14, 15, 16, 17, 18)
    expected = Matrix3(84, 90, 96, 201, 216, 231, 318, 342, 366)
    returnedResult = Matrix3.multiply(left, right, left)
    assert(returnedResult is left)
    assert(left == expected)

    left = Matrix3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    right = Matrix3(10, 11, 12, 13, 14, 15, 16, 17, 18)
    expected = Matrix3(11, 13, 15, 17, 19, 21, 23, 25, 27)
    result = Matrix3()
    returnedResult = Matrix3.add(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    left = Matrix3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    right = Matrix3(10, 11, 12, 13, 14, 15, 16, 17, 18)
    expected = Matrix3(11, 13, 15, 17, 19, 21, 23, 25, 27)
    returnedResult = Matrix3.add(left, right, left)
    assert(returnedResult is left)
    assert(left == expected)

    left = Matrix3(11, 13, 15, 17, 19, 21, 23, 25, 27)
    right = Matrix3(10, 11, 12, 13, 14, 15, 16, 17, 18)
    expected = Matrix3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    result = Matrix3()
    returnedResult = Matrix3.subtract(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    left = Matrix3(11, 13, 15, 17, 19, 21, 23, 25, 27)
    right = Matrix3(10, 11, 12, 13, 14, 15, 16, 17, 18)
    expected = Matrix3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    returnedResult = Matrix3.subtract(left, right, left)
    assert(returnedResult is left)
    assert(left == expected)

    m = Matrix3(2, 3, 4, 6, 7, 8, 10, 11, 12)
    scale = Cartesian3(2.0, 3.0, 4.0)
    expected = Matrix3.multiply(
        m,
        Matrix3.fromScale(scale),
        Matrix3()
    )
    result = Matrix3()
    returnedResult = Matrix3.multiplyByScale(m, scale, result)
    assert(returnedResult is result)
    assert(result == expected)

    m = Matrix3(1, 2, 3, 5, 6, 7, 9, 10, 11)
    scale = Cartesian3(1.0, 2.0, 3.0)
    expected = Matrix3.multiply(
        m,
        Matrix3.fromScale(scale),
        Matrix3()
    )
    returnedResult = Matrix3.multiplyByScale(m, scale, m)
    assert(returnedResult is m)
    assert(m == expected)

    m = Matrix3(2, 3, 4, 5, 6, 7, 8, 9, 10)
    scale = 2.0
    expected = Matrix3.multiply(
        m,
        Matrix3.fromUniformScale(scale),
        Matrix3()
    )
    result = Matrix3()
    returnedResult = Matrix3.multiplyByUniformScale(m, scale, result)
    assert(returnedResult is result)
    assert(result == expected)

    m = Matrix3(2, 3, 4, 5, 6, 7, 8, 9, 10)
    scale = 2.0
    expected = Matrix3.multiply(
        m,
        Matrix3.fromUniformScale(scale),
        Matrix3()
    )
    returnedResult = Matrix3.multiplyByUniformScale(m, scale, m)
    assert(returnedResult is m)
    assert(m == expected)

    left = Matrix3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    right = Cartesian3(10, 11, 12)
    expected = Cartesian3(68, 167, 266)
    result = Cartesian3()
    returnedResult = Matrix3.multiplyByVector(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    left = Matrix3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    right = 2
    expected = Matrix3(2, 4, 6, 8, 10, 12, 14, 16, 18)
    result = Matrix3()
    returnedResult = Matrix3.multiplyByScalar(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    matrix = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    expected = Matrix3(
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
        -7.0,
        -8.0,
        -9.0
    )
    result = Matrix3()
    returnedResult = Matrix3.negate(matrix, result)
    assert(result is returnedResult)
    assert(result == expected)

    matrix = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    expected = Matrix3(
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
        -7.0,
        -8.0,
        -9.0
    )
    returnedResult = Matrix3.negate(matrix, matrix)
    assert(matrix is returnedResult)
    assert(matrix == expected)

    matrix = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    expected = Matrix3(1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0)
    result = Matrix3()
    returnedResult = Matrix3.transpose(matrix, result)
    assert(result is returnedResult)
    assert(result == expected)

    matrix = Matrix3(1.0, 5.0, 2.0, 1.0, 1.0, 7.0, 0.0, -3.0, 4.0)
    expectedInverse = Matrix3.inverse(matrix, Matrix3())
    expectedInverseTranspose = Matrix3.transpose(
        expectedInverse,
        Matrix3()
    )
    result = Matrix3.inverseTranspose(matrix, Matrix3())
    assert(result == expectedInverseTranspose)

    matrix = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    expected = Matrix3(1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0)
    returnedResult = Matrix3.transpose(matrix, matrix)
    assert(matrix is returnedResult)
    assert(matrix == expected)

    matrix = Matrix3(1.0, 5.0, 2.0, 1.0, 1.0, 7.0, 0.0, -3.0, 4.0)
    expected = -1.0
    result = Matrix3.determinant(matrix)
    assert(result == expected)

    matrix = Matrix3(1.0, 5.0, 2.0, 1.0, 1.0, 7.0, 0.0, -3.0, 4.0)
    expected = Matrix3(
        -25.0,
        26.0,
        -33.0,
        4.0,
        -4.0,
        5.0,
        3.0,
        -3.0,
        4.0
    )
    result = Matrix3()
    returnedResult = Matrix3.inverse(matrix, result)
    assert(result is returnedResult)
    assert(result == expected)

    matrix = Matrix3(1.0, 5.0, 2.0, 1.0, 1.0, 7.0, 0.0, -3.0, 4.0)
    expected = Matrix3(
        -25.0,
        26.0,
        -33.0,
        4.0,
        -4.0,
        5.0,
        3.0,
        -3.0,
        4.0
    )
    returnedResult = Matrix3.inverse(matrix, matrix)
    assert(matrix is returnedResult)
    assert(matrix == expected)

    with pytest.raises(Exception) as exc_info:
        Matrix3.inverse(Matrix3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), Matrix3())

    assert(exc_info.value.args[0] == 'matrix is not invertible')

    a = Matrix3(4.0, -1.0, 1.0, -1.0, 3.0, -2.0, 1.0, -2.0, 3.0)

    expectedDiagonal = Matrix3(
        3.0,
        0.0,
        0.0,
        0.0,
        6.0,
        0.0,
        0.0,
        0.0,
        1.0
    )

    decomposition = Matrix3.computeEigenDecomposition(a)
    assert(Matrix3.equalsEpsilon(decomposition['diagonal'],
                                 expectedDiagonal,
                                 EPSILON14
                                 ))

    v = Matrix3.getColumn(decomposition['unitary'], 0, Cartesian3())
    lamb = Matrix3.getColumn(decomposition['diagonal'], 0, Cartesian3()).x
    assert(
        Cartesian3.equalsEpsilon(Cartesian3.multiplyByScalar(v, lamb, Cartesian3()),
                                 Matrix3.multiplyByVector(a, v, Cartesian3()),
                                 EPSILON14
                                 ))

    v = Matrix3.getColumn(decomposition['unitary'], 1, Cartesian3())
    lamb = Matrix3.getColumn(decomposition['diagonal'], 1, Cartesian3()).y
    assert(
        Cartesian3.equalsEpsilon(Cartesian3.multiplyByScalar(v, lamb, Cartesian3()),
                                 Matrix3.multiplyByVector(a, v, Cartesian3()),
                                 EPSILON14
                                 ))

    v = Matrix3.getColumn(decomposition['unitary'], 2, Cartesian3())
    lamb = Matrix3.getColumn(decomposition['diagonal'], 2, Cartesian3()).z
    assert(
        Cartesian3.equalsEpsilon(Cartesian3.multiplyByScalar(v, lamb, Cartesian3()),
                                 Matrix3.multiplyByVector(a, v, Cartesian3()),
                                 EPSILON14
                                 ))

    a = Matrix3(4.0, -1.0, 1.0, -1.0, 3.0, -2.0, 1.0, -2.0, 3.0)

    expectedDiagonal = Matrix3(
        3.0,
        0.0,
        0.0,
        0.0,
        6.0,
        0.0,
        0.0,
        0.0,
        1.0
    )
    result = {
        'unitary': Matrix3(),
        'diagonal': Matrix3(),
    }

    decomposition = Matrix3.computeEigenDecomposition(a, result)
    assert(decomposition is result)
    assert(Matrix3.equalsEpsilon(decomposition['diagonal'],
                                 expectedDiagonal,
                                 EPSILON14
                                 ))

    v = Matrix3.getColumn(decomposition['unitary'], 0, Cartesian3())
    lamb = Matrix3.getColumn(decomposition['diagonal'], 0, Cartesian3()).x
    assert(
        Cartesian3.equalsEpsilon(Cartesian3.multiplyByScalar(v, lamb, Cartesian3()),
                                 Matrix3.multiplyByVector(a, v, Cartesian3()),
                                 EPSILON14
                                 ))

    v = Matrix3.getColumn(decomposition['unitary'], 1, Cartesian3())
    lamb = Matrix3.getColumn(decomposition['diagonal'], 1, Cartesian3()).y
    assert(
        Cartesian3.equalsEpsilon(Cartesian3.multiplyByScalar(v, lamb, Cartesian3()),
                                 Matrix3.multiplyByVector(a, v, Cartesian3()),
                                 EPSILON14
                                 ))

    v = Matrix3.getColumn(decomposition['unitary'], 2, Cartesian3())
    lamb = Matrix3.getColumn(decomposition['diagonal'], 2, Cartesian3()).z
    assert(
        Cartesian3.equalsEpsilon(Cartesian3.multiplyByScalar(v, lamb, Cartesian3()),
                                 Matrix3.multiplyByVector(a, v, Cartesian3()),
                                 EPSILON14
                                 ))

    matrix = Matrix3(
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
        -7.0,
        -8.0,
        -9.0
    )
    expected = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    result = Matrix3()
    returnedResult = Matrix3.abs(matrix, result)
    assert(returnedResult == expected)

    matrix = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    returnedResult = Matrix3.abs(matrix, result)
    assert(returnedResult == expected)

    matrix = Matrix3(1.0, -2.0, -3.0, 4.0, 5.0, -6.0, 7.0, -8.0, 9.0)
    returnedResult = Matrix3.abs(matrix, result)
    assert(returnedResult == expected)

    matrix = Matrix3(
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
        -7.0,
        -8.0,
        -9.0
    )
    expected = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    returnedResult = Matrix3.abs(matrix, matrix)
    assert(matrix is returnedResult)
    assert(matrix == expected)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    assert(left == right)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    assert(left != right)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    assert(left != right)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 2.0, 7.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    assert(left != right)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 2.0, 3.0, 8.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    assert(left != right)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 2.0, 3.0, 4.0, 9.0, 6.0, 7.0, 8.0, 9.0)
    assert(left != right)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 7.0, 8.0, 9.0)
    assert(left != right)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 8.0, 9.0)
    assert(left != right)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 12.0, 9.0)
    assert(left != right)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 13.0)
    assert(left != right)

    assert(Matrix3.equals(None, None) is True)
    assert(Matrix3.equals(Matrix3(), None) is False)
    assert(Matrix3.equals(None, Matrix3()) is False)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    assert(Matrix3.equalsEpsilon(left, right, 1.0) is True)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    assert(Matrix3.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix3.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    assert(Matrix3.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix3.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 2.0, 7.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    assert(Matrix3.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix3.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 2.0, 3.0, 8.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    assert(Matrix3.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix3.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 2.0, 3.0, 4.0, 9.0, 6.0, 7.0, 8.0, 9.0)
    assert(Matrix3.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix3.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 7.0, 8.0, 9.0)
    assert(Matrix3.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix3.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 8.0, 9.0)
    assert(Matrix3.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix3.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 12.0, 9.0)
    assert(Matrix3.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix3.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    right = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 13.0)
    assert(Matrix3.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix3.equalsEpsilon(left, right, 4.0) is True)

    assert(Matrix3.equalsEpsilon(None, None, 1.0) is True)
    assert(Matrix3.equalsEpsilon(Matrix3(), None, 1.0) is False)
    assert(Matrix3.equalsEpsilon(None, Matrix3(), 1.0) is False)

    matrix = Matrix3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert(str(matrix) == "[1, 2, 3]\n[4, 5, 6]\n[7, 8, 9]")

    assert(Matrix3.clone(None) is None)

    matrix = Matrix3(1, 4, 7, 2, 5, 8, 3, 6, 9)
    assert(len(matrix) == 9)

    for index in range(0, len(matrix)):
        assert(matrix[index] == index + 1)

    matrix = Matrix3.IDENTITY()
    assert(matrix[Matrix3.COLUMN0ROW0] == 1.0)
    assert(matrix[Matrix3.COLUMN1ROW0] == 0.0)
    assert(matrix[Matrix3.COLUMN2ROW0] == 0.0)
    assert(matrix[Matrix3.COLUMN0ROW1] == 0.0)
    assert(matrix[Matrix3.COLUMN1ROW1] == 1.0)
    assert(matrix[Matrix3.COLUMN2ROW1] == 0.0)
    assert(matrix[Matrix3.COLUMN0ROW2] == 0.0)
    assert(matrix[Matrix3.COLUMN1ROW2] == 0.0)
    assert(matrix[Matrix3.COLUMN2ROW2] == 1.0)

    matrix = Matrix3.ZERO()
    assert(matrix[Matrix3.COLUMN0ROW0] == 0.0)
    assert(matrix[Matrix3.COLUMN1ROW0] == 0.0)
    assert(matrix[Matrix3.COLUMN2ROW0] == 0.0)
    assert(matrix[Matrix3.COLUMN0ROW1] == 0.0)
    assert(matrix[Matrix3.COLUMN1ROW1] == 0.0)
    assert(matrix[Matrix3.COLUMN2ROW1] == 0.0)
    assert(matrix[Matrix3.COLUMN0ROW2] == 0.0)
    assert(matrix[Matrix3.COLUMN1ROW2] == 0.0)
    assert(matrix[Matrix3.COLUMN2ROW2] == 0.0)


def test_matrix4():

    matrix = Matrix4()
    assert(matrix[Matrix4.COLUMN0ROW0] == 0.0)
    assert(matrix[Matrix4.COLUMN1ROW0] == 0.0)
    assert(matrix[Matrix4.COLUMN2ROW0] == 0.0)
    assert(matrix[Matrix4.COLUMN3ROW0] == 0.0)
    assert(matrix[Matrix4.COLUMN0ROW1] == 0.0)
    assert(matrix[Matrix4.COLUMN1ROW1] == 0.0)
    assert(matrix[Matrix4.COLUMN2ROW1] == 0.0)
    assert(matrix[Matrix4.COLUMN3ROW1] == 0.0)
    assert(matrix[Matrix4.COLUMN0ROW2] == 0.0)
    assert(matrix[Matrix4.COLUMN1ROW2] == 0.0)
    assert(matrix[Matrix4.COLUMN2ROW2] == 0.0)
    assert(matrix[Matrix4.COLUMN3ROW2] == 0.0)
    assert(matrix[Matrix4.COLUMN0ROW3] == 0.0)
    assert(matrix[Matrix4.COLUMN1ROW3] == 0.0)
    assert(matrix[Matrix4.COLUMN2ROW3] == 0.0)
    assert(matrix[Matrix4.COLUMN3ROW3] == 0.0)

    matrix = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    assert(matrix[Matrix4.COLUMN0ROW0] == 1.0)
    assert(matrix[Matrix4.COLUMN1ROW0] == 2.0)
    assert(matrix[Matrix4.COLUMN2ROW0] == 3.0)
    assert(matrix[Matrix4.COLUMN3ROW0] == 4.0)
    assert(matrix[Matrix4.COLUMN0ROW1] == 5.0)
    assert(matrix[Matrix4.COLUMN1ROW1] == 6.0)
    assert(matrix[Matrix4.COLUMN2ROW1] == 7.0)
    assert(matrix[Matrix4.COLUMN3ROW1] == 8.0)
    assert(matrix[Matrix4.COLUMN0ROW2] == 9.0)
    assert(matrix[Matrix4.COLUMN1ROW2] == 10.0)
    assert(matrix[Matrix4.COLUMN2ROW2] == 11.0)
    assert(matrix[Matrix4.COLUMN3ROW2] == 12.0)
    assert(matrix[Matrix4.COLUMN0ROW3] == 13.0)
    assert(matrix[Matrix4.COLUMN1ROW3] == 14.0)
    assert(matrix[Matrix4.COLUMN2ROW3] == 15.0)
    assert(matrix[Matrix4.COLUMN3ROW3] == 16.0)

    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    matrix = Matrix4.fromArray([
        1.0,
        5.0,
        9.0,
        13.0,
        2.0,
        6.0,
        10.0,
        14.0,
        3.0,
        7.0,
        11.0,
        15.0,
        4.0,
        8.0,
        12.0,
        16.0,
    ])
    assert(matrix == expected)

    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    result = Matrix4()
    matrix = Matrix4.fromArray(
        [
            1.0,
            5.0,
            9.0,
            13.0,
            2.0,
            6.0,
            10.0,
            14.0,
            3.0,
            7.0,
            11.0,
            15.0,
            4.0,
            8.0,
            12.0,
            16.0,
        ],
        0,
        result
    )
    assert(matrix is result)
    assert(matrix == expected)

    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    result = Matrix4()
    matrix = Matrix4.fromArray(
        [
            0.0,
            0.0,
            0.0,
            1.0,
            5.0,
            9.0,
            13.0,
            2.0,
            6.0,
            10.0,
            14.0,
            3.0,
            7.0,
            11.0,
            15.0,
            4.0,
            8.0,
            12.0,
            16.0,
        ],
        3,
        result
    )
    assert(matrix is result)
    assert(matrix == expected)

    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    matrix = Matrix4.fromRowMajorArray([
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ])
    assert(matrix == expected)

    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    result = Matrix4()
    matrix = Matrix4.fromRowMajorArray(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
        ],
        result
    )
    assert(matrix is result)
    assert(matrix == expected)

    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    matrix = Matrix4.fromColumnMajorArray([
        1.0,
        5.0,
        9.0,
        13.0,
        2.0,
        6.0,
        10.0,
        14.0,
        3.0,
        7.0,
        11.0,
        15.0,
        4.0,
        8.0,
        12.0,
        16.0,
    ])
    assert(matrix == expected)

    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    result = Matrix4()
    matrix = Matrix4.fromColumnMajorArray(
        [
            1.0,
            5.0,
            9.0,
            13.0,
            2.0,
            6.0,
            10.0,
            14.0,
            3.0,
            7.0,
            11.0,
            15.0,
            4.0,
            8.0,
            12.0,
            16.0,
        ],
        result
    )
    assert(matrix is result)
    assert(matrix == expected)

    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    returnedResult = Matrix4.clone(expected)
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    result = Matrix4()
    returnedResult = Matrix4.clone(expected, result)
    assert(returnedResult is result)
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        10.0,
        4.0,
        5.0,
        6.0,
        11.0,
        7.0,
        8.0,
        9.0,
        12.0,
        0.0,
        0.0,
        0.0,
        1.0
    )
    returnedResult = Matrix4.fromRotationTranslation(
        Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
        Cartesian3(10.0, 11.0, 12.0)
    )
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        10.0,
        4.0,
        5.0,
        6.0,
        11.0,
        7.0,
        8.0,
        9.0,
        12.0,
        0.0,
        0.0,
        0.0,
        1.0
    )
    result = Matrix4()
    returnedResult = Matrix4.fromRotationTranslation(
        Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
        Cartesian3(10.0, 11.0, 12.0),
        result
    )
    assert(returnedResult is result)
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = Matrix4(
        1.0,
        0.0,
        0.0,
        10.0,
        0.0,
        1.0,
        0.0,
        11.0,
        0.0,
        0.0,
        1.0,
        12.0,
        0.0,
        0.0,
        0.0,
        1.0
    )
    returnedResult = Matrix4.fromTranslation(
        Cartesian3(10.0, 11.0, 12.0)
    )
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = Matrix4(
        7.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        9.0,
        2.0,
        0.0,
        -8.0,
        0.0,
        3.0,
        0.0,
        0.0,
        0.0,
        1.0
    )
    returnedResult = Matrix4.fromTranslationQuaternionRotationScale(
        Cartesian3(1.0, 2.0, 3.0),  # translation
        Quaternion.fromAxisAngle(Cartesian3.UNIT_X(), np.radians(-90.0)),  # rotation
        Cartesian3(7.0, 8.0, 9.0)
    )  # scale
    assert(returnedResult is not expected)
    assert(Matrix4.equalsEpsilon(returnedResult, expected, EPSILON14))

    expected = Matrix4(
        7.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        9.0,
        2.0,
        0.0,
        -8.0,
        0.0,
        3.0,
        0.0,
        0.0,
        0.0,
        1.0
    )
    result = Matrix4()
    returnedResult = Matrix4.fromTranslationQuaternionRotationScale(
        Cartesian3(1.0, 2.0, 3.0),  # translation
        Quaternion.fromAxisAngle(Cartesian3.UNIT_X(), np.radians(-90.0)),  # rotation
        Cartesian3(7.0, 8.0, 9.0),  # scale
        result
    )
    assert(returnedResult is result)
    assert(returnedResult is not expected)
    assert(Matrix4.equalsEpsilon(returnedResult, expected, EPSILON14))

    expected = Matrix4(
        7.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        9.0,
        2.0,
        0.0,
        -8.0,
        0.0,
        3.0,
        0.0,
        0.0,
        0.0,
        1.0
    )

    trs = {
        'translation': Cartesian3(1.0, 2.0, 3.0),
        'rotation': Quaternion.fromAxisAngle(Cartesian3.UNIT_X(), np.radians(-90.0)),
        'scale': Cartesian3(7.0, 8.0, 9.0)
    }

    returnedResult = Matrix4.fromTranslationRotationScale(trs)
    assert(returnedResult is not expected)
    assert(Matrix4.equalsEpsilon(returnedResult, expected, EPSILON14))

    expected = Matrix4(
        7.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        9.0,
        2.0,
        0.0,
        -8.0,
        0.0,
        3.0,
        0.0,
        0.0,
        0.0,
        1.0
    )

    trs = {
        'translation': Cartesian3(1.0, 2.0, 3.0),
        'rotation': Quaternion.fromAxisAngle(Cartesian3.UNIT_X(), np.radians(-90.0)),
        'scale': Cartesian3(7.0, 8.0, 9.0)
    }

    result = Matrix4()
    returnedResult = Matrix4.fromTranslationRotationScale(trs, result)
    assert(returnedResult is result)
    assert(returnedResult is not expected)
    assert(Matrix4.equalsEpsilon(returnedResult, expected, EPSILON14))

    expected = Matrix4(
        1.0,
        0.0,
        0.0,
        10.0,
        0.0,
        1.0,
        0.0,
        11.0,
        0.0,
        0.0,
        1.0,
        12.0,
        0.0,
        0.0,
        0.0,
        1.0
    )
    result = Matrix4()
    returnedResult = Matrix4.fromTranslation(
        Cartesian3(10.0, 11.0, 12.0),
        result
    )
    assert(returnedResult is result)
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = Matrix4(
        7.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8.0,
        0.0,
        0.0,
        0.0,
        0.0,
        9.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0
    )
    returnedResult = Matrix4.fromScale(Cartesian3(7.0, 8.0, 9.0))
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = Matrix4(
        7.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8.0,
        0.0,
        0.0,
        0.0,
        0.0,
        9.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0
    )
    result = Matrix4()
    returnedResult = Matrix4.fromScale(
        Cartesian3(7.0, 8.0, 9.0),
        result
    )
    assert(returnedResult is result)
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = Matrix4(
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0
    )
    returnedResult = Matrix4.fromUniformScale(2.0)
    assert(returnedResult == expected)

    expected = Matrix4(
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0
    )
    result = Matrix4()
    returnedResult = Matrix4.fromUniformScale(2.0, result)
    assert(returnedResult is result)
    assert(returnedResult == expected)

    expected = Matrix4.fromColumnMajorArray([
        1.0,
        2.0,
        3.0,
        0.0,
        4.0,
        5.0,
        6.0,
        0.0,
        7.0,
        8.0,
        9.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ])
    returnedResult = Matrix4.fromRotation(
        Matrix3.fromColumnMajorArray([
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
        ])
    )
    assert(returnedResult == expected)

    expected = Matrix4.fromColumnMajorArray([
        1.0,
        2.0,
        3.0,
        0.0,
        4.0,
        5.0,
        6.0,
        0.0,
        7.0,
        8.0,
        9.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ])

    result = Matrix4()
    returnedResult = Matrix4.fromRotation(
        Matrix3.fromColumnMajorArray([
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
        ]),
        result
    )
    assert(returnedResult is result)
    assert(returnedResult == expected)

    expected = Matrix4(
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        -1.222222222222222,
        -2.222222222222222,
        0,
        0,
        -1,
        0
    )
    result = Matrix4()
    returnedResult = Matrix4.computePerspectiveFieldOfView(
        PI_OVER_TWO,
        1,
        1,
        10,
        result
    )
    assert(Matrix4.equalsEpsilon(returnedResult, expected, EPSILON15))

    returnedResult2 = Matrix4.computePerspectiveFieldOfView(
        PI_OVER_TWO,
        1,
        1,
        10,
    )
    assert(Matrix4.equalsEpsilon(returnedResult2, expected, EPSILON15))

    expected = Matrix4.IDENTITY()
    returnedResult = Matrix4.fromCamera({
        'position': Cartesian3.ZERO(),
        'direction': Cartesian3.negate(Cartesian3.UNIT_Z(), Cartesian3()),
        'up': Cartesian3.UNIT_Y(),
    })
    assert(expected == returnedResult)

    expected = Matrix4.IDENTITY()
    result = Matrix4()
    returnedResult = Matrix4.fromCamera(
        {
            'position': Cartesian3.ZERO(),
            'direction': Cartesian3.negate(Cartesian3.UNIT_Z(), Cartesian3()),
            'up': Cartesian3.UNIT_Y(),
        },
        result
    )
    assert(returnedResult is result)
    assert(returnedResult == expected)

    expected = Matrix4(
        2,
        0,
        0,
        -1,
        0,
        2,
        0,
        -5,
        0,
        0,
        -2,
        -1,
        0,
        0,
        0,
        1
    )
    result = Matrix4()
    returnedResult = Matrix4.computeOrthographicOffCenter(
        0,
        1,
        2,
        3,
        0,
        1,
        result
    )
    assert(returnedResult is result)
    assert(returnedResult == expected)

    returnedResult2 = Matrix4.computeOrthographicOffCenter(
        0,
        1,
        2,
        3,
        0,
        1,
    )
    assert(returnedResult2 == expected)

    expected = Matrix4(
        2.0,
        0.0,
        0.0,
        2.0,
        0.0,
        3.0,
        0.0,
        3.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0
    )
    returnedResult = Matrix4.computeViewportTransformation(
        {
            'x': 0,
            'y': 0,
            'width': 4.0,
            'height': 6.0,
        },
        0.0,
        2.0
    )
    assert(returnedResult == expected)

    expected = Matrix4(
        2.0,
        0.0,
        0.0,
        2.0,
        0.0,
        3.0,
        0.0,
        3.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0
    )
    result = Matrix4()
    returnedResult = Matrix4.computeViewportTransformation(
        {
            'x': 0,
            'y': 0,
            'width': 4.0,
            'height': 6.0,
        },
        0.0,
        2.0,
        result
    )
    assert(returnedResult == expected)
    assert(returnedResult is result)

    expected = Matrix4(
        2,
        0,
        3,
        0,
        0,
        2,
        5,
        0,
        0,
        0,
        -3,
        -4,
        0,
        0,
        -1,
        0
    )
    result = Matrix4()
    returnedResult = Matrix4.computePerspectiveOffCenter(
        1,
        2,
        2,
        3,
        1,
        2,
        result
    )
    assert(returnedResult == expected)
    assert(returnedResult is result)

    returnedResult2 = Matrix4.computePerspectiveOffCenter(
        1,
        2,
        2,
        3,
        1,
        2,
    )
    assert(returnedResult2 == expected)

    expected = Matrix4(
        2,
        0,
        3,
        0,
        0,
        2,
        5,
        0,
        0,
        0,
        -1,
        -2,
        0,
        0,
        -1,
        0
    )
    result = Matrix4()
    returnedResult = Matrix4.computeInfinitePerspectiveOffCenter(
        1,
        2,
        2,
        3,
        1,
        result
    )
    assert(returnedResult == expected)
    assert(returnedResult is result)

    returnedResult2 = Matrix4.computeInfinitePerspectiveOffCenter(
        1,
        2,
        2,
        3,
        1,
    )
    assert(returnedResult2 == expected)

    expected = Matrix4(
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        0,
        1
    )

    position = Cartesian3.ZERO()
    direction = Cartesian3.UNIT_Z()
    up = Cartesian3.UNIT_Y()
    right = Cartesian3.UNIT_X()
    returnedResult = Matrix4.computeView(position, direction, up, right)
    assert(returnedResult == expected)

    expected = Matrix4(
        -1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        0,
        1
    )
    returnedResult = Matrix4.computeLookAt(position, direction, up)
    assert(returnedResult == expected)

    expected = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ]
    returnedResult = Matrix4.toArray(
        Matrix4.fromColumnMajorArray(expected)
    )
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    expected = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ]
    result = [0] * 16
    returnedResult = Matrix4.toArray(
        Matrix4.fromColumnMajorArray(expected),
        result
    )
    assert(returnedResult is result)
    assert(returnedResult is not expected)
    assert(returnedResult == expected)

    i = 0
    for col in range(0, 4):
        for row in range(0, 4):
            index = Matrix4.getElementIndex(col, row)
            assert(index == i)
            i = i + 1

    matrix = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    expectedColumn0 = Cartesian4(1.0, 5.0, 9.0, 13.0)
    expectedColumn1 = Cartesian4(2.0, 6.0, 10.0, 14.0)
    expectedColumn2 = Cartesian4(3.0, 7.0, 11.0, 15.0)
    expectedColumn3 = Cartesian4(4.0, 8.0, 12.0, 16.0)

    resultColumn0 = Cartesian4()
    resultColumn1 = Cartesian4()
    resultColumn2 = Cartesian4()
    resultColumn3 = Cartesian4()
    returnedResultColumn0 = Matrix4.getColumn(matrix, 0, resultColumn0)
    returnedResultColumn1 = Matrix4.getColumn(matrix, 1, resultColumn1)
    returnedResultColumn2 = Matrix4.getColumn(matrix, 2, resultColumn2)
    returnedResultColumn3 = Matrix4.getColumn(matrix, 3, resultColumn3)

    assert(resultColumn0 is returnedResultColumn0)
    assert(resultColumn0 == expectedColumn0)
    assert(resultColumn1 is returnedResultColumn1)
    assert(resultColumn1 == expectedColumn1)
    assert(resultColumn2 is returnedResultColumn2)
    assert(resultColumn2 == expectedColumn2)
    assert(resultColumn3 is returnedResultColumn3)
    assert(resultColumn3 == expectedColumn3)

    matrix = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )

    result = Matrix4()

    expected = Matrix4(
        17.0,
        2.0,
        3.0,
        4.0,
        18.0,
        6.0,
        7.0,
        8.0,
        19.0,
        10.0,
        11.0,
        12.0,
        20.0,
        14.0,
        15.0,
        16.0
    )
    returnedResult = Matrix4.setColumn(
        matrix,
        0,
        Cartesian4(17.0, 18.0, 19.0, 20.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    expected = Matrix4(
        1.0,
        17.0,
        3.0,
        4.0,
        5.0,
        18.0,
        7.0,
        8.0,
        9.0,
        19.0,
        11.0,
        12.0,
        13.0,
        20.0,
        15.0,
        16.0
    )
    returnedResult = Matrix4.setColumn(
        matrix,
        1,
        Cartesian4(17.0, 18.0, 19.0, 20.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    expected = Matrix4(
        1.0,
        2.0,
        17.0,
        4.0,
        5.0,
        6.0,
        18.0,
        8.0,
        9.0,
        10.0,
        19.0,
        12.0,
        13.0,
        14.0,
        20.0,
        16.0
    )
    returnedResult = Matrix4.setColumn(
        matrix,
        2,
        Cartesian4(17.0, 18.0, 19.0, 20.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        17.0,
        5.0,
        6.0,
        7.0,
        18.0,
        9.0,
        10.0,
        11.0,
        19.0,
        13.0,
        14.0,
        15.0,
        20.0
    )
    returnedResult = Matrix4.setColumn(
        matrix,
        3,
        Cartesian4(17.0, 18.0, 19.0, 20.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    matrix = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    translation = Cartesian3(-1.0, -2.0, -3.0)
    result = Matrix4()

    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        -1.0,
        5.0,
        6.0,
        7.0,
        -2.0,
        9.0,
        10.0,
        11.0,
        -3.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    returnedResult = Matrix4.setTranslation(matrix, translation, result)
    assert(result is returnedResult)
    assert(result == expected)

    matrix = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    expectedRow0 = Cartesian4(1.0, 2.0, 3.0, 4.0)
    expectedRow1 = Cartesian4(5.0, 6.0, 7.0, 8.0)
    expectedRow2 = Cartesian4(9.0, 10.0, 11.0, 12.0)
    expectedRow3 = Cartesian4(13.0, 14.0, 15.0, 16.0)

    resultRow0 = Cartesian4()
    resultRow1 = Cartesian4()
    resultRow2 = Cartesian4()
    resultRow3 = Cartesian4()
    returnedResultRow0 = Matrix4.getRow(matrix, 0, resultRow0)
    returnedResultRow1 = Matrix4.getRow(matrix, 1, resultRow1)
    returnedResultRow2 = Matrix4.getRow(matrix, 2, resultRow2)
    returnedResultRow3 = Matrix4.getRow(matrix, 3, resultRow3)

    assert(resultRow0 is returnedResultRow0)
    assert(resultRow0 == expectedRow0)
    assert(resultRow1 is returnedResultRow1)
    assert(resultRow1 == expectedRow1)
    assert(resultRow2 is returnedResultRow2)
    assert(resultRow2 == expectedRow2)
    assert(resultRow3 is returnedResultRow3)
    assert(resultRow3 == expectedRow3)

    matrix = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    result = Matrix4()

    expected = Matrix4(
        91.0,
        92.0,
        93.0,
        94.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    returnedResult = Matrix4.setRow(
        matrix,
        0,
        Cartesian4(91.0, 92.0, 93.0, 94.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        95.0,
        96.0,
        97.0,
        98.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    returnedResult = Matrix4.setRow(
        matrix,
        1,
        Cartesian4(95.0, 96.0, 97.0, 98.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        99.0,
        910.0,
        911.0,
        912.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    returnedResult = Matrix4.setRow(
        matrix,
        2,
        Cartesian4(99.0, 910.0, 911.0, 912.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        913.0,
        914.0,
        915.0,
        916.0
    )
    returnedResult = Matrix4.setRow(
        matrix,
        3,
        Cartesian4(913.0, 914.0, 915.0, 916.0),
        result
    )
    assert(result is returnedResult)
    assert(result == expected)

    oldScale = Cartesian3(2.0, 3.0, 4.0)
    newScale = Cartesian3(5.0, 6.0, 7.0)

    matrix = Matrix4.fromScale(oldScale, Matrix4())
    result = Matrix4()

    assert(Matrix4.getScale(matrix, Cartesian3()) == oldScale)

    returnedResult = Matrix4.setScale(matrix, newScale, result)

    assert(Matrix4.getScale(returnedResult, Cartesian3()) ==
           newScale
           )
    assert(result is returnedResult)

    oldScale = Cartesian3(2.0, 3.0, 4.0)
    newScale = 5.0

    matrix = Matrix4.fromScale(oldScale, Matrix4())
    result = Matrix4()

    assert(Matrix4.getScale(matrix, Cartesian3()) == oldScale)

    returnedResult = Matrix4.setUniformScale(matrix, newScale, result)

    assert(Matrix4.getScale(returnedResult, Cartesian3()) ==
           Cartesian3(newScale, newScale, newScale)
           )
    assert(result is returnedResult)

    scale = Cartesian3(2.0, 3.0, 4.0)
    result = Cartesian3()
    computedScale = Matrix4.getScale(Matrix4.fromScale(scale), result)

    assert(computedScale is result)
    assert(Cartesian3.equalsEpsilon(computedScale, scale, EPSILON14))

    m = Matrix4.fromScale(Cartesian3(2.0, 3.0, 4.0))
    assert(np.isclose(Matrix4.getMaximumScale(m),
                      4.0,
                      EPSILON14
                      ))

    scaleVec = Cartesian3(2.0, 3.0, 4.0)
    scale = Matrix4.fromScale(scaleVec, Matrix4())
    rotation = Matrix3.fromRotationX(0.5, Matrix3())
    scaleRotation = Matrix4.setRotation(scale, rotation, Matrix4())

    extractedScale = Matrix4.getScale(scaleRotation, Cartesian3())
    extractedRotation = Matrix4.getRotation(scaleRotation, Matrix3())

    assert(Cartesian3.equalsEpsilon(extractedScale, scaleVec, EPSILON14))
    assert(Matrix3.equalsEpsilon(extractedRotation, rotation, EPSILON14))

    matrix = Matrix4.fromRotation(
        Matrix3.fromColumnMajorArray([
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
        ])
    )
    expectedRotation = Matrix3.fromColumnMajorArray([
        1.0 / math.sqrt(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0),
        2.0 / math.sqrt(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0),
        3.0 / math.sqrt(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0),
        4.0 / math.sqrt(4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0),
        5.0 / math.sqrt(4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0),
        6.0 / math.sqrt(4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0),
        7.0 / math.sqrt(7.0 * 7.0 + 8.0 * 8.0 + 9.0 * 9.0),
        8.0 / math.sqrt(7.0 * 7.0 + 8.0 * 8.0 + 9.0 * 9.0),
        9.0 / math.sqrt(7.0 * 7.0 + 8.0 * 8.0 + 9.0 * 9.0),
    ])
    rotation = Matrix4.getRotation(matrix, Matrix3())
    assert(Matrix3.equalsEpsilon(rotation, expectedRotation, EPSILON14))

    matrix = Matrix4.fromRotation(
        Matrix3.fromColumnMajorArray([
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
        ])
    )
    duplicateMatrix = Matrix4.clone(matrix, Matrix4())
    expectedRotation = Matrix3.fromColumnMajorArray([
        1.0 / math.sqrt(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0),
        2.0 / math.sqrt(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0),
        3.0 / math.sqrt(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0),
        4.0 / math.sqrt(4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0),
        5.0 / math.sqrt(4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0),
        6.0 / math.sqrt(4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0),
        7.0 / math.sqrt(7.0 * 7.0 + 8.0 * 8.0 + 9.0 * 9.0),
        8.0 / math.sqrt(7.0 * 7.0 + 8.0 * 8.0 + 9.0 * 9.0),
        9.0 / math.sqrt(7.0 * 7.0 + 8.0 * 8.0 + 9.0 * 9.0),
    ])
    result = Matrix4.getRotation(matrix, Matrix3())
    assert(Matrix3.equalsEpsilon(result, expectedRotation, EPSILON14))
    assert(matrix == duplicateMatrix)
    assert(matrix is not result)

    left = Matrix4(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16
    )
    right = Matrix4(
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32
    )
    expected = Matrix4(
        250,
        260,
        270,
        280,
        618,
        644,
        670,
        696,
        986,
        1028,
        1070,
        1112,
        1354,
        1412,
        1470,
        1528
    )
    result = Matrix4()
    returnedResult = Matrix4.multiply(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    left = Matrix4(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16
    )
    right = Matrix4(
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32
    )
    expected = Matrix4(
        250,
        260,
        270,
        280,
        618,
        644,
        670,
        696,
        986,
        1028,
        1070,
        1112,
        1354,
        1412,
        1470,
        1528
    )
    returnedResult = Matrix4.multiply(left, right, left)
    assert(returnedResult is left)
    assert(left == expected)

    left = Matrix4(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16
    )
    right = Matrix4(
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32
    )
    expected = Matrix4(
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        40,
        42,
        44,
        46,
        48
    )
    result = Matrix4()
    returnedResult = Matrix4.add(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    left = Matrix4(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16
    )
    right = Matrix4(
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32
    )
    expected = Matrix4(
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        40,
        42,
        44,
        46,
        48
    )
    returnedResult = Matrix4.add(left, right, left)
    assert(returnedResult is left)
    assert(left == expected)

    left = Matrix4(
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        40,
        42,
        44,
        46,
        48
    )
    right = Matrix4(
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32
    )
    expected = Matrix4(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16
    )
    result = Matrix4()
    returnedResult = Matrix4.subtract(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    left = Matrix4(
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        40,
        42,
        44,
        46,
        48
    )
    right = Matrix4(
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32
    )
    expected = Matrix4(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16
    )
    returnedResult = Matrix4.subtract(left, right, left)
    assert(returnedResult is left)
    assert(left == expected)

    left = Matrix4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 1)
    right = Matrix4(
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        0,
        0,
        0,
        1
    )
    expected = Matrix4(
        134,
        140,
        146,
        156,
        386,
        404,
        422,
        448,
        638,
        668,
        698,
        740,
        0,
        0,
        0,
        1
    )
    result = Matrix4()
    returnedResult = Matrix4.multiplyTransformation(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    left = Matrix4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 1)
    right = Matrix4(
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        0,
        0,
        0,
        1
    )
    expected = Matrix4(
        134,
        140,
        146,
        156,
        386,
        404,
        422,
        448,
        638,
        668,
        698,
        740,
        0,
        0,
        0,
        1
    )
    returnedResult = Matrix4.multiplyTransformation(left, right, left)
    assert(returnedResult is left)
    assert(left == expected)

    left = Matrix4.fromRotationTranslation(
        Matrix3.fromRotationZ(np.radians(45.0)),
        Cartesian3(1.0, 2.0, 3.0)
    )
    rightRotation = Matrix3.fromRotationX(np.radians(30.0))
    right = Matrix4.fromRotationTranslation(rightRotation)
    expected = Matrix4.multiplyTransformation(
        left,
        right,
        Matrix4()
    )
    result = Matrix4()
    returnedResult = Matrix4.multiplyByMatrix3(
        left,
        rightRotation,
        result
    )
    assert(returnedResult is result)
    assert(result == expected)

    left = Matrix4.fromRotationTranslation(
        Matrix3.fromRotationZ(np.radians(45.0)),
        Cartesian3(1.0, 2.0, 3.0)
    )
    rightRotation = Matrix3.fromRotationX(np.radians(30.0))
    right = Matrix4.fromRotationTranslation(rightRotation)
    expected = Matrix4.multiplyTransformation(
        left,
        right,
        Matrix4()
    )
    returnedResult = Matrix4.multiplyByMatrix3(left, rightRotation, left)
    assert(returnedResult is left)
    assert(left == expected)

    m = Matrix4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 1)
    translation = Cartesian3(17, 18, 19)
    expected = Matrix4.multiply(
        m,
        Matrix4.fromTranslation(translation),
        Matrix4()
    )
    result = Matrix4()
    returnedResult = Matrix4.multiplyByTranslation(
        m,
        translation,
        result
    )
    assert(returnedResult is result)
    assert(result == expected)

    m = Matrix4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 1)
    translation = Cartesian3(17, 18, 19)
    expected = Matrix4.multiply(
        m,
        Matrix4.fromTranslation(translation),
        Matrix4()
    )
    returnedResult = Matrix4.multiplyByTranslation(m, translation, m)
    assert(returnedResult is m)
    assert(m == expected)

    m = Matrix4.fromColumnMajorArray([
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
    ])
    scale = 2.0
    expected = Matrix4.fromColumnMajorArray([
        2 * scale,
        3 * scale,
        4 * scale,
        5,
        6 * scale,
        7 * scale,
        8 * scale,
        9,
        10 * scale,
        11 * scale,
        12 * scale,
        13,
        14,
        15,
        16,
        17,
    ])
    result = Matrix4()
    returnedResult = Matrix4.multiplyByUniformScale(m, scale, result)
    assert(returnedResult is result)
    assert(result == expected)

    m = Matrix4.fromColumnMajorArray([
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
    ])
    scale = 2.0
    expected = Matrix4.fromColumnMajorArray([
        2 * scale,
        3 * scale,
        4 * scale,
        5,
        6 * scale,
        7 * scale,
        8 * scale,
        9,
        10 * scale,
        11 * scale,
        12 * scale,
        13,
        14,
        15,
        16,
        17,
    ])
    returnedResult = Matrix4.multiplyByUniformScale(m, scale, m)
    assert(returnedResult is m)
    assert(m == expected)

    m = Matrix4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 1)
    scale = Cartesian3(1.0, 1.0, 1.0)
    expected = Matrix4.multiply(m, Matrix4.fromScale(scale), Matrix4())
    result = Matrix4()
    returnedResult = Matrix4.multiplyByScale(m, scale, result)
    assert(returnedResult is result)
    assert(result == expected)

    m = Matrix4(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 0, 0, 1)
    scale = Cartesian3(2.0, 3.0, 4.0)
    expected = Matrix4.multiply(m, Matrix4.fromScale(scale), Matrix4())
    result = Matrix4()
    returnedResult = Matrix4.multiplyByScale(m, scale, result)
    assert(returnedResult is result)
    assert(result == expected)

    m = Matrix4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 1)
    scale = Cartesian3(1.0, 2.0, 3.0)
    expected = Matrix4.multiply(
        m,
        Matrix4.fromScale(scale),
        Matrix4()
    )
    returnedResult = Matrix4.multiplyByScale(m, scale, m)
    assert(returnedResult is m)
    assert(m == expected)

    m = Matrix4.fromColumnMajorArray([
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
    ])
    scale = 2.0
    expected = Matrix4.fromColumnMajorArray([
        2 * scale,
        3 * scale,
        4 * scale,
        5,
        6 * scale,
        7 * scale,
        8 * scale,
        9,
        10 * scale,
        11 * scale,
        12 * scale,
        13,
        14,
        15,
        16,
        17,
    ])

    result = Matrix4()
    returnedResult = Matrix4.multiplyByUniformScale(m, scale, result)
    assert(returnedResult is result)
    assert(result == expected)

    m = Matrix4.fromColumnMajorArray([
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
    ])
    scale = 2.0
    expected = Matrix4.fromColumnMajorArray([
        2 * scale,
        3 * scale,
        4 * scale,
        5,
        6 * scale,
        7 * scale,
        8 * scale,
        9,
        10 * scale,
        11 * scale,
        12 * scale,
        13,
        14,
        15,
        16,
        17,
    ])

    returnedResult = Matrix4.multiplyByUniformScale(m, scale, m)
    assert(returnedResult is m)
    assert(m == expected)

    left = Matrix4(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16
    )
    right = Cartesian4(17, 18, 19, 20)
    expected = Cartesian4(190, 486, 782, 1078)
    result = Cartesian4()
    returnedResult = Matrix4.multiplyByVector(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    left = Matrix4(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16
    )
    right = Cartesian3(17, 18, 19)
    expected = Cartesian3(114, 334, 554)
    result = Cartesian3()
    returnedResult = Matrix4.multiplyByPoint(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    left = Matrix4(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16
    )
    right = Cartesian3(17, 18, 19)
    expected = Cartesian3(110, 326, 542)
    result = Cartesian3()
    returnedResult = Matrix4.multiplyByPointAsVector(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    left = Matrix4(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16
    )
    right = 2
    expected = Matrix4(
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32
    )
    result = Matrix4()
    returnedResult = Matrix4.multiplyByScalar(left, right, result)
    assert(returnedResult is result)
    assert(result == expected)

    matrix = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    expected = Matrix4(
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
        -7.0,
        -8.0,
        -9.0,
        -10.0,
        -11.0,
        -12.0,
        -13.0,
        -14.0,
        -15.0,
        -16.0
    )
    result = Matrix4()
    returnedResult = Matrix4.negate(matrix, result)
    assert(result is returnedResult)
    assert(result == expected)

    matrix = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    expected = Matrix4(
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
        -7.0,
        -8.0,
        -9.0,
        -10.0,
        -11.0,
        -12.0,
        -13.0,
        -14.0,
        -15.0,
        -16.0
    )
    returnedResult = Matrix4.negate(matrix, matrix)
    assert(matrix is returnedResult)
    assert(matrix == expected)

    matrix = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    expected = Matrix4(
        1.0,
        5.0,
        9.0,
        13.0,
        2.0,
        6.0,
        10.0,
        14.0,
        3.0,
        7.0,
        11.0,
        15.0,
        4.0,
        8.0,
        12.0,
        16.0
    )
    result = Matrix4()
    returnedResult = Matrix4.transpose(matrix, result)
    assert(result is returnedResult)
    assert(result == expected)

    matrix = Matrix4(
        1.0,
        2.0,
        6.0,
        4.0,
        8.0,
        6.0,
        -7.0,
        8.0,
        9.0,
        -20.0,
        -11.0,
        12.0,
        13.0,
        -27.0,
        15.0,
        16.0
    )
    expectedInverse = Matrix4.inverse(matrix, Matrix4())
    expectedInverseTranspose = Matrix4.transpose(
        expectedInverse,
        Matrix4()
    )
    result = Matrix4.inverseTranspose(matrix, Matrix4())
    assert(result == expectedInverseTranspose)

    matrix = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    expected = Matrix4(
        1.0,
        5.0,
        9.0,
        13.0,
        2.0,
        6.0,
        10.0,
        14.0,
        3.0,
        7.0,
        11.0,
        15.0,
        4.0,
        8.0,
        12.0,
        16.0
    )
    returnedResult = Matrix4.transpose(matrix, matrix)
    assert(matrix is returnedResult)
    assert(matrix == expected)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    assert(left == right)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    assert(left != right)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(1.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    assert(left != right)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(1.0, 2.0, 7.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    assert(left != right)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(1.0, 2.0, 3.0, 8.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    assert(left != right)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(1.0, 2.0, 3.0, 4.0, 9.0, 6.0, 7.0, 8.0, 9.0)
    assert(left != right)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 7.0, 8.0, 9.0)
    assert(left != right)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 8.0, 9.0)
    assert(left != right)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 12.0, 9.0)
    assert(left != right)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 13.0)
    assert(left != right)

    assert(Matrix4() is not None)
    assert(None is not Matrix4())

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 1.0) is True)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        5.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix4.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        6.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix4.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        2.0,
        7.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix4.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        2.0,
        3.0,
        8.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix4.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        9.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix4.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        10.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix4.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        11.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix4.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        12.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix4.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        13.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix4.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        14.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix4.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        15.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix4.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        16.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix4.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        17.0,
        14.0,
        15.0,
        16.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix4.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        18.0,
        15.0,
        16.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix4.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        19.0,
        16.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix4.equalsEpsilon(left, right, 4.0) is True)

    left = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    right = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        20.0
    )
    assert(Matrix4.equalsEpsilon(left, right, 3.9) is False)
    assert(Matrix4.equalsEpsilon(left, right, 4.0) is True)

    assert(Matrix4.equalsEpsilon(None, None, 1.0) is True)
    assert(Matrix4.equalsEpsilon(Matrix4(), None, 1.0) is False)
    assert(Matrix4.equalsEpsilon(None, Matrix4(), 1.0) is False)

    matrix = Matrix4(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16
    )
    assert(str(matrix) ==
           "[1, 2, 3, 4]\n[5, 6, 7, 8]\n[9, 10, 11, 12]\n[13, 14, 15, 16]"
           )

    matrix = Matrix4(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16
    )
    expected = Cartesian3(4, 8, 12)
    result = Cartesian3()
    returnedResult = Matrix4.getTranslation(matrix, result)
    assert(returnedResult is result)
    assert(expected == returnedResult)

    matrix = Matrix4(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16
    )
    expected = Matrix3(1, 2, 3, 5, 6, 7, 9, 10, 11)
    result = Matrix3()
    returnedResult = Matrix4.getMatrix3(matrix, result)
    assert(returnedResult is result)
    assert(expected == returnedResult)

    matrix = Matrix4(
        0.72,
        0.7,
        0.0,
        0.0,
        -0.4,
        0.41,
        0.82,
        0.0,
        0.57,
        -0.59,
        0.57,
        -3.86,
        0.0,
        0.0,
        0.0,
        1.0
    )

    expected = Matrix4(
        0.7150830193944467,
        -0.3976559229803265,
        0.5720664155155574,
        2.2081763638900513,
        0.6930574657657118,
        0.40901752077976433,
        -0.5884111702445733,
        -2.271267117144053,
        0.0022922521876059163,
        0.8210249357172755,
        0.5732623731786561,
        2.2127927604696125,
        0.0,
        0.0,
        0.0,
        1.0
    )

    result = Matrix4()
    returnedResult = Matrix4.inverse(matrix, result)
    assert(returnedResult is result)
    assert(Matrix4.equalsEpsilon(expected, returnedResult, EPSILON20))
    assert(Matrix4.equalsEpsilon(
        Matrix4.multiply(returnedResult, matrix, Matrix4()),
        Matrix4.IDENTITY(), EPSILON15))

    matrix = Matrix4.fromTranslation(Cartesian3(1.0, 2.0, 3.0))
    matrix = Matrix4.multiplyByUniformScale(matrix, 0.0, matrix)
    expected = Matrix4.fromTranslation(Cartesian3(-1.0, -2.0, -3.0))
    expected = Matrix4.multiplyByUniformScale(expected, 0.0, expected)

    result = Matrix4.inverse(matrix, Matrix4())
    assert(Matrix4.equalsEpsilon(expected, result, EPSILON20))

    trs = {
        'translation': Cartesian3(0.0, 0.0, 0.0),
        'rotation': Quaternion.fromAxisAngle(Cartesian3.UNIT_X(), 0.0),
        'scale': Cartesian3(1.0e-7, 1.0e-7, 1.1e-7)
    }

    matrix = Matrix4.fromTranslationRotationScale(trs)

    expected = Matrix4(
        1e7,
        0,
        0,
        0,
        0,
        1e7,
        0,
        0,
        0,
        0,
        (1.0 / 1.1) * 1e7,
        0,
        0,
        0,
        0,
        1
    )

    result = Matrix4.inverse(matrix, Matrix4())
    assert(Matrix4.equalsEpsilon(expected, result, EPSILON15))

    trs = {
        'translation': Cartesian3(0.0, 0.0, 0.0),
        'rotation': Quaternion.fromAxisAngle(Cartesian3.UNIT_X(), 0.0),
        'scale': Cartesian3(1.8e-8, 1.2e-8, 1.2e-8)
    }

    matrix = Matrix4.fromTranslationRotationScale(trs)

    expected = Matrix4(
        0,
        0,
        0,
        -matrix[12],
        0,
        0,
        0,
        -matrix[13],
        0,
        0,
        0,
        -matrix[14],
        0,
        0,
        0,
        1
    )

    result = Matrix4.inverse(matrix, Matrix4())
    assert(Matrix4.equalsEpsilon(expected, result, EPSILON20))

    matrix = Matrix4(
        1,
        0,
        0,
        10,
        0,
        0,
        1,
        20,
        0,
        1,
        0,
        30,
        0,
        0,
        0,
        1
    )

    expected = Matrix4(
        1,
        0,
        0,
        -10,
        0,
        0,
        1,
        -30,
        0,
        1,
        0,
        -20,
        0,
        0,
        0,
        1
    )

    result = Matrix4()
    returnedResult = Matrix4.inverseTransformation(matrix, result)
    assert(returnedResult is result)
    assert(expected == returnedResult)
    assert(Matrix4.multiply(returnedResult, matrix, Matrix4()) ==
           Matrix4.IDENTITY()
           )

    with pytest.raises(Exception) as exc_info:
        matrix = Matrix4(
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
        )
        m = Matrix4.inverse(matrix, Matrix4())

    assert(exc_info.value.args[0] == 'matrix is not invertible because its determinate is zero')

    matrix = Matrix4(
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
        -7.0,
        -8.0,
        -9.0,
        -10.0,
        -11.0,
        -12.0,
        -13.0,
        -14.0,
        -15.0,
        -16.0
    )
    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    result = Matrix4()
    returnedResult = Matrix4.abs(matrix, result)
    assert(returnedResult == expected)

    matrix = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    returnedResult = Matrix4.abs(matrix, result)
    assert(returnedResult == expected)

    matrix = Matrix4(
        1.0,
        -2.0,
        -3.0,
        4.0,
        5.0,
        -6.0,
        7.0,
        -8.0,
        9.0,
        -10.0,
        11.0,
        -12.0,
        13.0,
        -14.0,
        15.0,
        -16.0
    )
    returnedResult = Matrix4.abs(matrix, result)
    assert(returnedResult == expected)

    matrix = Matrix4(
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
        -7.0,
        -8.0,
        -9.0,
        -10.0,
        -11.0,
        -12.0,
        -13.0,
        -14.0,
        -15.0,
        -16.0
    )
    expected = Matrix4(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0
    )
    returnedResult = Matrix4.abs(matrix, matrix)
    assert(matrix is returnedResult)
    assert(matrix == expected)

    assert(Matrix4.clone(None) is None)

    matrix = Matrix4(
        1,
        5,
        9,
        13,
        2,
        6,
        10,
        14,
        3,
        7,
        11,
        15,
        4,
        8,
        12,
        16
    )
    assert(len(matrix) == 16)
    for index in range(0, len(matrix)):
        assert(matrix[index] == index + 1)

    matrix = Matrix4.IDENTITY()
    assert(matrix[Matrix4.COLUMN0ROW0] == 1.0)
    assert(matrix[Matrix4.COLUMN1ROW0] == 0.0)
    assert(matrix[Matrix4.COLUMN2ROW0] == 0.0)
    assert(matrix[Matrix4.COLUMN3ROW0] == 0.0)
    assert(matrix[Matrix4.COLUMN0ROW1] == 0.0)
    assert(matrix[Matrix4.COLUMN1ROW1] == 1.0)
    assert(matrix[Matrix4.COLUMN2ROW1] == 0.0)
    assert(matrix[Matrix4.COLUMN3ROW1] == 0.0)
    assert(matrix[Matrix4.COLUMN0ROW2] == 0.0)
    assert(matrix[Matrix4.COLUMN1ROW2] == 0.0)
    assert(matrix[Matrix4.COLUMN2ROW2] == 1.0)
    assert(matrix[Matrix4.COLUMN3ROW2] == 0.0)
    assert(matrix[Matrix4.COLUMN0ROW3] == 0.0)
    assert(matrix[Matrix4.COLUMN1ROW3] == 0.0)
    assert(matrix[Matrix4.COLUMN2ROW3] == 0.0)
    assert(matrix[Matrix4.COLUMN3ROW3] == 1.0)

    matrix = Matrix4.ZERO()
    assert(matrix[Matrix4.COLUMN0ROW0] == 0.0)
    assert(matrix[Matrix4.COLUMN1ROW0] == 0.0)
    assert(matrix[Matrix4.COLUMN2ROW0] == 0.0)
    assert(matrix[Matrix4.COLUMN3ROW0] == 0.0)
    assert(matrix[Matrix4.COLUMN0ROW1] == 0.0)
    assert(matrix[Matrix4.COLUMN1ROW1] == 0.0)
    assert(matrix[Matrix4.COLUMN2ROW1] == 0.0)
    assert(matrix[Matrix4.COLUMN3ROW1] == 0.0)
    assert(matrix[Matrix4.COLUMN0ROW2] == 0.0)
    assert(matrix[Matrix4.COLUMN1ROW2] == 0.0)
    assert(matrix[Matrix4.COLUMN2ROW2] == 0.0)
    assert(matrix[Matrix4.COLUMN3ROW2] == 0.0)
    assert(matrix[Matrix4.COLUMN0ROW3] == 0.0)
    assert(matrix[Matrix4.COLUMN1ROW3] == 0.0)
    assert(matrix[Matrix4.COLUMN2ROW3] == 0.0)
    assert(matrix[Matrix4.COLUMN3ROW3] == 0.0)
