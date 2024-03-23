import math
import numpy as np
from satsim.vecmath import Cartesian2, Cartesian3, Cartesian4
from satsim.math.const import PI_OVER_TWO, PI_OVER_FOUR, PI, EPSILON6, EPSILON7, EPSILON9, EPSILON14, EPSILON15


def test_cartesian2():

    cartesian = Cartesian2()
    assert(cartesian.x == 0.0)
    assert(cartesian.y == 0.0)

    cartesian = Cartesian2(1.0)
    assert(cartesian.x == 1.0)
    assert(cartesian.y == 0.0)

    cartesian = Cartesian2(1.0, 2.0)
    assert(cartesian.x == 1.0)
    assert(cartesian.y == 2.0)

    cartesian = Cartesian2()
    cartesian = Cartesian2.fromArray([1.0, 2.0])
    assert(cartesian == Cartesian2(1.0, 2.0))

    cartesian = Cartesian2()
    cartesian = Cartesian2.fromArray([0.0, 1.0, 2.0, 0.0], 1)
    assert(cartesian == Cartesian2(1.0, 2.0))

    cartesian = Cartesian2(1.0, 2.0)
    result = Cartesian2()
    returnedResult = Cartesian2.clone(cartesian, result)
    assert(cartesian is not result)
    assert(result is returnedResult)
    assert(cartesian == result)

    cartesian = Cartesian2(1.0, 2.0)
    returnedResult = Cartesian2.clone(cartesian, cartesian)
    assert(cartesian is returnedResult)

    cartesian = Cartesian2(2.0, 1.0)
    assert(Cartesian2.maximumComponent(cartesian) == cartesian.x)

    cartesian = Cartesian2(1.0, 2.0)
    assert(Cartesian2.maximumComponent(cartesian) == cartesian.y)

    cartesian = Cartesian2(1.0, 2.0)
    assert(Cartesian2.minimumComponent(cartesian) == cartesian.x)

    cartesian = Cartesian2(2.0, 1.0)
    assert(Cartesian2.minimumComponent(cartesian) == cartesian.y)

    result = Cartesian2()

    first = Cartesian2(2.0, 0.0)
    second = Cartesian2(1.0, 0.0)
    expected = Cartesian2(1.0, 0.0)
    assert(Cartesian2.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian2(1.0, 0.0)
    second = Cartesian2(2.0, 0.0)
    expected = Cartesian2(1.0, 0.0)
    assert(Cartesian2.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian2(2.0, -15.0)
    second = Cartesian2(1.0, -20.0)
    expected = Cartesian2(1.0, -20.0)
    assert(Cartesian2.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian2(2.0, -20.0)
    second = Cartesian2(1.0, -15.0)
    expected = Cartesian2(1.0, -20.0)
    assert(Cartesian2.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian2(2.0, -15.0)
    second = Cartesian2(1.0, -20.0)
    expected = Cartesian2(1.0, -20.0)
    assert(Cartesian2.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian2(2.0, -15.0)
    second = Cartesian2(1.0, -20.0)
    expected = Cartesian2(1.0, -20.0)
    assert(Cartesian2.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian2(2.0, 0.0)
    second = Cartesian2(1.0, 0.0)
    expected = Cartesian2(1.0, 0.0)
    result = Cartesian2()
    returnedResult = Cartesian2.minimumByComponent(first, second, result)
    assert(returnedResult is result)
    assert(returnedResult == expected)

    first = Cartesian2(2.0, 0.0)
    second = Cartesian2(1.0, 0.0)
    expected = Cartesian2(1.0, 0.0)
    assert(Cartesian2.minimumByComponent(first, second, first) ==
           expected
           )

    first.x = 1.0
    second.x = 2.0
    assert(Cartesian2.minimumByComponent(first, second, second) ==
           expected
           )

    first = Cartesian2(2.0, 0.0)
    second = Cartesian2(1.0, 0.0)
    expected = Cartesian2(1.0, 0.0)
    assert(Cartesian2.minimumByComponent(first, second, expected) ==
           expected
           )

    second.x = 3.0
    expected.x = 2.0
    assert(Cartesian2.minimumByComponent(first, second, expected) ==
           expected
           )

    first = Cartesian2(0.0, 2.0)
    second = Cartesian2(0.0, 1.0)
    expected = Cartesian2(0.0, 1.0)
    result = Cartesian2()
    assert(Cartesian2.minimumByComponent(first, second, result) ==
           expected
           )

    second.y = 3.0
    expected.y = 2.0
    assert(Cartesian2.minimumByComponent(first, second, result) ==
           expected
           )

    result = Cartesian2()

    first = Cartesian2(2.0, 0.0)
    second = Cartesian2(1.0, 0.0)
    expected = Cartesian2(2.0, 0.0)
    assert(Cartesian2.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian2(1.0, 0.0)
    second = Cartesian2(2.0, 0.0)
    expected = Cartesian2(2.0, 0.0)
    assert(Cartesian2.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian2(2.0, -15.0)
    second = Cartesian2(1.0, -20.0)
    expected = Cartesian2(2.0, -15.0)
    assert(Cartesian2.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian2(2.0, -20.0)
    second = Cartesian2(1.0, -15.0)
    expected = Cartesian2(2.0, -15.0)
    assert(Cartesian2.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian2(2.0, -15.0)
    second = Cartesian2(1.0, -20.0)
    expected = Cartesian2(2.0, -15.0)
    assert(Cartesian2.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian2(2.0, -15.0)
    second = Cartesian2(1.0, -20.0)
    expected = Cartesian2(2.0, -15.0)
    assert(Cartesian2.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian2(2.0, 0.0)
    second = Cartesian2(1.0, 0.0)
    expected = Cartesian2(2.0, 0.0)
    result = Cartesian2()
    returnedResult = Cartesian2.maximumByComponent(first, second, result)
    assert(returnedResult is result)
    assert(returnedResult == expected)

    first = Cartesian2(2.0, 0.0)
    second = Cartesian2(1.0, 0.0)
    expected = Cartesian2(2.0, 0.0)
    assert(Cartesian2.maximumByComponent(first, second, first) ==
           expected
           )

    first.x = 1.0
    second.x = 2.0
    assert(Cartesian2.maximumByComponent(first, second, second) ==
           expected
           )

    first = Cartesian2(2.0, 0.0)
    second = Cartesian2(1.0, 0.0)
    expected = Cartesian2(2.0, 0.0)
    result = Cartesian2()
    assert(Cartesian2.maximumByComponent(first, second, result) ==
           expected
           )

    second.x = 3.0
    expected.x = 3.0
    assert(Cartesian2.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian2(0.0, 2.0)
    second = Cartesian2(0.0, 1.0)
    expected = Cartesian2(0.0, 2.0)
    result = Cartesian2()
    assert(Cartesian2.maximumByComponent(first, second, result) ==
           expected
           )

    second.y = 3.0
    expected.y = 3.0
    assert(Cartesian2.maximumByComponent(first, second, result) ==
           expected
           )

    result = Cartesian2()

    value = Cartesian2(-1.0, 0.0)
    min = Cartesian2(0.0, 0.0)
    max = Cartesian2(1.0, 1.0)
    expected = Cartesian2(0.0, 0.0)
    assert(Cartesian2.clamp(value, min, max, result) == expected)

    value = Cartesian2(2.0, 0.0)
    min = Cartesian2(0.0, 0.0)
    max = Cartesian2(1.0, 1.0)
    expected = Cartesian2(1.0, 0.0)
    assert(Cartesian2.clamp(value, min, max, result) == expected)

    value = Cartesian2(0.0, -1.0)
    min = Cartesian2(0.0, 0.0)
    max = Cartesian2(1.0, 1.0)
    expected = Cartesian2(0.0, 0.0)
    assert(Cartesian2.clamp(value, min, max, result) == expected)

    value = Cartesian2(0.0, 2.0)
    min = Cartesian2(0.0, 0.0)
    max = Cartesian2(1.0, 1.0)
    expected = Cartesian2(0.0, 1.0)
    assert(Cartesian2.clamp(value, min, max, result) == expected)

    value = Cartesian2(0.0, 0.0)
    min = Cartesian2(0.0, 0.0)
    max = Cartesian2(1.0, 1.0)
    expected = Cartesian2(0.0, 0.0)
    assert(Cartesian2.clamp(value, min, max, result) == expected)

    value = Cartesian2(0.0, 0.0)
    min = Cartesian2(0.0, 0.0)
    max = Cartesian2(1.0, 1.0)
    expected = Cartesian2(0.0, 0.0)
    assert(Cartesian2.clamp(value, min, max, result) == expected)

    value = Cartesian2(-2.0, 3.0)
    min = Cartesian2(0.0, 0.0)
    max = Cartesian2(1.0, 1.0)
    expected = Cartesian2(0.0, 1.0)
    assert(Cartesian2.clamp(value, min, max, result) == expected)

    value = Cartesian2(0.0, 0.0)
    min = Cartesian2(1.0, 2.0)
    max = Cartesian2(1.0, 2.0)
    expected = Cartesian2(1.0, 2.0)
    assert(Cartesian2.clamp(value, min, max, result) == expected)

    value = Cartesian2(-1.0, -1.0)
    min = Cartesian2(0.0, 0.0)
    max = Cartesian2(1.0, 1.0)
    expected = Cartesian2(0.0, 0.0)
    result = Cartesian2()
    returnedResult = Cartesian2.clamp(value, min, max, result)
    assert(returnedResult is result)
    assert(returnedResult == expected)

    value = Cartesian2(-1.0, -1.0)
    min = Cartesian2(0.0, 0.0)
    max = Cartesian2(1.0, 1.0)
    expected = Cartesian2(0.0, 0.0)
    assert(Cartesian2.clamp(value, min, max, value) == expected)

    Cartesian2.fromElements(-1.0, -1.0, value)
    assert(Cartesian2.clamp(value, min, max, min) == expected)

    Cartesian2.fromElements(0.0, 0.0, value)
    assert(Cartesian2.clamp(value, min, max, max) == expected)

    cartesian = Cartesian2(2.0, 3.0)
    assert(Cartesian2.magnitudeSquared(cartesian) == 13)
    cartesian = Cartesian2(2.0, 3.0)
    assert(Cartesian2.magnitude(cartesian) == math.sqrt(13.0))

    distance = Cartesian2.distance(
        Cartesian2(1.0, 0.0),
        Cartesian2(2.0, 0.0)
    )
    assert(distance == 1.0)

    distanceSquared = Cartesian2.distanceSquared(
        Cartesian2(1.0, 0.0),
        Cartesian2(3.0, 0.0)
    )
    assert(distanceSquared == 4.0)

    cartesian = Cartesian2(2.0, 0.0)
    expectedResult = Cartesian2(1.0, 0.0)
    result = Cartesian2()
    returnedResult = Cartesian2.normalize(cartesian, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    cartesian = Cartesian2(2.0, 0.0)
    expectedResult = Cartesian2(1.0, 0.0)
    returnedResult = Cartesian2.normalize(cartesian, cartesian)
    assert(cartesian is returnedResult)
    assert(cartesian == expectedResult)

    left = Cartesian2(2.0, 3.0)
    right = Cartesian2(4.0, 5.0)
    result = Cartesian2()
    expectedResult = Cartesian2(8.0, 15.0)
    returnedResult = Cartesian2.multiplyComponents(left, right, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    left = Cartesian2(2.0, 3.0)
    right = Cartesian2(4.0, 5.0)
    expectedResult = Cartesian2(8.0, 15.0)
    returnedResult = Cartesian2.multiplyComponents(left, right, left)
    assert(left is returnedResult)
    assert(left == expectedResult)

    left = Cartesian2(2.0, 3.0)
    right = Cartesian2(4.0, 5.0)
    result = Cartesian2()
    expectedResult = Cartesian2(0.5, 0.6)
    returnedResult = Cartesian2.divideComponents(left, right, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    left = Cartesian2(2.0, 3.0)
    right = Cartesian2(4.0, 5.0)
    expectedResult = Cartesian2(0.5, 0.6)
    returnedResult = Cartesian2.divideComponents(left, right, left)
    assert(left is returnedResult)
    assert(left == expectedResult)

    left = Cartesian2(2.0, 3.0)
    right = Cartesian2(4.0, 5.0)
    expectedResult = 23.0
    result = Cartesian2.dot(left, right)
    assert(result == expectedResult)

    left = Cartesian2(0.0, 1.0)
    right = Cartesian2(1.0, 0.0)
    expectedResult = -1.0
    result = Cartesian2.cross(left, right)
    assert(result == expectedResult)

    left = Cartesian2(2.0, 3.0)
    right = Cartesian2(4.0, 5.0)
    result = Cartesian2()
    expectedResult = Cartesian2(6.0, 8.0)
    returnedResult = Cartesian2.add(left, right, result)
    assert(returnedResult is result)
    assert(result == expectedResult)

    left = Cartesian2(2.0, 3.0)
    right = Cartesian2(4.0, 5.0)
    expectedResult = Cartesian2(6.0, 8.0)
    returnedResult = Cartesian2.add(left, right, left)
    assert(returnedResult is left)
    assert(left == expectedResult)

    left = Cartesian2(2.0, 3.0)
    right = Cartesian2(1.0, 5.0)
    result = Cartesian2()
    expectedResult = Cartesian2(1.0, -2.0)
    returnedResult = Cartesian2.subtract(left, right, result)
    assert(returnedResult is result)
    assert(result == expectedResult)

    left = Cartesian2(2.0, 3.0)
    right = Cartesian2(1.0, 5.0)
    expectedResult = Cartesian2(1.0, -2.0)
    returnedResult = Cartesian2.subtract(left, right, left)
    assert(returnedResult is left)
    assert(left == expectedResult)

    cartesian = Cartesian2(1, 2)
    result = Cartesian2()
    scalar = 2
    expectedResult = Cartesian2(2, 4)
    returnedResult = Cartesian2.multiplyByScalar(
        cartesian,
        scalar,
        result
    )
    assert(result is returnedResult)
    assert(result == expectedResult)

    cartesian = Cartesian2(1, 2)
    scalar = 2
    expectedResult = Cartesian2(2, 4)
    returnedResult = Cartesian2.multiplyByScalar(
        cartesian,
        scalar,
        cartesian
    )
    assert(cartesian is returnedResult)
    assert(cartesian == expectedResult)

    cartesian = Cartesian2(1.0, 2.0)
    result = Cartesian2()
    scalar = 2
    expectedResult = Cartesian2(0.5, 1.0)
    returnedResult = Cartesian2.divideByScalar(cartesian, scalar, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    cartesian = Cartesian2(1.0, 2.0)
    scalar = 2
    expectedResult = Cartesian2(0.5, 1.0)
    returnedResult = Cartesian2.divideByScalar(
        cartesian,
        scalar,
        cartesian
    )
    assert(cartesian is returnedResult)
    assert(cartesian == expectedResult)

    cartesian = Cartesian2(1.0, -2.0)
    result = Cartesian2()
    expectedResult = Cartesian2(-1.0, 2.0)
    returnedResult = Cartesian2.negate(cartesian, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    cartesian = Cartesian2(1.0, -2.0)
    expectedResult = Cartesian2(-1.0, 2.0)
    returnedResult = Cartesian2.negate(cartesian, cartesian)
    assert(cartesian is returnedResult)
    assert(cartesian == expectedResult)

    cartesian = Cartesian2(1.0, -2.0)
    result = Cartesian2()
    expectedResult = Cartesian2(1.0, 2.0)
    returnedResult = Cartesian2.abs(cartesian, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    cartesian = Cartesian2(1.0, -2.0)
    expectedResult = Cartesian2(1.0, 2.0)
    returnedResult = Cartesian2.abs(cartesian, cartesian)
    assert(cartesian is returnedResult)
    assert(cartesian == expectedResult)

    start = Cartesian2(4.0, 8.0)
    end = Cartesian2(8.0, 20.0)
    t = 0.25
    result = Cartesian2()
    expectedResult = Cartesian2(5.0, 11.0)
    returnedResult = Cartesian2.lerp(start, end, t, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    start = Cartesian2(4.0, 8.0)
    end = Cartesian2(8.0, 20.0)
    t = 0.25
    expectedResult = Cartesian2(5.0, 11.0)
    returnedResult = Cartesian2.lerp(start, end, t, start)
    assert(start is returnedResult)
    assert(start == expectedResult)

    start = Cartesian2(4.0, 8.0)
    end = Cartesian2(8.0, 20.0)
    t = 2.0
    expectedResult = Cartesian2(12.0, 32.0)
    result = Cartesian2.lerp(start, end, t, Cartesian2())
    assert(result == expectedResult)

    start = Cartesian2(4.0, 8.0)
    end = Cartesian2(8.0, 20.0)
    t = -1.0
    expectedResult = Cartesian2(0.0, -4.0)
    result = Cartesian2.lerp(start, end, t, Cartesian2())
    assert(result == expectedResult)

    x = Cartesian2.UNIT_X()
    y = Cartesian2.UNIT_Y()
    assert(Cartesian2.angleBetween(x, y) == PI_OVER_TWO)
    assert(Cartesian2.angleBetween(y, x) == PI_OVER_TWO)

    x = Cartesian2(0.0, 1.0)
    y = Cartesian2(1.0, 1.0)
    assert(np.isclose(Cartesian2.angleBetween(x, y), PI_OVER_FOUR, 0, EPSILON14))
    assert(np.isclose(Cartesian2.angleBetween(y, x), PI_OVER_FOUR, 0, EPSILON14))

    x = Cartesian2(0.0, 1.0)
    y = Cartesian2(-1.0, -1.0)
    assert(np.isclose(Cartesian2.angleBetween(x, y), (PI * 3.0) / 4.0, EPSILON14))
    assert(np.isclose(Cartesian2.angleBetween(y, x), (PI * 3.0) / 4.0, EPSILON14))

    x = Cartesian2.UNIT_X()
    assert(Cartesian2.angleBetween(x, x) == 0.0)

    v = Cartesian2(0.0, 1.0)
    assert(Cartesian2.mostOrthogonalAxis(v, Cartesian2()) ==
           Cartesian2.UNIT_X()
           )

    v = Cartesian2(1.0, 0.0)
    assert(Cartesian2.mostOrthogonalAxis(v, Cartesian2()) ==
           Cartesian2.UNIT_Y()
           )

    cartesian = Cartesian2(1.0, 2.0)
    assert(Cartesian2.equals(cartesian, Cartesian2(1.0, 2.0)) is
           True
           )
    assert(Cartesian2.equals(cartesian, Cartesian2(2.0, 2.0)) is
           False
           )
    assert(Cartesian2.equals(cartesian, Cartesian2(2.0, 1.0)) is
           False
           )
    assert(Cartesian2.equals(cartesian, None) is False)

    cartesian = Cartesian2(1.0, 2.0)
    assert(Cartesian2.equalsEpsilon(cartesian, Cartesian2(1.0, 2.0), 0.0) is
           True
           )
    assert(Cartesian2.equalsEpsilon(cartesian, Cartesian2(1.0, 2.0), 1.0) is
           True
           )
    assert(Cartesian2.equalsEpsilon(cartesian, Cartesian2(2.0, 2.0), 1.0) is
           True
           )
    assert(Cartesian2.equalsEpsilon(cartesian, Cartesian2(1.0, 3.0), 1.0) is
           True
           )
    assert(
        Cartesian2.equalsEpsilon(cartesian, Cartesian2(1.0, 3.0), EPSILON6)
        is False)
    assert(Cartesian2.equalsEpsilon(cartesian, None, 1) is False)

    cartesian = Cartesian2(3000000.0, 4000000.0)
    assert(
        Cartesian2.equalsEpsilon(cartesian, Cartesian2(3000000.0, 4000000.0), 0.0)
        is True)
    assert(
        Cartesian2.equalsEpsilon(cartesian,
                                 Cartesian2(3000000.0, 4000000.2),
                                 EPSILON7
                                 )
        is True)
    assert(
        Cartesian2.equalsEpsilon(cartesian,
                                 Cartesian2(3000000.2, 4000000.0),
                                 EPSILON7
                                 )
        is True)
    assert(
        Cartesian2.equalsEpsilon(cartesian,
                                 Cartesian2(3000000.2, 4000000.2),
                                 EPSILON7
                                 )
        is True)
    assert(
        Cartesian2.equalsEpsilon(cartesian,
                                 Cartesian2(3000000.2, 4000000.2),
                                 EPSILON9
                                 )
        is False)
    assert(Cartesian2.equalsEpsilon(cartesian, None, 1) is False)

    assert(Cartesian2.equalsEpsilon(None, cartesian, 1) is False)

    cartesian = Cartesian2(1.123, 2.345)
    assert(str(cartesian) == "[1.123, 2.345]")

    assert(Cartesian2.clone() is None)
    assert(Cartesian2.clone(Cartesian2(1.0, 2.0)) == Cartesian2(1.0, 2.0))

    cartesian2 = Cartesian2.fromElements(2, 2)
    expectedResult = Cartesian2(2, 2)
    assert(cartesian2 == expectedResult)

    cartesian2 = Cartesian2()
    Cartesian2.fromElements(2, 2, cartesian2)
    expectedResult = Cartesian2(2, 2)
    assert(cartesian2 == expectedResult)

    cartesian = Cartesian2.ONE()
    assert(cartesian.x == 1.0)
    assert(cartesian.y == 1.0)

    cartesian = Cartesian2.ZERO()
    assert(cartesian.x == 0.0)
    assert(cartesian.y == 0.0)

    cartesian = Cartesian2.fromCartesian3(Cartesian3(1, 2, 3))
    assert(cartesian.x == 1.0)
    assert(cartesian.y == 2.0)

    cartesian = Cartesian2.fromCartesian4(Cartesian4(1, 2, 3, 4))
    assert(cartesian.x == 1.0)
    assert(cartesian.y == 2.0)


def test_cartesian3():

    cartesian = Cartesian3()
    assert(cartesian.x == 0.0)
    assert(cartesian.y == 0.0)
    assert(cartesian.z == 0.0)

    cartesian = Cartesian3(1.0, 2.0, 3.0)
    assert(cartesian.x == 1.0)
    assert(cartesian.y == 2.0)
    assert(cartesian.z == 3.0)

    fortyFiveDegrees = math.pi / 4.0
    sixtyDegrees = math.pi / 3.0
    cartesian = Cartesian3(1.0, math.sqrt(3.0), -2.0)
    spherical = {
        'clock': sixtyDegrees,
        'cone': fortyFiveDegrees + math.pi / 2.0,
        'magnitude': math.sqrt(8.0),
    }
    existing = Cartesian3()
    assert(Cartesian3.equalsEpsilon(cartesian, Cartesian3.fromSpherical(spherical, existing), EPSILON15))
    assert(Cartesian3.equalsEpsilon(cartesian, Cartesian3.fromSpherical(spherical), EPSILON15))
    assert(Cartesian3.equalsEpsilon(cartesian, existing, EPSILON15))

    cartesian = Cartesian3.fromArray([0.0, 1.0, 2.0, 3.0, 0.0], 1)
    assert(cartesian == Cartesian3(1.0, 2.0, 3.0))

    cartesian = Cartesian3()
    result = Cartesian3.fromArray([1.0, 2.0, 3.0], 0, cartesian)
    assert(result is cartesian)
    assert(result == Cartesian3(1.0, 2.0, 3.0))

    cartesian = Cartesian3(1.0, 2.0, 3.0)
    result = Cartesian3()
    returnedResult = Cartesian3.clone(cartesian, result)
    assert(cartesian is not result)
    assert(result is returnedResult)
    assert(cartesian == result)

    cartesian = Cartesian3(1.0, 2.0, 3.0)
    returnedResult = Cartesian3.clone(cartesian, cartesian)
    assert(cartesian is returnedResult)

    cartesian = Cartesian3(2.0, 1.0, 0.0)
    assert(Cartesian3.maximumComponent(cartesian) == cartesian.x)

    cartesian = Cartesian3(1.0, 2.0, 0.0)
    assert(Cartesian3.maximumComponent(cartesian) == cartesian.y)

    cartesian = Cartesian3(1.0, 2.0, 3.0)
    assert(Cartesian3.maximumComponent(cartesian) == cartesian.z)

    cartesian = Cartesian3(1.0, 2.0, 3.0)
    assert(Cartesian3.minimumComponent(cartesian) == cartesian.x)

    cartesian = Cartesian3(2.0, 1.0, 3.0)
    assert(Cartesian3.minimumComponent(cartesian) == cartesian.y)

    cartesian = Cartesian3(2.0, 1.0, 0.0)
    assert(Cartesian3.minimumComponent(cartesian) == cartesian.z)

    result = Cartesian3()

    first = Cartesian3(2.0, 0.0, 0.0)
    second = Cartesian3(1.0, 0.0, 0.0)
    expected = Cartesian3(1.0, 0.0, 0.0)
    assert(Cartesian3.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian3(1.0, 0.0, 0.0)
    second = Cartesian3(2.0, 0.0, 0.0)
    expected = Cartesian3(1.0, 0.0, 0.0)
    assert(Cartesian3.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian3(2.0, -15.0, 0.0)
    second = Cartesian3(1.0, -20.0, 0.0)
    expected = Cartesian3(1.0, -20.0, 0.0)
    assert(Cartesian3.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian3(2.0, -20.0, 0.0)
    second = Cartesian3(1.0, -15.0, 0.0)
    expected = Cartesian3(1.0, -20.0, 0.0)
    assert(Cartesian3.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian3(2.0, -15.0, 26.4)
    second = Cartesian3(1.0, -20.0, 26.5)
    expected = Cartesian3(1.0, -20.0, 26.4)
    assert(Cartesian3.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian3(2.0, -15.0, 26.5)
    second = Cartesian3(1.0, -20.0, 26.4)
    expected = Cartesian3(1.0, -20.0, 26.4)
    assert(Cartesian3.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian3(2.0, 0.0, 0.0)
    second = Cartesian3(1.0, 0.0, 0.0)
    expected = Cartesian3(1.0, 0.0, 0.0)
    result = Cartesian3()
    returnedResult = Cartesian3.minimumByComponent(first, second, result)
    assert(returnedResult is result)
    assert(returnedResult == expected)

    first = Cartesian3(2.0, 0.0, 0.0)
    second = Cartesian3(1.0, 0.0, 0.0)
    expected = Cartesian3(1.0, 0.0, 0.0)
    assert(Cartesian3.minimumByComponent(first, second, first) ==
           expected
           )

    first.x = 1.0
    second.x = 2.0
    assert(Cartesian3.minimumByComponent(first, second, second) ==
           expected
           )

    first = Cartesian3(2.0, 0.0, 0.0)
    second = Cartesian3(1.0, 0.0, 0.0)
    expected = Cartesian3(1.0, 0.0, 0.0)
    result = Cartesian3()
    assert(Cartesian3.minimumByComponent(first, second, result) ==
           expected
           )

    second.x = 3.0
    expected.x = 2.0
    assert(Cartesian3.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian3(0.0, 2.0, 0.0)
    second = Cartesian3(0.0, 1.0, 0.0)
    expected = Cartesian3(0.0, 1.0, 0.0)
    result = Cartesian3()
    assert(Cartesian3.minimumByComponent(first, second, result) ==
           expected
           )

    second.y = 3.0
    expected.y = 2.0
    assert(Cartesian3.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian3(0.0, 0.0, 2.0)
    second = Cartesian3(0.0, 0.0, 1.0)
    expected = Cartesian3(0.0, 0.0, 1.0)
    result = Cartesian3()
    assert(Cartesian3.minimumByComponent(first, second, result) ==
           expected
           )

    second.z = 3.0
    expected.z = 2.0
    assert(Cartesian3.minimumByComponent(first, second, result) ==
           expected
           )

    result = Cartesian3()

    first = Cartesian3(2.0, 0.0, 0.0)
    second = Cartesian3(1.0, 0.0, 0.0)
    expected = Cartesian3(2.0, 0.0, 0.0)
    assert(Cartesian3.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian3(1.0, 0.0, 0.0)
    second = Cartesian3(2.0, 0.0, 0.0)
    expected = Cartesian3(2.0, 0.0, 0.0)
    assert(Cartesian3.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian3(2.0, -15.0, 0.0)
    second = Cartesian3(1.0, -20.0, 0.0)
    expected = Cartesian3(2.0, -15.0, 0.0)
    assert(Cartesian3.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian3(2.0, -20.0, 0.0)
    second = Cartesian3(1.0, -15.0, 0.0)
    expected = Cartesian3(2.0, -15.0, 0.0)
    assert(Cartesian3.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian3(2.0, -15.0, 26.4)
    second = Cartesian3(1.0, -20.0, 26.5)
    expected = Cartesian3(2.0, -15.0, 26.5)
    assert(Cartesian3.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian3(2.0, -15.0, 26.5)
    second = Cartesian3(1.0, -20.0, 26.4)
    expected = Cartesian3(2.0, -15.0, 26.5)
    assert(Cartesian3.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian3(2.0, 0.0, 0.0)
    second = Cartesian3(1.0, 0.0, 0.0)
    expected = Cartesian3(2.0, 0.0, 0.0)
    result = Cartesian3()
    returnedResult = Cartesian3.maximumByComponent(first, second, result)
    assert(returnedResult is result)
    assert(returnedResult == expected)

    first = Cartesian3(2.0, 0.0, 0.0)
    second = Cartesian3(1.0, 0.0, 0.0)
    expected = Cartesian3(2.0, 0.0, 0.0)
    assert(Cartesian3.maximumByComponent(first, second, first) ==
           expected
           )

    first.x = 1.0
    second.x = 2.0
    assert(Cartesian3.maximumByComponent(first, second, second) ==
           expected
           )

    first = Cartesian3(2.0, 0.0, 0.0)
    second = Cartesian3(1.0, 0.0, 0.0)
    expected = Cartesian3(2.0, 0.0, 0.0)
    result = Cartesian3()
    assert(Cartesian3.maximumByComponent(first, second, result) ==
           expected
           )

    second.x = 3.0
    expected.x = 3.0
    assert(Cartesian3.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian3(0.0, 2.0, 0.0)
    second = Cartesian3(0.0, 1.0, 0.0)
    expected = Cartesian3(0.0, 2.0, 0.0)
    result = Cartesian3()
    assert(Cartesian3.maximumByComponent(first, second, result) ==
           expected
           )

    second.y = 3.0
    expected.y = 3.0
    assert(Cartesian3.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian3(0.0, 0.0, 2.0)
    second = Cartesian3(0.0, 0.0, 1.0)
    expected = Cartesian3(0.0, 0.0, 2.0)
    result = Cartesian3()
    assert(Cartesian3.maximumByComponent(first, second, result) ==
           expected
           )

    second.z = 3.0
    expected.z = 3.0
    assert(Cartesian3.maximumByComponent(first, second, result) ==
           expected
           )

    result = Cartesian3()

    value = Cartesian3(-1.0, 0.0, 0.0)
    min = Cartesian3(0.0, 0.0, 0.0)
    max = Cartesian3(1.0, 1.0, 1.0)
    expected = Cartesian3(0.0, 0.0, 0.0)
    assert(Cartesian3.clamp(value, min, max, result) == expected)

    value = Cartesian3(2.0, 0.0, 0.0)
    min = Cartesian3(0.0, 0.0, 0.0)
    max = Cartesian3(1.0, 1.0, 1.0)
    expected = Cartesian3(1.0, 0.0, 0.0)
    assert(Cartesian3.clamp(value, min, max, result) == expected)

    value = Cartesian3(0.0, -1.0, 0.0)
    min = Cartesian3(0.0, 0.0, 0.0)
    max = Cartesian3(1.0, 1.0, 1.0)
    expected = Cartesian3(0.0, 0.0, 0.0)
    assert(Cartesian3.clamp(value, min, max, result) == expected)

    value = Cartesian3(0.0, 2.0, 0.0)
    min = Cartesian3(0.0, 0.0, 0.0)
    max = Cartesian3(1.0, 1.0, 1.0)
    expected = Cartesian3(0.0, 1.0, 0.0)
    assert(Cartesian3.clamp(value, min, max, result) == expected)

    value = Cartesian3(0.0, 0.0, -1.0)
    min = Cartesian3(0.0, 0.0, 0.0)
    max = Cartesian3(1.0, 1.0, 1.0)
    expected = Cartesian3(0.0, 0.0, 0.0)
    assert(Cartesian3.clamp(value, min, max, result) == expected)

    value = Cartesian3(0.0, 0.0, 2.0)
    min = Cartesian3(0.0, 0.0, 0.0)
    max = Cartesian3(1.0, 1.0, 1.0)
    expected = Cartesian3(0.0, 0.0, 1.0)
    assert(Cartesian3.clamp(value, min, max, result) == expected)

    value = Cartesian3(-2.0, 3.0, 4.0)
    min = Cartesian3(0.0, 0.0, 0.0)
    max = Cartesian3(1.0, 1.0, 1.0)
    expected = Cartesian3(0.0, 1.0, 1.0)
    assert(Cartesian3.clamp(value, min, max, result) == expected)

    value = Cartesian3(0.0, 0.0, 0.0)
    min = Cartesian3(1.0, 2.0, 3.0)
    max = Cartesian3(1.0, 2.0, 3.0)
    expected = Cartesian3(1.0, 2.0, 3.0)
    assert(Cartesian3.clamp(value, min, max, result) == expected)

    value = Cartesian3(-1.0, -1.0, -1.0)
    min = Cartesian3(0.0, 0.0, 0.0)
    max = Cartesian3(1.0, 1.0, 1.0)
    expected = Cartesian3(0.0, 0.0, 0.0)
    result = Cartesian3()
    returnedResult = Cartesian3.clamp(value, min, max, result)
    assert(returnedResult is result)
    assert(returnedResult == expected)

    value = Cartesian3(-1.0, -1.0, -1.0)
    min = Cartesian3(0.0, 0.0, 0.0)
    max = Cartesian3(1.0, 1.0, 1.0)
    expected = Cartesian3(0.0, 0.0, 0.0)
    assert(Cartesian3.clamp(value, min, max, value) == expected)

    Cartesian3.fromElements(-1.0, -1.0, -1.0, value)
    assert(Cartesian3.clamp(value, min, max, min) == expected)

    Cartesian3.fromElements(0.0, 0.0, 0.0, min)
    assert(Cartesian3.clamp(value, min, max, max) == expected)

    cartesian = Cartesian3(3.0, 4.0, 5.0)
    assert(Cartesian3.magnitudeSquared(cartesian) == 50.0)

    cartesian = Cartesian3(3.0, 4.0, 5.0)
    assert(Cartesian3.magnitude(cartesian) == math.sqrt(50.0))

    distance = Cartesian3.distance(
        Cartesian3(1.0, 0.0, 0.0),
        Cartesian3(2.0, 0.0, 0.0)
    )
    assert(distance == 1.0)

    distanceSquared = Cartesian3.distanceSquared(
        Cartesian3(1.0, 0.0, 0.0),
        Cartesian3(3.0, 0.0, 0.0)
    )
    assert(distanceSquared == 4.0)

    cartesian = Cartesian3(2.0, 0.0, 0.0)
    expectedResult = Cartesian3(1.0, 0.0, 0.0)
    result = Cartesian3()
    returnedResult = Cartesian3.normalize(cartesian, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    cartesian = Cartesian3(2.0, 0.0, 0.0)
    expectedResult = Cartesian3(1.0, 0.0, 0.0)
    returnedResult = Cartesian3.normalize(cartesian, cartesian)
    assert(cartesian is returnedResult)
    assert(cartesian == expectedResult)

    left = Cartesian3(2.0, 3.0, 6.0)
    right = Cartesian3(4.0, 5.0, 7.0)
    result = Cartesian3()
    expectedResult = Cartesian3(8.0, 15.0, 42.0)
    returnedResult = Cartesian3.multiplyComponents(left, right, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    left = Cartesian3(2.0, 3.0, 6.0)
    right = Cartesian3(4.0, 5.0, 7.0)
    expectedResult = Cartesian3(8.0, 15.0, 42.0)
    returnedResult = Cartesian3.multiplyComponents(left, right, left)
    assert(left is returnedResult)
    assert(left == expectedResult)

    left = Cartesian3(2.0, 3.0, 6.0)
    right = Cartesian3(4.0, 5.0, 8.0)
    result = Cartesian3()
    expectedResult = Cartesian3(0.5, 0.6, 0.75)
    returnedResult = Cartesian3.divideComponents(left, right, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    left = Cartesian3(2.0, 3.0, 6.0)
    right = Cartesian3(4.0, 5.0, 8.0)
    expectedResult = Cartesian3(0.5, 0.6, 0.75)
    returnedResult = Cartesian3.divideComponents(left, right, left)
    assert(left is returnedResult)
    assert(left == expectedResult)

    left = Cartesian3(2.0, 3.0, 6.0)
    right = Cartesian3(4.0, 5.0, 7.0)
    expectedResult = 65.0
    result = Cartesian3.dot(left, right)
    assert(result == expectedResult)

    left = Cartesian3(2.0, 3.0, 6.0)
    right = Cartesian3(4.0, 5.0, 7.0)
    result = Cartesian3()
    expectedResult = Cartesian3(6.0, 8.0, 13.0)
    returnedResult = Cartesian3.add(left, right, result)
    assert(returnedResult is result)
    assert(result == expectedResult)

    left = Cartesian3(2.0, 3.0, 6.0)
    right = Cartesian3(4.0, 5.0, 7.0)
    expectedResult = Cartesian3(6.0, 8.0, 13.0)
    returnedResult = Cartesian3.add(left, right, left)
    assert(returnedResult is left)
    assert(left == expectedResult)

    left = Cartesian3(2.0, 3.0, 4.0)
    right = Cartesian3(1.0, 5.0, 7.0)
    result = Cartesian3()
    expectedResult = Cartesian3(1.0, -2.0, -3.0)
    returnedResult = Cartesian3.subtract(left, right, result)
    assert(returnedResult is result)
    assert(result == expectedResult)

    left = Cartesian3(2.0, 3.0, 4.0)
    right = Cartesian3(1.0, 5.0, 7.0)
    expectedResult = Cartesian3(1.0, -2.0, -3.0)
    returnedResult = Cartesian3.subtract(left, right, left)
    assert(returnedResult is left)
    assert(left == expectedResult)

    cartesian = Cartesian3(1.0, 2.0, 3.0)
    result = Cartesian3()
    scalar = 2
    expectedResult = Cartesian3(2.0, 4.0, 6.0)
    returnedResult = Cartesian3.multiplyByScalar(
        cartesian,
        scalar,
        result
    )
    assert(result is returnedResult)
    assert(result == expectedResult)

    cartesian = Cartesian3(1.0, 2.0, 3.0)
    scalar = 2
    expectedResult = Cartesian3(2.0, 4.0, 6.0)
    returnedResult = Cartesian3.multiplyByScalar(
        cartesian,
        scalar,
        cartesian
    )
    assert(cartesian is returnedResult)
    assert(cartesian == expectedResult)

    cartesian = Cartesian3(1.0, 2.0, 3.0)
    result = Cartesian3()
    scalar = 2
    expectedResult = Cartesian3(0.5, 1.0, 1.5)
    returnedResult = Cartesian3.divideByScalar(cartesian, scalar, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    cartesian = Cartesian3(1.0, 2.0, 3.0)
    scalar = 2
    expectedResult = Cartesian3(0.5, 1.0, 1.5)
    returnedResult = Cartesian3.divideByScalar(
        cartesian,
        scalar,
        cartesian
    )
    assert(cartesian is returnedResult)
    assert(cartesian == expectedResult)

    cartesian = Cartesian3(1.0, -2.0, -5.0)
    expectedResult = Cartesian3(-1.0, 2.0, 5.0)
    result = Cartesian3.negate(cartesian, Cartesian3())
    assert(result == expectedResult)

    cartesian = Cartesian3(1.0, -2.0, -5.0)
    result = Cartesian3()
    expectedResult = Cartesian3(-1.0, 2.0, 5.0)
    returnedResult = Cartesian3.negate(cartesian, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    cartesian = Cartesian3(1.0, -2.0, -5.0)
    expectedResult = Cartesian3(-1.0, 2.0, 5.0)
    returnedResult = Cartesian3.negate(cartesian, cartesian)
    assert(cartesian is returnedResult)
    assert(cartesian == expectedResult)

    cartesian = Cartesian3(1.0, -2.0, -4.0)
    expectedResult = Cartesian3(1.0, 2.0, 4.0)
    result = Cartesian3.abs(cartesian, Cartesian3())
    assert(result == expectedResult)

    cartesian = Cartesian3(1.0, -2.0, -4.0)
    result = Cartesian3()
    expectedResult = Cartesian3(1.0, 2.0, 4.0)
    returnedResult = Cartesian3.abs(cartesian, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    cartesian = Cartesian3(1.0, -2.0, -4.0)
    expectedResult = Cartesian3(1.0, 2.0, 4.0)
    returnedResult = Cartesian3.abs(cartesian, cartesian)
    assert(cartesian is returnedResult)
    assert(cartesian == expectedResult)

    start = Cartesian3(4.0, 8.0, 10.0)
    end = Cartesian3(8.0, 20.0, 20.0)
    t = 0.25
    result = Cartesian3()
    expectedResult = Cartesian3(5.0, 11.0, 12.5)
    returnedResult = Cartesian3.lerp(start, end, t, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    start = Cartesian3(4.0, 8.0, 10.0)
    end = Cartesian3(8.0, 20.0, 20.0)
    t = 0.25
    expectedResult = Cartesian3(5.0, 11.0, 12.5)
    returnedResult = Cartesian3.lerp(start, end, t, start)
    assert(start is returnedResult)
    assert(start == expectedResult)

    start = Cartesian3(4.0, 8.0, 10.0)
    end = Cartesian3(8.0, 20.0, 20.0)
    t = 2.0
    expectedResult = Cartesian3(12.0, 32.0, 30.0)
    result = Cartesian3.lerp(start, end, t, Cartesian3())
    assert(result == expectedResult)

    start = Cartesian3(4.0, 8.0, 10.0)
    end = Cartesian3(8.0, 20.0, 20.0)
    t = -1.0
    expectedResult = Cartesian3(0.0, -4.0, 0.0)
    result = Cartesian3.lerp(start, end, t, Cartesian3())
    assert(result == expectedResult)

    x = Cartesian3.UNIT_X()
    y = Cartesian3.UNIT_Y()
    assert(Cartesian3.angleBetween(x, y) == PI_OVER_TWO)
    assert(Cartesian3.angleBetween(y, x) == PI_OVER_TWO)

    x = Cartesian3(0.0, 1.0, 0.0)
    y = Cartesian3(1.0, 1.0, 0.0)
    assert(np.isclose(Cartesian3.angleBetween(x, y), PI_OVER_FOUR, 0, EPSILON14))
    assert(np.isclose(Cartesian3.angleBetween(y, x), PI_OVER_FOUR, 0, EPSILON14))

    x = Cartesian3(0.0, 1.0, 0.0)
    y = Cartesian3(0.0, -1.0, -1.0)
    assert(np.isclose(Cartesian3.angleBetween(x, y), (PI * 3.0) / 4.0, 0, EPSILON14))
    assert(np.isclose(Cartesian3.angleBetween(y, x), (PI * 3.0) / 4.0, 0, EPSILON14))

    x = Cartesian3.UNIT_X()
    assert(Cartesian3.angleBetween(x, x) == 0.0)

    v = Cartesian3(0.0, 1.0, 2.0)
    assert(Cartesian3.mostOrthogonalAxis(v, Cartesian3()) ==
           Cartesian3.UNIT_X()
           )

    v = Cartesian3(1.0, 0.0, 2.0)
    assert(Cartesian3.mostOrthogonalAxis(v, Cartesian3()) ==
           Cartesian3.UNIT_Y()
           )

    v = Cartesian3(1.0, 3.0, 0.0)
    assert(Cartesian3.mostOrthogonalAxis(v, Cartesian3()) ==
           Cartesian3.UNIT_Z()
           )

    v = Cartesian3(3.0, 1.0, 0.0)
    assert(Cartesian3.mostOrthogonalAxis(v, Cartesian3()) ==
           Cartesian3.UNIT_Z()
           )

    cartesian = Cartesian3(1.0, 2.0, 3.0)
    assert(Cartesian3.equals(cartesian, Cartesian3(1.0, 2.0, 3.0)) is
           True
           )
    assert(Cartesian3.equals(cartesian, Cartesian3(2.0, 2.0, 3.0)) is
           False
           )
    assert(Cartesian3.equals(cartesian, Cartesian3(2.0, 1.0, 3.0)) is
           False
           )
    assert(Cartesian3.equals(cartesian, Cartesian3(1.0, 2.0, 4.0)) is
           False
           )
    assert(Cartesian3.equals(cartesian, None) is False)

    cartesian = Cartesian3(1.0, 2.0, 3.0)
    assert(Cartesian3.equalsEpsilon(cartesian, Cartesian3(1.0, 2.0, 3.0), 0.0) is
           True
           )
    assert(Cartesian3.equalsEpsilon(cartesian, Cartesian3(1.0, 2.0, 3.0), 1.0) is
           True
           )
    assert(Cartesian3.equalsEpsilon(cartesian, Cartesian3(2.0, 2.0, 3.0), 1.0) is
           True
           )
    assert(Cartesian3.equalsEpsilon(cartesian, Cartesian3(1.0, 3.0, 3.0), 1.0) is
           True
           )
    assert(Cartesian3.equalsEpsilon(cartesian, Cartesian3(1.0, 2.0, 4.0), 1.0) is
           True
           )
    assert(
        Cartesian3.equalsEpsilon(cartesian,
                                 Cartesian3(2.0, 2.0, 3.0),
                                 EPSILON6
                                 )
        is False)
    assert(
        Cartesian3.equalsEpsilon(cartesian,
                                 Cartesian3(1.0, 3.0, 3.0),
                                 EPSILON6
                                 )
        is False)
    assert(
        Cartesian3.equalsEpsilon(cartesian,
                                 Cartesian3(1.0, 2.0, 4.0),
                                 EPSILON6
                                 )
        is False)
    assert(Cartesian3.equalsEpsilon(cartesian, None, 1) is False)

    cartesian = Cartesian3(3000000.0, 4000000.0, 5000000.0)
    assert(
        Cartesian3.equalsEpsilon(cartesian,
                                 Cartesian3(3000000.0, 4000000.0, 5000000.0),
                                 0.0
                                 )
        is True)
    assert(
        Cartesian3.equalsEpsilon(cartesian,
                                 Cartesian3(3000000.2, 4000000.0, 5000000.0),
                                 EPSILON7
                                 )
        is True)
    assert(
        Cartesian3.equalsEpsilon(cartesian,
                                 Cartesian3(3000000.0, 4000000.2, 5000000.0),
                                 EPSILON7
                                 )
        is True)
    assert(
        Cartesian3.equalsEpsilon(cartesian,
                                 Cartesian3(3000000.0, 4000000.0, 5000000.2),
                                 EPSILON7
                                 )
        is True)
    assert(
        Cartesian3.equalsEpsilon(cartesian,
                                 Cartesian3(3000000.2, 4000000.2, 5000000.2),
                                 EPSILON7
                                 )
        is True)
    assert(
        Cartesian3.equalsEpsilon(cartesian,
                                 Cartesian3(3000000.2, 4000000.2, 5000000.2),
                                 EPSILON9
                                 )
        is False)
    assert(Cartesian3.equalsEpsilon(cartesian, None, 1) is False)

    assert(Cartesian3.equalsEpsilon(None, cartesian, 1) is False)

    cartesian = Cartesian3(1.123, 2.345, 6.789)
    assert(str(cartesian) == "[1.123, 2.345, 6.789]")

    left = Cartesian3(1, 2, 5)
    right = Cartesian3(4, 3, 6)
    result = Cartesian3()
    expectedResult = Cartesian3(-3, 14, -5)
    returnedResult = Cartesian3.cross(left, right, result)
    assert(returnedResult is result)
    assert(result == expectedResult)

    left = Cartesian3(1, 2, 5)
    right = Cartesian3(4, 3, 6)
    expectedResult = Cartesian3(-3, 14, -5)
    returnedResult = Cartesian3.cross(left, right, left)
    assert(returnedResult is left)
    assert(left == expectedResult)

    left = Cartesian3(0.0, 0.0, 6.0)
    right = Cartesian3(0.0, 0.0, -6.0)
    result = Cartesian3()
    expectedResult = Cartesian3(0.0, 0.0, 0.0)
    returnedResult = Cartesian3.midpoint(left, right, result)
    assert(returnedResult is result)
    assert(result == expectedResult)

    assert(Cartesian3.clone() is None)
    assert(Cartesian2.clone(Cartesian3(1.0, 2.0, 3.0)) == Cartesian3(1.0, 2.0, 3.0))

    right = Cartesian3(4.0, 5.0, 6.0)

    left = Cartesian3(4.0, 5.0, 6.0)

    right = Cartesian3(4.0, 5.0, 6.0)

    left = Cartesian3(4.0, 5.0, 6.0)
    end = Cartesian3(8.0, 20.0, 6.0)
    t = 0.25

    start = Cartesian3(4.0, 8.0, 6.0)
    t = 0.25

    start = Cartesian3(4.0, 8.0, 6.0)
    end = Cartesian3(8.0, 20.0, 6.0)

    right = Cartesian3(8.0, 20.0, 6.0)

    left = Cartesian3(4.0, 8.0, 6.0)

    right = Cartesian3(4, 3, 6)

    left = Cartesian3(1, 2, 5)

    cartesian = Cartesian3.fromElements(2, 2, 4)
    expectedResult = Cartesian3(2, 2, 4)
    assert(cartesian == expectedResult)

    cartesian3 = Cartesian3()
    Cartesian3.fromElements(2, 2, 4, cartesian3)
    expectedResult = Cartesian3(2, 2, 4)
    assert(cartesian3 == expectedResult)

    # lon = -115
    # lat = 37
    # ellipsoid = Ellipsoid.WGS84
    # actual = Cartesian3.fromDegrees(lon, lat)
    # expected = ellipsoid.cartographicToCartesian(
    #   Cartographic.fromDegrees(lon, lat)
    # )
    # assert(actual == expected)

    # lon = -115
    # lat = 37
    # height = 100000
    # ellipsoid = Ellipsoid.WGS84
    # actual = Cartesian3.fromDegrees(lon, lat, height)
    # expected = ellipsoid.cartographicToCartesian(
    #   Cartographic.fromDegrees(lon, lat, height)
    # )
    # assert(actual == expected)

    # lon = -115
    # lat = 37
    # height = 100000
    # ellipsoid = Ellipsoid.WGS84
    # result = Cartesian3()
    # actual = Cartesian3.fromDegrees(lon, lat, height, ellipsoid, result)
    # expected = ellipsoid.cartographicToCartesian(
    #   Cartographic.fromDegrees(lon, lat, height)
    # )
    # assert(actual == expected)
    # assert(actual is result)

    # lon = np.radians(150)
    # lat = np.radians(-40)
    # ellipsoid = Ellipsoid.WGS84
    # actual = Cartesian3.fromRadians(lon, lat)
    # expected = ellipsoid.cartographicToCartesian(
    #   Cartographic(lon, lat)
    # )
    # assert(actual == expected)

    # lon = np.radians(150)
    # lat = np.radians(-40)
    # height = 100000
    # ellipsoid = Ellipsoid.WGS84
    # actual = Cartesian3.fromRadians(lon, lat, height)
    # expected = ellipsoid.cartographicToCartesian(
    #   Cartographic(lon, lat, height)
    # )
    # assert(actual == expected)

    # lon = np.radians(150)
    # lat = np.radians(-40)
    # height = 100000
    # ellipsoid = Ellipsoid.WGS84
    # result = Cartesian3()
    # actual = Cartesian3.fromRadians(lon, lat, height, ellipsoid, result)
    # expected = ellipsoid.cartographicToCartesian(
    #   Cartographic(lon, lat, height)
    # )
    # assert(actual == expected)
    # assert(actual is result)

    # lon1 = 90
    # lat1 = -70
    # lon2 = -100
    # lat2 = 40

    # ellipsoid = Ellipsoid.WGS84
    # actual = Cartesian3.fromDegreesArray([lon1, lat1, lon2, lat2])
    # expected = ellipsoid.cartographicArrayToCartesianArray([
    #   Cartographic.fromDegrees(lon1, lat1),
    #   Cartographic.fromDegrees(lon2, lat2),
    # ])
    # assert(actual == expected)

    # lon1 = np.radians(90)
    # lat1 = np.radians(-70)
    # lon2 = np.radians(-100)
    # lat2 = np.radians(40)

    # ellipsoid = Ellipsoid.WGS84
    # actual = Cartesian3.fromRadiansArray([lon1, lat1, lon2, lat2])
    # expected = ellipsoid.cartographicArrayToCartesianArray([
    #   Cartographic(lon1, lat1),
    #   Cartographic(lon2, lat2),
    # ])
    # assert(actual == expected)

    # lon1 = np.radians(90)
    # lat1 = np.radians(-70)
    # lon2 = np.radians(-100)
    # lat2 = np.radians(40)

    # ellipsoid = Ellipsoid.WGS84
    # result = [Cartesian3(), Cartesian3()]
    # actual = Cartesian3.fromRadiansArray(
    #   [lon1, lat1, lon2, lat2],
    #   ellipsoid,
    #   result
    # )
    # expected = ellipsoid.cartographicArrayToCartesianArray([
    #   Cartographic(lon1, lat1),
    #   Cartographic(lon2, lat2),
    # ])
    # assert(result == expected)
    # assert(actual is result)

    # lon1 = 90
    # lat1 = -70
    # alt1 = 200000
    # lon2 = -100
    # lat2 = 40
    # alt2 = 100000

    # ellipsoid = Ellipsoid.WGS84
    # actual = Cartesian3.fromDegreesArrayHeights([
    #   lon1,
    #   lat1,
    #   alt1,
    #   lon2,
    #   lat2,
    #   alt2,
    # ])
    # expected = ellipsoid.cartographicArrayToCartesianArray([
    #   Cartographic.fromDegrees(lon1, lat1, alt1),
    #   Cartographic.fromDegrees(lon2, lat2, alt2),
    # ])
    # assert(actual == expected)

    # lon1 = np.radians(90)
    # lat1 = np.radians(-70)
    # alt1 = 200000
    # lon2 = np.radians(-100)
    # lat2 = np.radians(40)
    # alt2 = 100000

    # ellipsoid = Ellipsoid.WGS84
    # actual = Cartesian3.fromRadiansArrayHeights([
    #   lon1,
    #   lat1,
    #   alt1,
    #   lon2,
    #   lat2,
    #   alt2,
    # ])
    # expected = ellipsoid.cartographicArrayToCartesianArray([
    #   Cartographic(lon1, lat1, alt1),
    #   Cartographic(lon2, lat2, alt2),
    # ])
    # assert(actual == expected)

    # lon1 = np.radians(90)
    # lat1 = np.radians(-70)
    # alt1 = 200000
    # lon2 = np.radians(-100)
    # lat2 = np.radians(40)
    # alt2 = 100000

    # ellipsoid = Ellipsoid.WGS84
    # result = [Cartesian3(), Cartesian3()]
    # actual = Cartesian3.fromRadiansArrayHeights(
    #   [lon1, lat1, alt1, lon2, lat2, alt2],
    #   ellipsoid,
    #   result
    # )
    # expected = ellipsoid.cartographicArrayToCartesianArray([
    #   Cartographic(lon1, lat1, alt1),
    #   Cartographic(lon2, lat2, alt2),
    # ])
    # assert(result == expected)
    # assert(actual is result)

    a = Cartesian3(0.0, 1.0, 0.0)
    b = Cartesian3(1.0, 0.0, 0.0)
    result = Cartesian3.projectVector(a, b, Cartesian3())
    assert(result == Cartesian3(0.0, 0.0, 0.0))

    a = Cartesian3(1.0, 1.0, 0.0)
    b = Cartesian3(1.0, 0.0, 0.0)
    result = Cartesian3.projectVector(a, b, Cartesian3())
    assert(result == Cartesian3(1.0, 0.0, 0.0))

    cartesian = Cartesian3.ONE()
    assert(cartesian.x == 1.0)
    assert(cartesian.y == 1.0)
    assert(cartesian.z == 1.0)

    cartesian = Cartesian3.ZERO()
    assert(cartesian.x == 0.0)
    assert(cartesian.y == 0.0)
    assert(cartesian.z == 0.0)

    cartesian = Cartesian3.fromCartesian4(Cartesian4(1, 2, 3, 4))
    assert(cartesian.x == 1.0)
    assert(cartesian.y == 2.0)
    assert(cartesian.z == 3.0)


def test_cartesian4():

    cartesian = Cartesian4()
    assert(cartesian.x == 0.0)
    assert(cartesian.y == 0.0)
    assert(cartesian.z == 0.0)
    assert(cartesian.w == 0.0)

    cartesian = Cartesian4(1.0, 2.0, 3.0, 4.0)
    assert(cartesian.x == 1.0)
    assert(cartesian.y == 2.0)
    assert(cartesian.z == 3.0)
    assert(cartesian.w == 4.0)

    cartesian = Cartesian4.fromArray([1.0, 2.0, 3.0, 4.0])
    assert(cartesian == Cartesian4(1.0, 2.0, 3.0, 4.0))

    cartesian = Cartesian4.fromArray([0.0, 1.0, 2.0, 3.0, 4.0, 0.0], 1)
    assert(cartesian == Cartesian4(1.0, 2.0, 3.0, 4.0))

    cartesian = Cartesian4()
    result = Cartesian4.fromArray([1.0, 2.0, 3.0, 4.0], 0, cartesian)
    assert(result is cartesian)
    assert(result == Cartesian4(1.0, 2.0, 3.0, 4.0))

    cartesian4 = Cartesian4.fromElements(2, 2, 4, 7)
    expectedResult = Cartesian4(2, 2, 4, 7)
    assert(cartesian4 == expectedResult)

    cartesian4 = Cartesian4()
    Cartesian4.fromElements(2, 2, 4, 7, cartesian4)
    expectedResult = Cartesian4(2, 2, 4, 7)
    assert(cartesian4 == expectedResult)

    # cartesian4 = Cartesian4.fromColor(Color(1.0, 2.0, 3.0, 4.0))
    # assert(cartesian4 == Cartesian4(1.0, 2.0, 3.0, 4.0))

    # cartesian4 = Cartesian4()
    # result = Cartesian4.fromColor(
    #   Color(1.0, 2.0, 3.0, 4.0),
    #   cartesian4
    # )
    # assert(cartesian4 is result)
    # assert(cartesian4 == Cartesian4(1.0, 2.0, 3.0, 4.0))

    cartesian = Cartesian4(1.0, 2.0, 3.0, 4.0)
    result = Cartesian4.clone(cartesian, Cartesian4())
    assert(cartesian is not result)
    assert(cartesian == result)

    cartesian = Cartesian4(1.0, 2.0, 3.0, 4.0)
    result = Cartesian4()
    returnedResult = Cartesian4.clone(cartesian, result)
    assert(cartesian is not result)
    assert(result is returnedResult)
    assert(cartesian == result)

    cartesian = Cartesian4(1.0, 2.0, 3.0, 4.0)
    returnedResult = Cartesian4.clone(cartesian, cartesian)
    assert(cartesian is returnedResult)

    cartesian = Cartesian4(2.0, 1.0, 0.0, -1.0)
    assert(Cartesian4.maximumComponent(cartesian) == cartesian.x)

    cartesian = Cartesian4(1.0, 2.0, 0.0, -1.0)
    assert(Cartesian4.maximumComponent(cartesian) == cartesian.y)

    cartesian = Cartesian4(1.0, 2.0, 3.0, -1.0)
    assert(Cartesian4.maximumComponent(cartesian) == cartesian.z)

    cartesian = Cartesian4(1.0, 2.0, 3.0, 4.0)
    assert(Cartesian4.maximumComponent(cartesian) == cartesian.w)

    cartesian = Cartesian4(1.0, 2.0, 3.0, 4.0)
    assert(Cartesian4.minimumComponent(cartesian) == cartesian.x)

    cartesian = Cartesian4(2.0, 1.0, 3.0, 4.0)
    assert(Cartesian4.minimumComponent(cartesian) == cartesian.y)

    cartesian = Cartesian4(2.0, 1.0, 0.0, 4.0)
    assert(Cartesian4.minimumComponent(cartesian) == cartesian.z)

    cartesian = Cartesian4(2.0, 1.0, 0.0, -1.0)
    assert(Cartesian4.minimumComponent(cartesian) == cartesian.w)

    result = Cartesian4()

    first = Cartesian4(2.0, 0.0, 0.0, 0.0)
    second = Cartesian4(1.0, 0.0, 0.0, 0.0)
    expected = Cartesian4(1.0, 0.0, 0.0, 0.0)
    assert(Cartesian4.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(1.0, 0.0, 0.0, 0.0)
    second = Cartesian4(2.0, 0.0, 0.0, 0.0)
    expected = Cartesian4(1.0, 0.0, 0.0, 0.0)
    assert(Cartesian4.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(2.0, -15.0, 0.0, 0.0)
    second = Cartesian4(1.0, -20.0, 0.0, 0.0)
    expected = Cartesian4(1.0, -20.0, 0.0, 0.0)
    assert(Cartesian4.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(2.0, -20.0, 0.0, 0.0)
    second = Cartesian4(1.0, -15.0, 0.0, 0.0)
    expected = Cartesian4(1.0, -20.0, 0.0, 0.0)
    assert(Cartesian4.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(2.0, -15.0, 26.4, 0.0)
    second = Cartesian4(1.0, -20.0, 26.5, 0.0)
    expected = Cartesian4(1.0, -20.0, 26.4, 0.0)
    assert(Cartesian4.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(2.0, -15.0, 26.5, 0.0)
    second = Cartesian4(1.0, -20.0, 26.4, 0.0)
    expected = Cartesian4(1.0, -20.0, 26.4, 0.0)
    assert(Cartesian4.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(2.0, -15.0, 26.4, -450.0)
    second = Cartesian4(1.0, -20.0, 26.5, 450.0)
    expected = Cartesian4(1.0, -20.0, 26.4, -450.0)
    assert(Cartesian4.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(2.0, -15.0, 26.5, 450.0)
    second = Cartesian4(1.0, -20.0, 26.4, -450.0)
    expected = Cartesian4(1.0, -20.0, 26.4, -450.0)
    assert(Cartesian4.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(2.0, 0.0, 0.0, 0.0)
    second = Cartesian4(1.0, 0.0, 0.0, 0.0)
    expected = Cartesian4(1.0, 0.0, 0.0, 0.0)
    result = Cartesian4()
    returnedResult = Cartesian4.minimumByComponent(first, second, result)
    assert(returnedResult is result)
    assert(returnedResult == expected)

    first = Cartesian4(2.0, 0.0, 0.0, 0.0)
    second = Cartesian4(1.0, 0.0, 0.0, 0.0)
    expected = Cartesian4(1.0, 0.0, 0.0, 0.0)
    assert(Cartesian4.minimumByComponent(first, second, first) ==
           expected
           )

    first.x = 1.0
    second.x = 2.0
    assert(Cartesian4.minimumByComponent(first, second, second) ==
           expected
           )

    first = Cartesian4(2.0, 0.0, 0.0, 0.0)
    second = Cartesian4(1.0, 0.0, 0.0, 0.0)
    expected = Cartesian4(1.0, 0.0, 0.0, 0.0)
    result = Cartesian4()
    assert(Cartesian4.minimumByComponent(first, second, result) ==
           expected
           )

    second.x = 3.0
    expected.x = 2.0
    assert(Cartesian4.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(0.0, 2.0, 0.0, 0.0)
    second = Cartesian4(0.0, 1.0, 0.0, 0.0)
    expected = Cartesian4(0.0, 1.0, 0.0, 0.0)
    result = Cartesian4()
    assert(Cartesian4.minimumByComponent(first, second, result) ==
           expected
           )

    second.y = 3.0
    expected.y = 2.0
    assert(Cartesian4.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(0.0, 0.0, 2.0, 0.0)
    second = Cartesian4(0.0, 0.0, 1.0, 0.0)
    expected = Cartesian4(0.0, 0.0, 1.0, 0.0)
    result = Cartesian4()
    assert(Cartesian4.minimumByComponent(first, second, result) ==
           expected
           )

    second.z = 3.0
    expected.z = 2.0
    assert(Cartesian4.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(0.0, 0.0, 0.0, 2.0)
    second = Cartesian4(0.0, 0.0, 0.0, 1.0)
    expected = Cartesian4(0.0, 0.0, 0.0, 1.0)
    result = Cartesian4()
    assert(Cartesian4.minimumByComponent(first, second, result) ==
           expected
           )

    second.w = 3.0
    expected.w = 2.0
    assert(Cartesian4.minimumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(2.0, 0.0, 0.0, 0.0)
    second = Cartesian4(1.0, 0.0, 0.0, 0.0)
    expected = Cartesian4(2.0, 0.0, 0.0, 0.0)
    result = Cartesian4()
    assert(Cartesian4.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(1.0, 0.0, 0.0, 0.0)
    second = Cartesian4(2.0, 0.0, 0.0, 0.0)
    expected = Cartesian4(2.0, 0.0, 0.0, 0.0)
    assert(Cartesian4.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(2.0, -15.0, 0.0, 0.0)
    second = Cartesian4(1.0, -20.0, 0.0, 0.0)
    expected = Cartesian4(2.0, -15.0, 0.0, 0.0)
    assert(Cartesian4.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(2.0, -20.0, 0.0, 0.0)
    second = Cartesian4(1.0, -15.0, 0.0, 0.0)
    expected = Cartesian4(2.0, -15.0, 0.0, 0.0)
    assert(Cartesian4.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(2.0, -15.0, 26.4, 0.0)
    second = Cartesian4(1.0, -20.0, 26.5, 0.0)
    expected = Cartesian4(2.0, -15.0, 26.5, 0.0)
    assert(Cartesian4.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(2.0, -15.0, 26.5, 0.0)
    second = Cartesian4(1.0, -20.0, 26.4, 0.0)
    expected = Cartesian4(2.0, -15.0, 26.5, 0.0)
    assert(Cartesian4.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(2.0, -15.0, 26.5, 450.0)
    second = Cartesian4(1.0, -20.0, 26.4, -450.0)
    expected = Cartesian4(2.0, -15.0, 26.5, 450.0)
    assert(Cartesian4.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(2.0, -15.0, 26.5, -450.0)
    second = Cartesian4(1.0, -20.0, 26.4, 450.0)
    expected = Cartesian4(2.0, -15.0, 26.5, 450.0)
    assert(Cartesian4.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(2.0, 0.0, 0.0, 0.0)
    second = Cartesian4(1.0, 0.0, 0.0, 0.0)
    expected = Cartesian4(2.0, 0.0, 0.0, 0.0)
    result = Cartesian4()
    returnedResult = Cartesian4.maximumByComponent(first, second, result)
    assert(returnedResult is result)
    assert(returnedResult == expected)

    first = Cartesian4(2.0, 0.0, 0.0, 0.0)
    second = Cartesian4(1.0, 0.0, 0.0, 0.0)
    expected = Cartesian4(2.0, 0.0, 0.0, 0.0)
    assert(Cartesian4.maximumByComponent(first, second, first) ==
           expected
           )

    first.x = 1.0
    second.x = 2.0
    assert(Cartesian4.maximumByComponent(first, second, second) ==
           expected
           )

    first = Cartesian4(2.0, 0.0, 0.0, 0.0)
    second = Cartesian4(1.0, 0.0, 0.0, 0.0)
    expected = Cartesian4(2.0, 0.0, 0.0, 0.0)
    assert(Cartesian4.maximumByComponent(first, second, second) ==
           expected
           )

    first.x = 1.0
    second.x = 2.0
    assert(Cartesian4.maximumByComponent(first, second, second) ==
           expected
           )

    first = Cartesian4(2.0, 0.0, 0.0, 0.0)
    second = Cartesian4(1.0, 0.0, 0.0, 0.0)
    expected = Cartesian4(2.0, 0.0, 0.0, 0.0)
    result = Cartesian4()
    assert(Cartesian4.maximumByComponent(first, second, result) ==
           expected
           )

    second.x = 3.0
    expected.x = 3.0
    assert(Cartesian4.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(0.0, 2.0, 0.0, 0.0)
    second = Cartesian4(0.0, 1.0, 0.0, 0.0)
    expected = Cartesian4(0.0, 2.0, 0.0, 0.0)
    result = Cartesian4()
    assert(Cartesian4.maximumByComponent(first, second, result) ==
           expected
           )

    second.y = 3.0
    expected.y = 3.0
    assert(Cartesian4.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(0.0, 0.0, 2.0, 0.0)
    second = Cartesian4(0.0, 0.0, 1.0, 0.0)
    expected = Cartesian4(0.0, 0.0, 2.0, 0.0)
    result = Cartesian4()
    assert(Cartesian4.maximumByComponent(first, second, result) ==
           expected
           )

    second.z = 3.0
    expected.z = 3.0
    assert(Cartesian4.maximumByComponent(first, second, result) ==
           expected
           )

    first = Cartesian4(0.0, 0.0, 0.0, 2.0)
    second = Cartesian4(0.0, 0.0, 0.0, 1.0)
    expected = Cartesian4(0.0, 0.0, 0.0, 2.0)
    result = Cartesian4()
    assert(Cartesian4.maximumByComponent(first, second, result) ==
           expected
           )

    second.w = 3.0
    expected.w = 3.0
    assert(Cartesian4.maximumByComponent(first, second, result) ==
           expected
           )

    result = Cartesian4()

    value = Cartesian4(-1.0, 0.0, 0.0)
    min = Cartesian4(0.0, 0.0, 0.0)
    max = Cartesian4(1.0, 1.0, 1.0)
    expected = Cartesian4(0.0, 0.0, 0.0)
    assert(Cartesian4.clamp(value, min, max, result) == expected)

    value = Cartesian4(2.0, 0.0, 0.0)
    min = Cartesian4(0.0, 0.0, 0.0)
    max = Cartesian4(1.0, 1.0, 1.0)
    expected = Cartesian4(1.0, 0.0, 0.0)
    assert(Cartesian4.clamp(value, min, max, result) == expected)

    value = Cartesian4(0.0, -1.0, 0.0)
    min = Cartesian4(0.0, 0.0, 0.0)
    max = Cartesian4(1.0, 1.0, 1.0)
    expected = Cartesian4(0.0, 0.0, 0.0)
    assert(Cartesian4.clamp(value, min, max, result) == expected)

    value = Cartesian4(0.0, 2.0, 0.0)
    min = Cartesian4(0.0, 0.0, 0.0)
    max = Cartesian4(1.0, 1.0, 1.0)
    expected = Cartesian4(0.0, 1.0, 0.0)
    assert(Cartesian4.clamp(value, min, max, result) == expected)

    value = Cartesian4(0.0, 0.0, -1.0)
    min = Cartesian4(0.0, 0.0, 0.0)
    max = Cartesian4(1.0, 1.0, 1.0)
    expected = Cartesian4(0.0, 0.0, 0.0)
    assert(Cartesian4.clamp(value, min, max, result) == expected)

    value = Cartesian4(0.0, 0.0, 2.0)
    min = Cartesian4(0.0, 0.0, 0.0)
    max = Cartesian4(1.0, 1.0, 1.0)
    expected = Cartesian4(0.0, 0.0, 1.0)
    assert(Cartesian4.clamp(value, min, max, result) == expected)

    value = Cartesian4(-2.0, 3.0, 4.0)
    min = Cartesian4(0.0, 0.0, 0.0)
    max = Cartesian4(1.0, 1.0, 1.0)
    expected = Cartesian4(0.0, 1.0, 1.0)
    assert(Cartesian4.clamp(value, min, max, result) == expected)

    value = Cartesian4(0.0, 0.0, 0.0)
    min = Cartesian4(1.0, 2.0, 3.0)
    max = Cartesian4(1.0, 2.0, 3.0)
    expected = Cartesian4(1.0, 2.0, 3.0)
    assert(Cartesian4.clamp(value, min, max, result) == expected)

    value = Cartesian4(-1.0, -1.0, -1.0, -1.0)
    min = Cartesian4(0.0, 0.0, 0.0, 0.0)
    max = Cartesian4(1.0, 1.0, 1.0, 1.0)
    expected = Cartesian4(0.0, 0.0, 0.0, 0.0)
    result = Cartesian4()
    returnedResult = Cartesian4.clamp(value, min, max, result)
    assert(returnedResult is result)
    assert(returnedResult == expected)

    value = Cartesian4(-1.0, -1.0, -1.0, -1.0)
    min = Cartesian4(0.0, 0.0, 0.0, 0.0)
    max = Cartesian4(1.0, 1.0, 1.0, 1.0)
    expected = Cartesian4(0.0, 0.0, 0.0, 0.0)
    assert(Cartesian4.clamp(value, min, max, value) == expected)

    Cartesian4.fromElements(-1.0, -1.0, -1.0, -1.0, value)
    assert(Cartesian4.clamp(value, min, max, min) == expected)

    Cartesian4.fromElements(0.0, 0.0, 0.0, 0.0, min)
    assert(Cartesian4.clamp(value, min, max, max) == expected)

    cartesian = Cartesian4(3.0, 4.0, 5.0, 6.0)
    assert(Cartesian4.magnitudeSquared(cartesian) == 86.0)

    cartesian = Cartesian4(3.0, 4.0, 5.0, 6.0)
    assert(Cartesian4.magnitude(cartesian) == math.sqrt(86.0))

    distance = Cartesian4.distance(
        Cartesian4(1.0, 0.0, 0.0, 0.0),
        Cartesian4(2.0, 0.0, 0.0, 0.0)
    )
    assert(distance == 1.0)

    distanceSquared = Cartesian4.distanceSquared(
        Cartesian4(1.0, 0.0, 0.0, 0.0),
        Cartesian4(3.0, 0.0, 0.0, 0.0)
    )
    assert(distanceSquared == 4.0)

    cartesian = Cartesian4(2.0, 0.0, 0.0, 0.0)
    expectedResult = Cartesian4(1.0, 0.0, 0.0, 0.0)
    result = Cartesian4()
    returnedResult = Cartesian4.normalize(cartesian, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    cartesian = Cartesian4(2.0, 0.0, 0.0, 0.0)
    expectedResult = Cartesian4(1.0, 0.0, 0.0, 0.0)
    returnedResult = Cartesian4.normalize(cartesian, cartesian)
    assert(cartesian is returnedResult)
    assert(cartesian == expectedResult)

    left = Cartesian4(2.0, 3.0, 6.0, 8.0)
    right = Cartesian4(4.0, 5.0, 7.0, 9.0)
    result = Cartesian4()
    expectedResult = Cartesian4(8.0, 15.0, 42.0, 72.0)
    returnedResult = Cartesian4.multiplyComponents(left, right, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    left = Cartesian4(2.0, 3.0, 6.0, 8.0)
    right = Cartesian4(4.0, 5.0, 7.0, 9.0)
    expectedResult = Cartesian4(8.0, 15.0, 42.0, 72.0)
    returnedResult = Cartesian4.multiplyComponents(left, right, left)
    assert(left is returnedResult)
    assert(left == expectedResult)

    left = Cartesian4(2.0, 3.0, 6.0, 15.0)
    right = Cartesian4(4.0, 5.0, 8.0, 2.0)
    result = Cartesian4()
    expectedResult = Cartesian4(0.5, 0.6, 0.75, 7.5)
    returnedResult = Cartesian4.divideComponents(left, right, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    left = Cartesian4(2.0, 3.0, 6.0, 15.0)
    right = Cartesian4(4.0, 5.0, 8.0, 2.0)
    expectedResult = Cartesian4(0.5, 0.6, 0.75, 7.5)
    returnedResult = Cartesian4.divideComponents(left, right, left)
    assert(left is returnedResult)
    assert(left == expectedResult)

    left = Cartesian4(2.0, 3.0, 6.0, 8.0)
    right = Cartesian4(4.0, 5.0, 7.0, 9.0)
    expectedResult = 137.0
    result = Cartesian4.dot(left, right)
    assert(result == expectedResult)

    left = Cartesian4(2.0, 3.0, 6.0, 8.0)
    right = Cartesian4(4.0, 5.0, 7.0, 9.0)
    result = Cartesian4()
    expectedResult = Cartesian4(6.0, 8.0, 13.0, 17.0)
    returnedResult = Cartesian4.add(left, right, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    left = Cartesian4(2.0, 3.0, 6.0, 8.0)
    right = Cartesian4(4.0, 5.0, 7.0, 9.0)
    expectedResult = Cartesian4(6.0, 8.0, 13.0, 17.0)
    returnedResult = Cartesian4.add(left, right, left)
    assert(left is returnedResult)
    assert(left == expectedResult)

    left = Cartesian4(2.0, 3.0, 4.0, 8.0)
    right = Cartesian4(1.0, 5.0, 7.0, 9.0)
    result = Cartesian4()
    expectedResult = Cartesian4(1.0, -2.0, -3.0, -1.0)
    returnedResult = Cartesian4.subtract(left, right, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    left = Cartesian4(2.0, 3.0, 4.0, 8.0)
    right = Cartesian4(1.0, 5.0, 7.0, 9.0)
    expectedResult = Cartesian4(1.0, -2.0, -3.0, -1.0)
    returnedResult = Cartesian4.subtract(left, right, left)
    assert(returnedResult is left)
    assert(left == expectedResult)

    cartesian = Cartesian4(1.0, 2.0, 3.0, 4.0)
    result = Cartesian4()
    scalar = 2
    expectedResult = Cartesian4(2.0, 4.0, 6.0, 8.0)
    returnedResult = Cartesian4.multiplyByScalar(
        cartesian,
        scalar,
        result
    )
    assert(result is returnedResult)
    assert(result == expectedResult)

    cartesian = Cartesian4(1.0, 2.0, 3.0, 4.0)
    scalar = 2
    expectedResult = Cartesian4(2.0, 4.0, 6.0, 8.0)
    returnedResult = Cartesian4.multiplyByScalar(
        cartesian,
        scalar,
        cartesian
    )
    assert(cartesian is returnedResult)
    assert(cartesian == expectedResult)

    cartesian = Cartesian4(1.0, 2.0, 3.0, 4.0)
    result = Cartesian4()
    scalar = 2
    expectedResult = Cartesian4(0.5, 1.0, 1.5, 2.0)
    returnedResult = Cartesian4.divideByScalar(cartesian, scalar, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    cartesian = Cartesian4(1.0, 2.0, 3.0, 4.0)
    scalar = 2
    expectedResult = Cartesian4(0.5, 1.0, 1.5, 2.0)
    returnedResult = Cartesian4.divideByScalar(
        cartesian,
        scalar,
        cartesian
    )
    assert(cartesian is returnedResult)
    assert(cartesian == expectedResult)

    cartesian = Cartesian4(1.0, -2.0, -5.0, 4.0)
    result = Cartesian4()
    expectedResult = Cartesian4(-1.0, 2.0, 5.0, -4.0)
    returnedResult = Cartesian4.negate(cartesian, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    cartesian = Cartesian4(1.0, -2.0, -5.0)
    expectedResult = Cartesian4(-1.0, 2.0, 5.0)
    returnedResult = Cartesian4.negate(cartesian, cartesian)
    assert(cartesian is returnedResult)
    assert(cartesian == expectedResult)

    cartesian = Cartesian4(1.0, -2.0, -4.0, -3.0)
    result = Cartesian4()
    expectedResult = Cartesian4(1.0, 2.0, 4.0, 3.0)
    returnedResult = Cartesian4.abs(cartesian, result)
    assert(result is returnedResult)
    assert(result == expectedResult)

    cartesian = Cartesian4(1.0, -2.0, -4.0, -3.0)
    expectedResult = Cartesian4(1.0, 2.0, 4.0, 3.0)
    returnedResult = Cartesian4.abs(cartesian, cartesian)
    assert(cartesian is returnedResult)
    assert(cartesian == expectedResult)

    start = Cartesian4(4.0, 8.0, 10.0, 20.0)
    end = Cartesian4(8.0, 20.0, 20.0, 30.0)
    t = 0.25
    expectedResult = Cartesian4(5.0, 11.0, 12.5, 22.5)
    returnedResult = Cartesian4.lerp(start, end, t, start)
    assert(start is returnedResult)
    assert(start == expectedResult)

    start = Cartesian4(4.0, 8.0, 10.0, 20.0)
    end = Cartesian4(8.0, 20.0, 20.0, 30.0)
    t = 2.0
    result = Cartesian4()
    expectedResult = Cartesian4(12.0, 32.0, 30.0, 40.0)
    result = Cartesian4.lerp(start, end, t, result)
    assert(result == expectedResult)

    start = Cartesian4(4.0, 8.0, 10.0, 20.0)
    end = Cartesian4(8.0, 20.0, 20.0, 30.0)
    t = -1.0
    result = Cartesian4()
    expectedResult = Cartesian4(0.0, -4.0, 0.0, 10.0)
    result = Cartesian4.lerp(start, end, t, result)
    assert(result == expectedResult)

    v = Cartesian4(0.0, 1.0, 2.0, 3.0)
    assert(Cartesian4.mostOrthogonalAxis(v, Cartesian4()) ==
           Cartesian4.UNIT_X()
           )

    v = Cartesian4(1.0, 0.0, 2.0, 3.0)
    assert(Cartesian4.mostOrthogonalAxis(v, Cartesian4()) ==
           Cartesian4.UNIT_Y()
           )

    v = Cartesian4(2.0, 3.0, 0.0, 1.0)
    assert(Cartesian4.mostOrthogonalAxis(v, Cartesian4()) ==
           Cartesian4.UNIT_Z()
           )

    v = Cartesian4(3.0, 2.0, 0.0, 1.0)
    assert(Cartesian4.mostOrthogonalAxis(v, Cartesian4()) ==
           Cartesian4.UNIT_Z()
           )

    v = Cartesian4(1.0, 2.0, 3.0, 0.0)
    assert(Cartesian4.mostOrthogonalAxis(v, Cartesian4()) ==
           Cartesian4.UNIT_W()
           )

    v = Cartesian4(2.0, 3.0, 1.0, 0.0)
    assert(Cartesian4.mostOrthogonalAxis(v, Cartesian4()) ==
           Cartesian4.UNIT_W()
           )

    v = Cartesian4(3.0, 1.0, 2.0, 0.0)
    assert(Cartesian4.mostOrthogonalAxis(v, Cartesian4()) ==
           Cartesian4.UNIT_W()
           )

    v = Cartesian4(3.0, 2.0, 1.0, 0.0)
    assert(Cartesian4.mostOrthogonalAxis(v, Cartesian4()) ==
           Cartesian4.UNIT_W()
           )

    cartesian = Cartesian4(1.0, 2.0, 3.0, 4.0)
    assert(
        Cartesian4.equals(cartesian, Cartesian4(1.0, 2.0, 3.0, 4.0))
        is True)
    assert(
        Cartesian4.equals(cartesian, Cartesian4(2.0, 2.0, 3.0, 4.0))
        is False)
    assert(
        Cartesian4.equals(cartesian, Cartesian4(2.0, 1.0, 3.0, 4.0))
        is False)
    assert(
        Cartesian4.equals(cartesian, Cartesian4(1.0, 2.0, 4.0, 4.0))
        is False)
    assert(
        Cartesian4.equals(cartesian, Cartesian4(1.0, 2.0, 3.0, 5.0))
        is False)
    assert(Cartesian4.equals(cartesian, None) is False)

    cartesian = Cartesian4(1.0, 2.0, 3.0, 4.0)
    assert(
        Cartesian4.equalsEpsilon(cartesian, Cartesian4(1.0, 2.0, 3.0, 4.0), 0.0, 0.0)
        is True)
    assert(
        Cartesian4.equalsEpsilon(cartesian, Cartesian4(1.0, 2.0, 3.0, 4.0), 0, 1.0)
        is True)
    assert(
        Cartesian4.equalsEpsilon(cartesian, Cartesian4(2.0, 2.0, 3.0, 4.0), 0, 1.0)
        is True)
    assert(
        Cartesian4.equalsEpsilon(cartesian, Cartesian4(1.0, 3.0, 3.0, 4.0), 0, 1.0)
        is True)
    assert(
        Cartesian4.equalsEpsilon(cartesian, Cartesian4(1.0, 2.0, 4.0, 4.0), 0, 1.0)
        is True)
    assert(
        Cartesian4.equalsEpsilon(cartesian, Cartesian4(1.0, 2.0, 3.0, 5.0), 0, 1.0)
        is True)
    assert(
        Cartesian4.equalsEpsilon(cartesian,
                                 Cartesian4(2.0, 2.0, 3.0, 4.0),
                                 0,
                                 EPSILON6
                                 )
        is False)
    assert(
        Cartesian4.equalsEpsilon(cartesian,
                                 Cartesian4(1.0, 3.0, 3.0, 4.0),
                                 0,
                                 EPSILON6
                                 )
        is False)
    assert(
        Cartesian4.equalsEpsilon(cartesian,
                                 Cartesian4(1.0, 2.0, 4.0, 4.0),
                                 0,
                                 EPSILON6
                                 )
        is False)
    assert(
        Cartesian4.equalsEpsilon(cartesian,
                                 Cartesian4(1.0, 2.0, 3.0, 5.0),
                                 0,
                                 EPSILON6
                                 )
        is False)
    assert(Cartesian4.equalsEpsilon(cartesian, None, 0, 1) is False)

    cartesian = Cartesian4(3000000.0, 4000000.0, 5000000.0, 6000000.0)
    assert(
        Cartesian4.equalsEpsilon(cartesian,
                                 Cartesian4(3000000.0, 4000000.0, 5000000.0, 6000000.0),
                                 0,
                                 0.0
                                 )
        is True)
    assert(
        Cartesian4.equalsEpsilon(cartesian,
                                 Cartesian4(3000000.2, 4000000.0, 5000000.0, 6000000.0),
                                 EPSILON7
                                 )
        is True)
    assert(
        Cartesian4.equalsEpsilon(cartesian,
                                 Cartesian4(3000000.0, 4000000.2, 5000000.0, 6000000.0),
                                 EPSILON7
                                 )
        is True)
    assert(
        Cartesian4.equalsEpsilon(cartesian,
                                 Cartesian4(3000000.0, 4000000.0, 5000000.2, 6000000.0),
                                 EPSILON7
                                 )
        is True)
    assert(
        Cartesian4.equalsEpsilon(cartesian,
                                 Cartesian4(3000000.0, 4000000.0, 5000000.0, 6000000.2),
                                 EPSILON7
                                 )
        is True)
    assert(
        Cartesian4.equalsEpsilon(cartesian,
                                 Cartesian4(3000000.2, 4000000.2, 5000000.2, 6000000.2),
                                 EPSILON7
                                 )
        is True)
    assert(
        Cartesian4.equalsEpsilon(cartesian,
                                 Cartesian4(3000000.2, 4000000.2, 5000000.2, 6000000.2),
                                 EPSILON9
                                 )
        is False)
    assert(Cartesian4.equalsEpsilon(cartesian, None, 1) is False)
    assert(Cartesian4.equalsEpsilon(None, cartesian, 1) is False)

    cartesian = Cartesian4(1.123, 2.345, 6.789, 6.123)
    assert(str(cartesian) == "[1.123, 2.345, 6.789, 6.123]")

    assert(Cartesian4.clone() is None)
    assert(Cartesian4.clone(Cartesian4(1.0, 2.0, 3.0, 4.0)) == Cartesian4(1.0, 2.0, 3.0, 4.0))

    cartesian = Cartesian4.ONE()
    assert(cartesian.x == 1.0)
    assert(cartesian.y == 1.0)
    assert(cartesian.z == 1.0)
    assert(cartesian.w == 1.0)

    cartesian = Cartesian4.ZERO()
    assert(cartesian.x == 0.0)
    assert(cartesian.y == 0.0)
    assert(cartesian.z == 0.0)
    assert(cartesian.w == 0.0)
