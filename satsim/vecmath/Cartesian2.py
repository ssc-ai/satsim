import math


class Cartesian2:
    """ A 2D Cartesian point. """

    def __init__(self, x=0.0, y=0.0):
        """ Constructor.

        Args:
            x: `float`, The X component. default: 0
            y: `float`, The Y component. default: 0
        """
        self.x = x
        self.y = y

    def __str__(self):
        return str([self.x, self.y])

    def __eq__(self, other):
        return (
            self is other or
            (self.x == other.x and self.y == other.y)
        )

    @staticmethod
    def fromElements(x, y, result=None):
        """ Creates a Cartesian2 instance from x and y coordinates.

        Args:
            x: `float`, The X component. default: 0
            y: `float`, The Y component. default: 0
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter or a new Cartesian2 instance if one was not provided.
        """
        if result is None:
            return Cartesian2(x, y)

        result.x = x
        result.y = y
        return result

    @staticmethod
    def fromArray(array, startingIndex=0, result=None):
        """ Creates a Cartesian2 from two consecutive elements in an array.

        Args:
            array: `list`, The array whose two consecutive elements correspond to the x and y components, respectively.
            startingIndex: `int`, The offset into the array of the first element, which corresponds to the x component.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter or a new Cartesian2 instance if one was not provided.
        """
        if result is None:
            result = Cartesian2()

        result.x = array[startingIndex]
        result.y = array[startingIndex + 1]
        return result

    @staticmethod
    def clone(cartesian=None, result=None):
        """ Duplicates a Cartesian2 instance.

        Args:
            cartesian: `Cartesian2`, The cartesian to duplicate.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter or a new Cartesian2 instance if one was not provided.
        """
        if cartesian is None:
            return None

        if result is None:
            return Cartesian2(cartesian.x, cartesian.y)

        result.x = cartesian.x
        result.y = cartesian.y
        return result

    @staticmethod
    def maximumComponent(cartesian):
        """ Computes the value of the maximum component for the supplied Cartesian.

        Args:
            cartesian: `Cartesian2`, The cartesian to use.

        Returns:
            A `number`, The value of the maximum component.
        """
        return max(cartesian.x, cartesian.y)

    @staticmethod
    def minimumComponent(cartesian):
        """ Computes the value of the minimum component for the supplied Cartesian.

        Args:
            cartesian: `Cartesian2`, The cartesian to use.

        Returns:
            A `number`, The value of the minimum component.
        """
        return min(cartesian.x, cartesian.y)

    @staticmethod
    def minimumByComponent(first, second, result):
        """ Computes the value of the minimum component for the supplied Cartesian.

        Args:
            first: `Cartesian2`, The cartesian to compare.
            second: `Cartesian2`, The cartesian to compare.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `number`, The value of the minimum component.
        """
        result.x = min(first.x, second.x)
        result.y = min(first.y, second.y)
        return result

    @staticmethod
    def maximumByComponent(first, second, result):
        """ Computes the value of the maximum component for the supplied Cartesian.

        Args:
            first: `Cartesian2`, The cartesian to compare.
            second: `Cartesian2`, The cartesian to compare.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `number`, The value of the maximum component.
        """
        result.x = max(first.x, second.x)
        result.y = max(first.y, second.y)
        return result

    @staticmethod
    def clamp(value, min_val, max_val, result):
        """ Computes the value of the minimum component for the supplied Cartesian.

        Args:
            value: `Cartesian2`, The value to clamp.
            min_val: `Cartesian2`, The minimum bound.
            max_val: `Cartesian2`, The maximum bound.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `number`, The clamped value such that min <= result <= max.
        """
        x = max(min(value.x, max_val.x), min_val.x)
        y = max(min(value.y, max_val.y), min_val.y)

        result.x = x
        result.y = y

        return result

    @staticmethod
    def fromCartesian3(cartesian, result=None):
        """ Creates a Cartesian2 instance from an existing Cartesian3.  This simply takes the
        x and y properties of the Cartesian3 and drops z.

        Args:
            cartesian: `Cartesian3`, The cartesian to create from.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter or a new Cartesian2 instance if one was not provided.
        """
        return Cartesian2.clone(cartesian, result)

    @staticmethod
    def fromCartesian4(cartesian, result=None):
        """ Creates a Cartesian2 instance from an existing Cartesian3.  This simply takes the
        x and y properties of the Cartesian3 and drops z and w.

        Args:
            cartesian: `Cartesian4`, The cartesian to create from.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter or a new Cartesian2 instance if one was not provided.
        """
        return Cartesian2.clone(cartesian, result=None)

    @staticmethod
    def magnitudeSquared(cartesian):
        """ Computes the provided Cartesian's squared magnitude.

        Args:
            cartesian: `Cartesian2`, he Cartesian instance whose squared magnitude is to be computed.

        Returns:
            A `float`, The squared magnitude.
        """
        return cartesian.x * cartesian.x + cartesian.y * cartesian.y

    @staticmethod
    def magnitude(cartesian):
        """ Computes the provided Cartesian's magnitude (length).

        Args:
            cartesian: `Cartesian2`, he Cartesian instance whose magnitude is to be computed.

        Returns:
            A `float`, The magnitude.
        """
        return math.sqrt(Cartesian2.magnitudeSquared(cartesian))

    @staticmethod
    def distance(left, right):
        """ Computes the distance between two points.

        Args:
            left, `Cartesian2`, The first point to compute the distance from.
            right, `Cartesian2`, The second point to compute the distance to.

        Returns:
            A `float`, The distance between two points.
        """
        Cartesian2.subtract(left, right, _distanceScratch)
        return Cartesian2.magnitude(_distanceScratch)

    @staticmethod
    def distanceSquared(left, right):
        """ Computes the squared distance between two points.  Comparing squared distances
        using this function is more efficient than comparing distances.

        Args:
            left, `Cartesian2`, The first point to compute the distance from.
            right, `Cartesian2`, The second point to compute the distance to.

        Returns:
            A `float`, The distance between two points.
        """
        Cartesian2.subtract(left, right, _distanceScratch)
        return Cartesian2.magnitudeSquared(_distanceScratch)

    @staticmethod
    def normalize(cartesian, result):
        """ Computes the normalized form of the supplied Cartesian.

        Args:
            cartesian, `Cartesian2`, The Cartesian to be normalized.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        magnitude = Cartesian2.magnitude(cartesian)
        result.x = cartesian.x / magnitude
        result.y = cartesian.y / magnitude
        return result

    @staticmethod
    def dot(left, right):
        """ Computes the dot (scalar) product of two Cartesians.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right, `Cartesian2`, The second Cartesian.

        Returns:
            A `float`, The dot product.
        """
        return left.x * right.x + left.y * right.y

    @staticmethod
    def cross(left, right):
        """ Computes the magnitude of the cross product that would result from implicitly setting the Z coordinate of the input vectors to 0.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right, `Cartesian2`, The second Cartesian.

        Returns:
            A `float`, The cross product.
        """
        return left.x * right.y - left.y * right.x

    @staticmethod
    def multiplyComponents(left, right, result):
        """ Computes the componentwise product of two Cartesians.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right, `Cartesian2`, The second Cartesian.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        result.x = left.x * right.x
        result.y = left.y * right.y
        return result

    @staticmethod
    def divideComponents(left, right, result):
        """ Computes the component wise quotient of two Cartesians.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right, `Cartesian2`, The second Cartesian.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `float`, The modified result parameter.
        """
        result.x = left.x / right.x
        result.y = left.y / right.y
        return result

    @staticmethod
    def add(left, right, result):
        """ Computes the component wise sum of two Cartesians.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right, `Cartesian2`, The second Cartesian.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `float`, The modified result parameter.
        """
        result.x = left.x + right.x
        result.y = left.y + right.y
        return result

    @staticmethod
    def subtract(left, right, result):
        """ Computes the component wise difference of two Cartesians.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right, `Cartesian2`, The second Cartesian.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        result.x = left.x - right.x
        result.y = left.y - right.y
        return result

    @staticmethod
    def multiplyByScalar(cartesian, scalar, result):
        """ Multiplies the provided Cartesian componentwise by the provided scalar.

        Args:
            cartesian, `Cartesian2`, The Cartesian to be scaled.
            scalar, `float`, The scalar to multiply with.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        result.x = cartesian.x * scalar
        result.y = cartesian.y * scalar
        return result

    @staticmethod
    def divideByScalar(cartesian, scalar, result):
        """ Divides the provided Cartesian componentwise by the provided scalar.

        Args:
            cartesian, `Cartesian2`, The Cartesian to be divided.
            scalar, `float`, The scalar to divide by.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        result.x = cartesian.x / scalar
        result.y = cartesian.y / scalar
        return result

    @staticmethod
    def negate(cartesian, result):
        """ Negates the provided Cartesian.

        Args:
            cartesian, `Cartesian2`, The Cartesian to be negated.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        result.x = -cartesian.x
        result.y = -cartesian.y
        return result

    @staticmethod
    def abs(cartesian, result):
        """ Computes the absolute value of the provided Cartesian.

        Args:
            cartesian, `Cartesian2`, The Cartesian whose absolute value is to be computed.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        result.x = abs(cartesian.x)
        result.y = abs(cartesian.y)
        return result

    @staticmethod
    def lerp(start, end, t, result):
        """ Computes the linear interpolation or extrapolation at t using the provided cartesians.

        Args:
            start, `Cartesian2`, The value corresponding to t at 0.0.
            end, `Cartesian2`, The value corresponding to t at 1.0.
            t, `Cartesian2`, The point along t at which to interpolate.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        Cartesian2.multiplyByScalar(end, t, _lerpScratch)
        result = Cartesian2.multiplyByScalar(start, 1.0 - t, result)
        return Cartesian2.add(_lerpScratch, result, result)

    @staticmethod
    def angleBetween(left, right):
        """ Returns the angle, in radians, between the provided Cartesians.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right, `Cartesian2`, The second Cartesian.

        Returns:
            A `float`, The angle between the Cartesians.
        """
        Cartesian2.normalize(left, _angleBetweenScratch)
        Cartesian2.normalize(right, _angleBetweenScratch2)
        return math.acos(max(min(Cartesian2.dot(_angleBetweenScratch, _angleBetweenScratch2), 1.0), -1.0))

    @staticmethod
    def mostOrthogonalAxis(cartesian, result):
        """ Returns the angle, in radians, between the provided Cartesians.

        Args:
            cartesian, `Cartesian2`, The Cartesian on which to find the most orthogonal axis.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The most orthogonal axis.
        """
        f = Cartesian2.normalize(cartesian, _mostOrthogonalAxisScratch)
        Cartesian2.abs(f, f)
        if f.x <= f.y:
            result = Cartesian2.clone(Cartesian2.UNIT_X(), result)
        else:
            result = Cartesian2.clone(Cartesian2.UNIT_Y(), result)
        return result

    @staticmethod
    def equals(left, right):
        """ Compares the provided Cartesians componentwise and return `True` if equal,
        `False` otherwise.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right: `Cartesian2`, The second Cartesian.

        Returns:
            A `boolean`, `True` if equal, `False` otherwise.
        """
        return (
            left is right or
            (left is not None and right is not None and
                left.x == right.x and left.y == right.y)
        )

    @staticmethod
    def equalsEpsilon(left, right, relativeEpsilon=0.0, absoluteEpsilon=0.0):
        """ Compares the provided Cartesians componentwise and return `True`
        if they pass an absolute or relative tolerance test, `False` otherwise.

        Args:
            left, `Cartesian2`, The first Cartesian.
            right: `Cartesian2`, The second Cartesian.
            relativeEpsilon, `float`, The relative epsilon tolerance to use for equality testing. default=0
            absoluteEpsilon, `float`, The absolute epsilon tolerance to use for equality testing. default=0

        Returns:
            A `boolean`, `True` if they pass an absolute or relative tolerance test, `False` otherwise.
        """
        return (
            left is right or
            (left is not None and
                right is not None and
                math.isclose(left.x, right.x, rel_tol=relativeEpsilon, abs_tol=absoluteEpsilon) and
                math.isclose(left.y, right.y, rel_tol=relativeEpsilon, abs_tol=absoluteEpsilon))
        )

    @staticmethod
    def ZERO():
        """ A Cartesian2 instance initialized to (0.0, 0.0)."""
        return Cartesian2(0,0)

    @staticmethod
    def ONE():
        """ A Cartesian2 instance initialized to (1.0, 1.0)."""
        return Cartesian2(1,1)

    @staticmethod
    def UNIT_X():
        """ A Cartesian2 instance initialized to (1.0, 0.0)."""
        return Cartesian2(1,0)

    @staticmethod
    def UNIT_Y():
        """ A Cartesian2 instance initialized to (0.0, 1.0)."""
        return Cartesian2(0,1)


_angleBetweenScratch = Cartesian2()
_angleBetweenScratch2 = Cartesian2()
_lerpScratch = Cartesian2()
_mostOrthogonalAxisScratch = Cartesian2()
_distanceScratch = Cartesian2()
