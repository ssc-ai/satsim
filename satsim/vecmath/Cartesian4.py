import math


class Cartesian4:
    """ A 4D Cartesian point. """

    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
        """ Constructor.

        Args:
            x: `float`, The X component. default: 0
            y: `float`, The Y component. default: 0
            z: `float`, The Z component. default: 0
            w: `float`, The W component. default: 0
        """
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __str__(self):
        return str([self.x, self.y, self.z, self.w])

    def __eq__(self, other):
        return (
            self is other or
            (self.x == other.x and
                self.y == other.y and
                self.z == other.z and
                self.w == other.w)
        )

    @staticmethod
    def fromElements(x, y, z, w, result=None):
        """ Creates a Cartesian4 instance from x and y coordinates.

        Args:
            x: `float`, The X component. default: 0
            y: `float`, The Y component. default: 0
            z: `float`, The Z component. default: 0
            w: `float`, The W component. default: 0
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `Cartesian4`, The modified result parameter or a new Cartesian4 instance if one was not provided.
        """
        if result is None:
            return Cartesian4(x, y, z, w)

        result.x = x
        result.y = y
        result.z = z
        result.w = w
        return result

    @staticmethod
    def fromArray(array, startingIndex=0, result=None):
        """ Creates a Cartesian4 from four consecutive elements in an array.

        Args:
            array: `list`, The array whose four consecutive elements correspond to the x y, z, and w components, respectively.
            startingIndex: `int`, The offset into the array of the first element, which corresponds to the x component.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `Cartesian4`, The modified result parameter or a new Cartesian4 instance if one was not provided.
        """
        if result is None:
            result = Cartesian4()

        result.x = array[startingIndex]
        result.y = array[startingIndex + 1]
        result.z = array[startingIndex + 2]
        result.w = array[startingIndex + 3]
        return result

    @staticmethod
    def clone(cartesian=None, result=None):
        """ Duplicates a Cartesian4 instance.

        Args:
            cartesian: `Cartesian4`, The cartesian to duplicate.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `Cartesian4`, The modified result parameter or a new Cartesian4 instance if one was not provided.
        """
        if cartesian is None:
            return None

        if result is None:
            return Cartesian4(cartesian.x, cartesian.y, cartesian.z, cartesian.w)

        result.x = cartesian.x
        result.y = cartesian.y
        result.z = cartesian.z
        result.w = cartesian.w
        return result

    @staticmethod
    def maximumComponent(cartesian):
        """ Computes the value of the maximum component for the supplied Cartesian.

        Args:
            cartesian: `Cartesian4`, The cartesian to use.

        Returns:
            A `number`, The value of the maximum component.
        """
        return max(cartesian.x, cartesian.y, cartesian.z, cartesian.w)

    @staticmethod
    def minimumComponent(cartesian):
        """ Computes the value of the minimum component for the supplied Cartesian.

        Args:
            cartesian: `Cartesian4`, The cartesian to use.

        Returns:
            A `number`, The value of the minimum component.
        """
        return min(cartesian.x, cartesian.y, cartesian.z, cartesian.w)

    @staticmethod
    def minimumByComponent(first, second, result):
        """ Computes the value of the minimum component for the supplied Cartesian.

        Args:
            first: `Cartesian4`, The cartesian to compare.
            second: `Cartesian4`, The cartesian to compare.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `number`, The value of the minimum component.
        """
        result.x = min(first.x, second.x)
        result.y = min(first.y, second.y)
        result.z = min(first.z, second.z)
        result.w = min(first.w, second.w)
        return result

    @staticmethod
    def maximumByComponent(first, second, result):
        """ Computes the value of the maximum component for the supplied Cartesian.

        Args:
            first: `Cartesian4`, The cartesian to compare.
            second: `Cartesian4`, The cartesian to compare.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `number`, The value of the maximum component.
        """
        result.x = max(first.x, second.x)
        result.y = max(first.y, second.y)
        result.z = max(first.z, second.z)
        result.w = max(first.w, second.w)
        return result

    @staticmethod
    def clamp(value, min_val, max_val, result):
        """ Computes the value of the minimum component for the supplied Cartesian.

        Args:
            value: `Cartesian4`, The value to clamp.
            min_val: `Cartesian4`, The minimum bound.
            max_val: `Cartesian4`, The maximum bound.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `number`, The clamped value such that min <= result <= max.
        """
        x = max(min(value.x, max_val.x), min_val.x)
        y = max(min(value.y, max_val.y), min_val.y)
        z = max(min(value.z, max_val.z), min_val.z)
        w = max(min(value.w, max_val.w), min_val.w)

        result.x = x
        result.y = y
        result.z = z
        result.w = w

        return result

    @staticmethod
    def magnitudeSquared(cartesian):
        """ Computes the provided Cartesian's squared magnitude.

        Args:
            cartesian: `Cartesian4`, he Cartesian instance whose squared magnitude is to be computed.

        Returns:
            A `float`, The squared magnitude.
        """
        return cartesian.x * cartesian.x + cartesian.y * cartesian.y + cartesian.z * cartesian.z + cartesian.w * cartesian.w

    @staticmethod
    def magnitude(cartesian):
        """ Computes the provided Cartesian's magnitude (length).

        Args:
            cartesian: `Cartesian4`, he Cartesian instance whose magnitude is to be computed.

        Returns:
            A `float`, The magnitude.
        """
        return math.sqrt(Cartesian4.magnitudeSquared(cartesian))

    @staticmethod
    def distance(left, right):
        """ Computes the distance between two points.

        Args:
            left, `Cartesian4`, The first point to compute the distance from.
            right, `Cartesian4`, The second point to compute the distance to.

        Returns:
            A `float`, The distance between two points.
        """
        Cartesian4.subtract(left, right, _distanceScratch)
        return Cartesian4.magnitude(_distanceScratch)

    @staticmethod
    def distanceSquared(left, right):
        """ Computes the squared distance between two points.  Comparing squared distances
        using this function is more efficient than comparing distances.

        Args:
            left, `Cartesian4`, The first point to compute the distance from.
            right, `Cartesian4`, The second point to compute the distance to.

        Returns:
            A `float`, The distance between two points.
        """
        Cartesian4.subtract(left, right, _distanceScratch)
        return Cartesian4.magnitudeSquared(_distanceScratch)

    @staticmethod
    def normalize(cartesian, result):
        """ Computes the normalized form of the supplied Cartesian.

        Args:
            cartesian, `Cartesian4`, The Cartesian to be normalized.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `Cartesian4`, The modified result parameter.
        """
        magnitude = Cartesian4.magnitude(cartesian)
        result.x = cartesian.x / magnitude
        result.y = cartesian.y / magnitude
        result.z = cartesian.z / magnitude
        result.w = cartesian.w / magnitude
        return result

    @staticmethod
    def dot(left, right):
        """ Computes the dot (scalar) product of two Cartesians.

        Args:
            left, `Cartesian4`, The first Cartesian.
            right, `Cartesian4`, The second Cartesian.

        Returns:
            A `float`, The dot product.
        """
        return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w

    @staticmethod
    def multiplyComponents(left, right, result):
        """ Computes the componentwise product of two Cartesians.

        Args:
            left, `Cartesian4`, The first Cartesian.
            right, `Cartesian4`, The second Cartesian.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `Cartesian4`, The modified result parameter.
        """
        result.x = left.x * right.x
        result.y = left.y * right.y
        result.z = left.z * right.z
        result.w = left.w * right.w
        return result

    @staticmethod
    def divideComponents(left, right, result):
        """ Computes the component wise quotient of two Cartesians.

        Args:
            left, `Cartesian4`, The first Cartesian.
            right, `Cartesian4`, The second Cartesian.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `float`, The modified result parameter.
        """
        result.x = left.x / right.x
        result.y = left.y / right.y
        result.z = left.z / right.z
        result.w = left.w / right.w
        return result

    @staticmethod
    def add(left, right, result):
        """ Computes the component wise sum of two Cartesians.

        Args:
            left, `Cartesian4`, The first Cartesian.
            right, `Cartesian4`, The second Cartesian.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `float`, The modified result parameter.
        """
        result.x = left.x + right.x
        result.y = left.y + right.y
        result.z = left.z + right.z
        result.w = left.w + right.w
        return result

    @staticmethod
    def subtract(left, right, result):
        """ Computes the component wise difference of two Cartesians.

        Args:
            left, `Cartesian4`, The first Cartesian.
            right, `Cartesian4`, The second Cartesian.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `Cartesian4`, The modified result parameter.
        """
        result.x = left.x - right.x
        result.y = left.y - right.y
        result.z = left.z - right.z
        result.w = left.w - right.w
        return result

    @staticmethod
    def multiplyByScalar(cartesian, scalar, result):
        """ Multiplies the provided Cartesian componentwise by the provided scalar.

        Args:
            cartesian, `Cartesian4`, The Cartesian to be scaled.
            scalar, `float`, The scalar to multiply with.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `Cartesian4`, The modified result parameter.
        """
        result.x = cartesian.x * scalar
        result.y = cartesian.y * scalar
        result.z = cartesian.z * scalar
        result.w = cartesian.w * scalar
        return result

    @staticmethod
    def divideByScalar(cartesian, scalar, result):
        """ Divides the provided Cartesian componentwise by the provided scalar.

        Args:
            cartesian, `Cartesian4`, The Cartesian to be divided.
            scalar, `float`, The scalar to divide by.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `Cartesian4`, The modified result parameter.
        """
        result.x = cartesian.x / scalar
        result.y = cartesian.y / scalar
        result.z = cartesian.z / scalar
        result.w = cartesian.w / scalar
        return result

    @staticmethod
    def negate(cartesian, result):
        """ Negates the provided Cartesian.

        Args:
            cartesian, `Cartesian4`, The Cartesian to be negated.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `Cartesian4`, The modified result parameter.
        """
        result.x = -cartesian.x
        result.y = -cartesian.y
        result.z = -cartesian.z
        result.w = -cartesian.w
        return result

    @staticmethod
    def abs(cartesian, result):
        """ Computes the absolute value of the provided Cartesian.

        Args:
            cartesian, `Cartesian4`, The Cartesian whose absolute value is to be computed.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `Cartesian4`, The modified result parameter.
        """
        result.x = abs(cartesian.x)
        result.y = abs(cartesian.y)
        result.z = abs(cartesian.z)
        result.w = abs(cartesian.w)
        return result

    @staticmethod
    def lerp(start, end, t, result):
        """ Computes the linear interpolation or extrapolation at t using the provided cartesians.

        Args:
            start, `Cartesian4`, The value corresponding to t at 0.0.
            end, `Cartesian4`, The value corresponding to t at 1.0.
            t, `Cartesian4`, The point along t at which to interpolate.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `Cartesian4`, The modified result parameter.
        """
        Cartesian4.multiplyByScalar(end, t, _lerpScratch)
        result = Cartesian4.multiplyByScalar(start, 1.0 - t, result)
        return Cartesian4.add(_lerpScratch, result, result)

    @staticmethod
    def mostOrthogonalAxis(cartesian, result):
        """ Returns the angle, in radians, between the provided Cartesians.

        Args:
            cartesian, `Cartesian4`, The Cartesian on which to find the most orthogonal axis.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `Cartesian4`, The most orthogonal axis.
        """
        f = Cartesian4.normalize(cartesian, _mostOrthogonalAxisScratch)
        Cartesian4.abs(f, f)

        if f.x <= f.y:
            if f.x <= f.z:
                if f.x <= f.w:
                    result = Cartesian4.clone(Cartesian4.UNIT_X(), result)
                else:
                    result = Cartesian4.clone(Cartesian4.UNIT_W(), result)
            elif f.z <= f.w:
                result = Cartesian4.clone(Cartesian4.UNIT_Z(), result)
            else:
                result = Cartesian4.clone(Cartesian4.UNIT_W(), result)
        elif f.y <= f.z:
            if f.y <= f.w:
                result = Cartesian4.clone(Cartesian4.UNIT_Y(), result)
            else:
                result = Cartesian4.clone(Cartesian4.UNIT_W(), result)
        elif f.z <= f.w:
            result = Cartesian4.clone(Cartesian4.UNIT_Z(), result)
        else:
            result = Cartesian4.clone(Cartesian4.UNIT_W(), result)

        return result

    @staticmethod
    def equals(left, right):
        """ Compares the provided Cartesians componentwise and return `True` if equal,
        `False` otherwise

        Args:
            left, `Cartesian4`, The first Cartesian.
            right: `Cartesian4`, The second Cartesian.

        Returns:
            A `boolean`, `True` if equal, `False` otherwise.
        """
        return (
            left is right or
            (left is not None and right is not None and
                left.x == right.x and left.y == right.y and left.z == right.z and left.w == right.w)
        )

    @staticmethod
    def equalsEpsilon(left, right, relativeEpsilon=0.0, absoluteEpsilon=0.0):
        """ Compares the provided Cartesians componentwise and return `True`
        if they pass an absolute or relative tolerance test, `False` otherwise.

        Args:
            left, `Cartesian4`, The first Cartesian.
            right: `Cartesian4`, The second Cartesian.
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
                math.isclose(left.y, right.y, rel_tol=relativeEpsilon, abs_tol=absoluteEpsilon) and
                math.isclose(left.z, right.z, rel_tol=relativeEpsilon, abs_tol=absoluteEpsilon) and
                math.isclose(left.w, right.w, rel_tol=relativeEpsilon, abs_tol=absoluteEpsilon))
        )

    @staticmethod
    def ZERO():
        """ A Cartesian4 instance initialized to (0.0, 0.0, 0.0, 0.0)."""
        return Cartesian4(0,0,0,0)

    @staticmethod
    def ONE():
        """ A Cartesian4 instance initialized to (1.0, 1.0, 1.0, 1.0)."""
        return Cartesian4(1,1,1,1)

    @staticmethod
    def UNIT_X():
        """ A Cartesian4 instance initialized to (1.0, 0.0, 0.0, 0.0)."""
        return Cartesian4(1,0,0,0)

    @staticmethod
    def UNIT_Y():
        """ A Cartesian4 instance initialized to (0.0, 1.0, 0.0, 0.0)."""
        return Cartesian4(0,1,0,0)

    @staticmethod
    def UNIT_Z():
        """ A Cartesian4 instance initialized to (0.0, 0.0, 1.0, 0.0)."""
        return Cartesian4(0,0,1,0)

    @staticmethod
    def UNIT_W():
        """ A Cartesian4 instance initialized to (0.0, 0.0, 0.0, 1.0)."""
        return Cartesian4(0,0,0,1)


_lerpScratch = Cartesian4()
_mostOrthogonalAxisScratch = Cartesian4()
_distanceScratch = Cartesian4()
