import math

from satsim.math.const import EPSILON6
from satsim.vecmath import Cartesian3, Matrix3


class Quaternion:
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
    def fromAxisAngle(axis, angle, result=None):
        """ Computes a quaternion representing a rotation around an axis.

        Args:
            axis: `Cartesian3`, The axis of rotation.
            angle: `float`, The angle in radians to rotate around the axis.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter or a new Quaternion instance if one was not provide
        """
        halfAngle = angle / 2.0
        s = math.sin(halfAngle)
        Cartesian3.normalize(axis, _fromAxisAngleScratch)

        x = _fromAxisAngleScratch.x * s
        y = _fromAxisAngleScratch.y * s
        z = _fromAxisAngleScratch.z * s
        w = math.cos(halfAngle)
        if result is None:
            return Quaternion(x, y, z, w)
        result.x = x
        result.y = y
        result.z = z
        result.w = w
        return result

    @staticmethod
    def fromRotationMatrix(matrix, result=None):
        """ Computes a Quaternion from the provided Matrix3 instance.

        Args:
            axis: `Cartesian3`, The axis of rotation.
            angle: `float`, The angle in radians to rotate around the axis.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter or a new Quaternion instance if one was not provide
        """
        m00 = matrix[Matrix3.COLUMN0ROW0]
        m11 = matrix[Matrix3.COLUMN1ROW1]
        m22 = matrix[Matrix3.COLUMN2ROW2]
        trace = m00 + m11 + m22

        if trace > 0.0:
            root = math.sqrt(trace + 1.0)
            w = 0.5 * root
            root = 0.5 / root

            x = (matrix[Matrix3.COLUMN1ROW2] - matrix[Matrix3.COLUMN2ROW1]) * root
            y = (matrix[Matrix3.COLUMN2ROW0] - matrix[Matrix3.COLUMN0ROW2]) * root
            z = (matrix[Matrix3.COLUMN0ROW1] - matrix[Matrix3.COLUMN1ROW0]) * root
        else:
            next = [1, 2, 0]

            i = 0
            if m11 > m00:
                i = 1
            if m22 > m00 and m22 > m11:
                i = 2
            j = next[i]
            k = next[j]

            root = math.sqrt(matrix[Matrix3.getElementIndex(i, i)] - matrix[Matrix3.getElementIndex(j, j)] - matrix[Matrix3.getElementIndex(k, k)] + 1.0)

            quat = [0, 0, 0]
            quat[i] = 0.5 * root
            root = 0.5 / root
            w = (matrix[Matrix3.getElementIndex(k, j)] - matrix[Matrix3.getElementIndex(j, k)]) * root
            quat[j] = (matrix[Matrix3.getElementIndex(j, i)] + matrix[Matrix3.getElementIndex(i, j)]) * root
            quat[k] = (matrix[Matrix3.getElementIndex(k, i)] + matrix[Matrix3.getElementIndex(i, k)]) * root

            x = -quat[0]
            y = -quat[1]
            z = -quat[2]

        if result is None:
            return Quaternion(x, y, z, w)

        result.x = x
        result.y = y
        result.z = z
        result.w = w
        return result

    @staticmethod
    def fromHeadingPitchRoll(headingPitchRoll, result=None):
        """ Computes a rotation from the given heading, pitch and roll angles. Heading is the rotation about the
        negative z axis. Pitch is the rotation about the negative y axis. Roll is the rotation about
        the positive x axis.

        Args:
            headingPitchRoll: `dict`, The rotation expressed as a heading, pitch and roll.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter or a new Quaternion instance if one was not provide
        """
        if result is None:
            result = Quaternion()

        Quaternion.fromAxisAngle(
            Cartesian3.UNIT_X(),
            headingPitchRoll['roll'],
            _scratchRollQuaternion)
        Quaternion.fromAxisAngle(
            Cartesian3.UNIT_Y(),
            -headingPitchRoll['pitch'],
            _scratchPitchQuaternion)
        Quaternion.multiply(
            _scratchPitchQuaternion,
            _scratchRollQuaternion,
            result)
        Quaternion.fromAxisAngle(
            Cartesian3.UNIT_Z(),
            -headingPitchRoll['heading'],
            _scratchHeadingQuaternion)
        return Quaternion.multiply(_scratchHeadingQuaternion, result, result)

    @staticmethod
    def toHeadingPitchRoll(quaternion, result={}):
        """ Computes the heading, pitch and roll from a quaternion.

        Args:
            quaternion: `Quaternion`, The quaternion from which to retrieve heading, pitch, and roll, all expressed in radians.
            result: `dict`, The object onto which to store the result.

        Returns:
            A `dict`, The result as a dict with heading, pitch, and roll.
        """
        test = 2 * (quaternion.w * quaternion.y - quaternion.z * quaternion.x)
        denominatorRoll = 1 - 2 * (quaternion.x * quaternion.x + quaternion.y * quaternion.y)
        numeratorRoll = 2 * (quaternion.w * quaternion.x + quaternion.y * quaternion.z)
        denominatorHeading = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        numeratorHeading = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        result['heading'] = -math.atan2(numeratorHeading, denominatorHeading)
        result['roll'] = math.atan2(numeratorRoll, denominatorRoll)
        result['pitch'] = -math.asin(max(min(test, 1.0), -1.0))
        return result

    @staticmethod
    def clone(quaternion=None, result=None):
        """ Duplicates a Quaternion instance.

        Args:
            quaternion: `Quaternion`, The quaternion to duplicate.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter or a new Quaternion instance if one was not provided.
        """
        if quaternion is None:
            return None

        if result is None:
            return Quaternion(quaternion.x, quaternion.y, quaternion.z, quaternion.w)

        result.x = quaternion.x
        result.y = quaternion.y
        result.z = quaternion.z
        result.w = quaternion.w
        return result

    @staticmethod
    def conjugate(quaternion, result):
        """ Computes the conjugate of the provided quaternion.

        Args:
            quaternion: `Quaternion`, The quaternion to conjugate.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        result.x = -quaternion.x
        result.y = -quaternion.y
        result.z = -quaternion.z
        result.w = quaternion.w
        return result

    @staticmethod
    def magnitudeSquared(quaternion):
        """ Computes the provided Cartesian's squared magnitude.

        Args:
            quaternion: `Quaternion`, he Cartesian instance whose squared magnitude is to be computed.

        Returns:
            A `float`, The squared magnitude.
        """
        return (
            quaternion.x * quaternion.x +
            quaternion.y * quaternion.y +
            quaternion.z * quaternion.z +
            quaternion.w * quaternion.w
        )

    @staticmethod
    def magnitude(quaternion):
        """ Computes the provided Cartesian's magnitude (length).

        Args:
            quaternion: `Quaternion`, he Cartesian instance whose magnitude is to be computed.

        Returns:
            A `float`, The magnitude.
        """
        return math.sqrt(Quaternion.magnitudeSquared(quaternion))

    @staticmethod
    def normalize(quaternion, result):
        """ Computes the normalized form of the supplied Cartesian.

        Args:
            quaternion, `Quaternion`, The Cartesian to be normalized.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        inverseMagnitude = 1.0 / Quaternion.magnitude(quaternion)
        x = quaternion.x * inverseMagnitude
        y = quaternion.y * inverseMagnitude
        z = quaternion.z * inverseMagnitude
        w = quaternion.w * inverseMagnitude

        result.x = x
        result.y = y
        result.z = z
        result.w = w
        return result

    @staticmethod
    def inverse(quaternion, result):
        """ Computes the inverse of the provided quaternion.

        Args:
            quaternion, `Quaternion`, The quaternion to invert.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        magnitudeSquared = Quaternion.magnitudeSquared(quaternion)
        result = Quaternion.conjugate(quaternion, result)
        return Quaternion.multiplyByScalar(result, 1.0 / magnitudeSquared, result)

    @staticmethod
    def add(left, right, result):
        """ Computes the component wise sum of two Quaternions.

        Args:
            left, `Quaternion`, The first Cartesian.
            right, `Quaternion`, The second Cartesian.
            result: `Quaternion`, The object onto which to store the result.

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
        """ Computes the component wise difference of two Quaternions.

        Args:
            left, `Quaternion`, The first Cartesian.
            right, `Quaternion`, The second Cartesian.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        result.x = left.x - right.x
        result.y = left.y - right.y
        result.z = left.z - right.z
        result.w = left.w - right.w
        return result

    @staticmethod
    def negate(quaternion, result):
        """ Negates the provided Cartesian.

        Args:
            quaternion, `Quaternion`, The Cartesian to be negated.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        result.x = -quaternion.x
        result.y = -quaternion.y
        result.z = -quaternion.z
        result.w = -quaternion.w
        return result

    @staticmethod
    def dot(left, right):
        """ Computes the dot (scalar) product of two Quaternions.

        Args:
            left, `Quaternion`, The first Cartesian.
            right, `Quaternion`, The second Cartesian.

        Returns:
            A `float`, The dot product.
        """
        return (
            left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w
        )

    @staticmethod
    def multiply(left, right, result):
        """ Computes the product of two quaternions.

        Args:
            left, `Quaternion`, left The first quaternion.
            right, `float`, right The second quaternion.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        leftX = left.x
        leftY = left.y
        leftZ = left.z
        leftW = left.w

        rightX = right.x
        rightY = right.y
        rightZ = right.z
        rightW = right.w

        x = leftW * rightX + leftX * rightW + leftY * rightZ - leftZ * rightY
        y = leftW * rightY - leftX * rightZ + leftY * rightW + leftZ * rightX
        z = leftW * rightZ + leftX * rightY - leftY * rightX + leftZ * rightW
        w = leftW * rightW - leftX * rightX - leftY * rightY - leftZ * rightZ

        result.x = x
        result.y = y
        result.z = z
        result.w = w
        return result

    @staticmethod
    def multiplyByScalar(quaternion, scalar, result):
        """ Multiplies the provided Cartesian componentwise by the provided scalar.

        Args:
            quaternion, `Quaternion`, The Cartesian to be scaled.
            scalar, `float`, The scalar to multiply with.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        result.x = quaternion.x * scalar
        result.y = quaternion.y * scalar
        result.z = quaternion.z * scalar
        result.w = quaternion.w * scalar
        return result

    @staticmethod
    def divideByScalar(quaternion, scalar, result):
        """ Divides the provided Cartesian componentwise by the provided scalar.

        Args:
            quaternion, `Quaternion`, The Cartesian to be divided.
            scalar, `float`, The scalar to divide by.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        result.x = quaternion.x / scalar
        result.y = quaternion.y / scalar
        result.z = quaternion.z / scalar
        result.w = quaternion.w / scalar
        return result

    @staticmethod
    def computeAxis(quaternion, result):
        """ Computes the axis of rotation of the provided quaternion.

        Args:
            quaternion, `Quaternion`, The quaternion to use.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        w = quaternion.w
        if abs(w - 1.0) < EPSILON6:
            result.x = result.y = result.z = 0
            return result

        scalar = 1.0 / math.sqrt(1.0 - w * w)

        result.x = quaternion.x * scalar
        result.y = quaternion.y * scalar
        result.z = quaternion.z * scalar
        return result

    @staticmethod
    def computeAngle(quaternion):
        """ Computes the angle of rotation of the provided quaternion.

        Args:
            quaternion, `Quaternion`, The quaternion to use.

        Returns:
            A `float`, The angle of rotation.
        """
        if abs(quaternion.w - 1.0) < EPSILON6:
            return 0.0
        return 2.0 * math.acos(quaternion.w)

    @staticmethod
    def lerp(start, end, t, result):
        """ Computes the linear interpolation or extrapolation at t using the provided cartesians.

        Args:
            start, `Quaternion`, The value corresponding to t at 0.0.
            end, `Quaternion`, The value corresponding to t at 1.0.
            t, `float`, The point along t at which to interpolate.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        Quaternion.multiplyByScalar(end, t, _lerpScratch)
        result = Quaternion.multiplyByScalar(start, 1.0 - t, result)
        return Quaternion.add(_lerpScratch, result, result)

    @staticmethod
    def slerp(start, end, t, result):
        """ Computes the spherical linear interpolation or extrapolation at t using the provided quaternions.

        Args:
            start, `Quaternion`, The value corresponding to t at 0.0.
            end, `Quaternion`, The value corresponding to t at 1.0.
            t, `float`, The point along t at which to interpolate.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        dot = Quaternion.dot(start, end)

        r = end
        if dot < 0.0:
            dot = -dot
            r = Quaternion.negate(end, _slerpEndNegated)

        if 1.0 - dot < EPSILON6:
            return Quaternion.lerp(start, r, t, result)

        theta = math.acos(dot)
        Quaternion.multiplyByScalar(start, math.sin((1 - t) * theta), _slerpScaledP)
        Quaternion.multiplyByScalar(r, math.sin(t * theta), _slerpScaledR)
        result = Quaternion.add(_slerpScaledP, _slerpScaledR, result)
        return Quaternion.multiplyByScalar(result, 1.0 / math.sin(theta), result)

    @staticmethod
    def log(quaternion, result):
        """ The logarithmic quaternion function.

        Args:
            quaternion, `Quaternion`, The unit quaternion.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        theta = math.acos(max(min(quaternion.w, 1.0), -1.0))
        thetaOverSinTheta = 0.0

        if theta != 0.0:
            thetaOverSinTheta = theta / math.sin(theta)

        return Cartesian3.multiplyByScalar(quaternion, thetaOverSinTheta, result)

    @staticmethod
    def exp(cartesian, result):
        """ The exponential quaternion function.

        Args:
            cartesian, `Cartesian3`, The cartesian.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        theta = Cartesian3.magnitude(cartesian)
        sinThetaOverTheta = 0.0

        if theta != 0.0:
            sinThetaOverTheta = math.sin(theta) / theta

        result.x = cartesian.x * sinThetaOverTheta
        result.y = cartesian.y * sinThetaOverTheta
        result.z = cartesian.z * sinThetaOverTheta
        result.w = math.cos(theta)

        return result

    @staticmethod
    def computeInnerQuadrangle(q0, q1, q2, result):
        """ Computes an inner quadrangle point.

        Args:
            q0, `Quaternion`, The first quaternion.
            q1, `Quaternion`, The second quaternion.
            q2, `Quaternion`, The third quaternion.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        qInv = Quaternion.conjugate(q1, _squadScratchQuaternion0)
        Quaternion.multiply(qInv, q2, _squadScratchQuaternion1)
        cart0 = Quaternion.log(_squadScratchQuaternion1, _squadScratchCartesian0)

        Quaternion.multiply(qInv, q0, _squadScratchQuaternion1)
        cart1 = Quaternion.log(_squadScratchQuaternion1, _squadScratchCartesian1)

        Cartesian3.add(cart0, cart1, cart0)
        Cartesian3.multiplyByScalar(cart0, 0.25, cart0)
        Cartesian3.negate(cart0, cart0)
        Quaternion.exp(cart0, _squadScratchQuaternion0)

        return Quaternion.multiply(q1, _squadScratchQuaternion0, result)

    @staticmethod
    def squad(q0, q1, s0, s1, t, result):
        """ Computes the spherical quadrangle interpolation between quaternions.

        Args:
            q0, `Quaternion`, The first quaternion.
            q1, `Quaternion`, The second quaternion.
            s0, `Quaternion`, The first inner quadrangle.
            s1, `Quaternion`, The first inner quadrangle.
            t, `float`, The point along t at which to interpolate.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        slerp0 = Quaternion.slerp(q0, q1, t, _squadScratchQuaternion0)
        slerp1 = Quaternion.slerp(s0, s1, t, _squadScratchQuaternion1)
        return Quaternion.slerp(slerp0, slerp1, 2.0 * t * (1.0 - t), result)

    @staticmethod
    def fastSlerp(start, end, t, result):
        """ Computes the spherical linear interpolation or extrapolation at t using the provided quaternions.
        Note: This implementation is faster but is only accurate to 10e-6.

        Args:
            start, `Quaternion`, The value corresponding to t at 0.0.
            end, `Quaternion`, The value corresponding to t at 1.0.
            t, `float`, The point along t at which to interpolate.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        x = Quaternion.dot(start, end)

        sign = 0.0
        if x >= 0:
            sign = 1.0
        else:
            sign = -1.0
            x = -x

        xm1 = x - 1.0
        d = 1.0 - t
        sqrT = t * t
        sqrD = d * d

        for i in range(7, -1, -1):  # (let i = 7 i >= 0 --i):
            bT[i] = (u[i] * sqrT - v[i]) * xm1
            bD[i] = (u[i] * sqrD - v[i]) * xm1

        cT = sign * t * (1.0 + bT[0] * (1.0 + bT[1] * (1.0 + bT[2] * (1.0 + bT[3] * (1.0 + bT[4] * (1.0 + bT[5] * (1.0 + bT[6] * (1.0 + bT[7]))))))))
        cD = d * (1.0 + bD[0] * (1.0 + bD[1] * (1.0 + bD[2] * (1.0 + bD[3] * (1.0 + bD[4] * (1.0 + bD[5] * (1.0 + bD[6] * (1.0 + bD[7]))))))))

        temp = Quaternion.multiplyByScalar(
            start,
            cD,
            _fastSlerpScratchQuaternion
        )
        Quaternion.multiplyByScalar(end, cT, result)
        return Quaternion.add(temp, result, result)

    @staticmethod
    def fastSquad(q0, q1, s0, s1, t, result):
        """ Computes the spherical quadrangle interpolation between quaternions.
        Note: This implementation is faster but is only accurate to 10e-6.

        Args:
            q0, `Quaternion`, The first quaternion.
            q1, `Quaternion`, The second quaternion.
            s0, `Quaternion`, The first inner quadrangle.
            s1, `Quaternion`, The first inner quadrangle.
            t, `float`, The point along t at which to interpolate.
            result: `Quaternion`, The object onto which to store the result.

        Returns:
            A `Quaternion`, The modified result parameter.
        """
        slerp0 = Quaternion.fastSlerp(q0, q1, t, _squadScratchQuaternion0)
        slerp1 = Quaternion.fastSlerp(s0, s1, t, _squadScratchQuaternion1)
        return Quaternion.fastSlerp(slerp0, slerp1, 2.0 * t * (1.0 - t), result)

    @staticmethod
    def equals(left, right):
        """ Compares the provided Quaternions componentwise and return `True` if equal,
        `False` otherwise

        Args:
            left, `Quaternion`, The first Cartesian.
            right: `Quaternion`, The second Cartesian.

        Returns:
            A `boolean`, `True` if equal, `False` otherwise.
        """
        return (
            left is right or
            (left is not None and right is not None and
                left.x == right.x and left.y == right.y and left.z == right.z and left.w == right.w)
        )

    @staticmethod
    def equalsEpsilon(left, right, epsilon=0):
        """ Compares the provided Quaternions componentwise and return `True`
        if they pass an absolute or relative tolerance test, `False` otherwise.

        Args:
            left, `Quaternion`, The first Cartesian.
            right: `Quaternion`, The second Cartesian.
            relativeEpsilon, `float`, The relative epsilon tolerance to use for equality testing. default=0
            absoluteEpsilon, `float`, The absolute epsilon tolerance to use for equality testing. default=0

        Returns:
            A `boolean`, `True` if they pass an absolute or relative tolerance test, `False` otherwise.
        """
        return (
            left is right or
            (left is not None and
                right is not None and
                abs(left.x - right.x) <= epsilon and
                abs(left.y - right.y) <= epsilon and
                abs(left.z - right.z) <= epsilon and
                abs(left.w - right.w) <= epsilon)
        )


_fromAxisAngleScratch = Cartesian3()
_scratchHeadingQuaternion = Quaternion()
_scratchPitchQuaternion = Quaternion()
_scratchRollQuaternion = Quaternion()
_lerpScratch = Quaternion()
_slerpEndNegated = Quaternion()
_slerpScaledP = Quaternion()
_slerpScaledR = Quaternion()
_squadScratchCartesian0 = Cartesian3()
_squadScratchCartesian1 = Cartesian3()
_squadScratchQuaternion0 = Quaternion()
_squadScratchQuaternion1 = Quaternion()
_fastSlerpScratchQuaternion = Quaternion()


def _init_uv():
    opmu = 1.90110745351730037
    u = [0.0] * 8
    v = [0.0] * 8

    for i in range(0, 7):
        s = i + 1.0
        t = 2.0 * s + 1.0
        u[i] = 1.0 / (s * t)
        v[i] = s / t

    u[7] = opmu / (8.0 * 17.0)
    v[7] = (opmu * 8.0) / 17.0

    return u, v


bT = [0.0] * 8
bD = [0.0] * 8
u, v = _init_uv()
