import math

from satsim.vecmath import Cartesian3, Matrix3, Cartesian4
from satsim.math.const import EPSILON7, EPSILON21


class Matrix4:
    """ A 4x4 matrix, indexable as a column-major order array.
    Constructor parameters are in row-major order for code readability.
    """

    packedLength = 16

    def __init__(self,
                 column0Row0=0.0,
                 column1Row0=0.0,
                 column2Row0=0.0,
                 column3Row0=0.0,
                 column0Row1=0.0,
                 column1Row1=0.0,
                 column2Row1=0.0,
                 column3Row1=0.0,
                 column0Row2=0.0,
                 column1Row2=0.0,
                 column2Row2=0.0,
                 column3Row2=0.0,
                 column0Row3=0.0,
                 column1Row3=0.0,
                 column2Row3=0.0,
                 column3Row3=0.0):
        self.m = [column0Row0, column0Row1, column0Row2, column0Row3,
                  column1Row0, column1Row1, column1Row2, column1Row3,
                  column2Row0, column2Row1, column2Row2, column2Row3,
                  column3Row0, column3Row1, column3Row2, column3Row3,]
        """ Constructor.

        Args:
            column0Row0: `float`, The value for column 0, row 0.
            column1Row0: `float`, The value for column 1, row 0.
            column2Row0: `float`, The value for column 2, row 0.
            column3Row0: `float`, The value for column 3, row 0.
            column0Row1: `float`, The value for column 0, row 1.
            column1Row1: `float`, The value for column 1, row 1.
            column2Row1: `float`, The value for column 2, row 1.
            column3Row1: `float`, The value for column 3, row 1.
            column0Row2: `float`, The value for column 0, row 2.
            column1Row2: `float`, The value for column 1, row 2.
            column2Row2: `float`, The value for column 2, row 2.
            column3Row2: `float`, The value for column 3, row 2.
            column0Row3: `float`, The value for column 0, row 3.
            column1Row3: `float`, The value for column 1, row 3.
            column2Row3: `float`, The value for column 2, row 3.
            column3Row3: `float`, The value for column 3, row 3.
        """
    def __getitem__(self, key):
        return self.m[key]

    def __setitem__(self, key, value):
        self.m[key] = value

    def __len__(self):
        return Matrix4.packedLength

    def __str__(self):
        return f"""[{self[0]}, {self[4]}, {self[8]}, {self[12]}]
[{self[1]}, {self[5]}, {self[9]}, {self[13]}]
[{self[2]}, {self[6]}, {self[10]}, {self[14]}]
[{self[3]}, {self[7]}, {self[11]}, {self[15]}]"""

    def __eq__(self, other):
        return (
            self is other or
            (self is not None and
                other is not None and
                self[12] == other[12] and
                self[13] == other[13] and
                self[14] == other[14] and
                self[0] == other[0] and
                self[1] == other[1] and
                self[2] == other[2] and
                self[4] == other[4] and
                self[5] == other[5] and
                self[6] == other[6] and
                self[8] == other[8] and
                self[9] == other[9] and
                self[10] == other[10] and
                self[3] == other[3] and
                self[7] == other[7] and
                self[11] == other[11] and
                self[15] == other[15])
        )

    @staticmethod
    def clone(matrix=None, result=None):
        """ Creates a Matrix4 from 16 consecutive elements in an array.

        Args:
            array: `list`, The array whose 16 consecutive elements correspond to the positions of the matrix.  Assumes column-major order.
            startingIndex: `int`, The offset into the array of the first element, which corresponds to the x component.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        if matrix is None:
            return None

        if result is None:
            return Matrix4(
                matrix[0],
                matrix[4],
                matrix[8],
                matrix[12],
                matrix[1],
                matrix[5],
                matrix[9],
                matrix[13],
                matrix[2],
                matrix[6],
                matrix[10],
                matrix[14],
                matrix[3],
                matrix[7],
                matrix[11],
                matrix[15])

        result[0] = matrix[0]
        result[1] = matrix[1]
        result[2] = matrix[2]
        result[3] = matrix[3]
        result[4] = matrix[4]
        result[5] = matrix[5]
        result[6] = matrix[6]
        result[7] = matrix[7]
        result[8] = matrix[8]
        result[9] = matrix[9]
        result[10] = matrix[10]
        result[11] = matrix[11]
        result[12] = matrix[12]
        result[13] = matrix[13]
        result[14] = matrix[14]
        result[15] = matrix[15]
        return result

    @staticmethod
    def fromArray(array, startingIndex=0, result=None):
        """ Creates a Matrix4 from 16 consecutive elements in an array.

        Args:
            array: `list`, The array whose 16 consecutive elements correspond to the positions of the matrix.  Assumes column-major order.
            startingIndex: `int`, The offset into the array of the first element, which corresponds to the x component.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        if result is None:
            result = Matrix4()

        result[0] = array[startingIndex]
        result[1] = array[startingIndex + 1]
        result[2] = array[startingIndex + 2]
        result[3] = array[startingIndex + 3]
        result[4] = array[startingIndex + 4]
        result[5] = array[startingIndex + 5]
        result[6] = array[startingIndex + 6]
        result[7] = array[startingIndex + 7]
        result[8] = array[startingIndex + 8]
        result[9] = array[startingIndex + 9]
        result[10] = array[startingIndex + 10]
        result[11] = array[startingIndex + 11]
        result[12] = array[startingIndex + 12]
        result[13] = array[startingIndex + 13]
        result[14] = array[startingIndex + 14]
        result[15] = array[startingIndex + 15]
        return result

    @staticmethod
    def fromColumnMajorArray(values, result=None):
        """ Creates a Matrix4 from a column-major order array.
        The resulting matrix will be in column-major order.

        Args:
            values: `list`, The column-major order array.
            startingIndex: `int`, The offset into the array of the first element, which corresponds to the x component.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        return Matrix4.clone(values, result)

    @staticmethod
    def fromRowMajorArray(values, result=None):
        """ Creates a Matrix4 from a row-major order array.
        The resulting matrix will be in column-major order.

        Args:
            values: `list`, The row-major order array.
            startingIndex: `int`, The offset into the array of the first element, which corresponds to the x component.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        if result is None:
            return Matrix4(values[0],
                           values[1],
                           values[2],
                           values[3],
                           values[4],
                           values[5],
                           values[6],
                           values[7],
                           values[8],
                           values[9],
                           values[10],
                           values[11],
                           values[12],
                           values[13],
                           values[14],
                           values[15])

        result[0] = values[0]
        result[1] = values[4]
        result[2] = values[8]
        result[3] = values[12]
        result[4] = values[1]
        result[5] = values[5]
        result[6] = values[9]
        result[7] = values[13]
        result[8] = values[2]
        result[9] = values[6]
        result[10] = values[10]
        result[11] = values[14]
        result[12] = values[3]
        result[13] = values[7]
        result[14] = values[11]
        result[15] = values[15]
        return result

    @staticmethod
    def fromRotationTranslation(rotation, translation=Cartesian3.ZERO(), result=None):
        """ Computes a Matrix4 instance from a Matrix3 representing the rotation
        and a Cartesian3 representing the translation.

        Args:
            rotation: `Matrix3`, The upper left portion of the matrix representing the rotation.
            translation: `Cartesian3`, The upper right portion of the matrix representing the translation.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        if result is None:
            return Matrix4(rotation[0],
                           rotation[3],
                           rotation[6],
                           translation.x,
                           rotation[1],
                           rotation[4],
                           rotation[7],
                           translation.y,
                           rotation[2],
                           rotation[5],
                           rotation[8],
                           translation.z,
                           0.0,
                           0.0,
                           0.0,
                           1.0)

        result[0] = rotation[0]
        result[1] = rotation[1]
        result[2] = rotation[2]
        result[3] = 0.0
        result[4] = rotation[3]
        result[5] = rotation[4]
        result[6] = rotation[5]
        result[7] = 0.0
        result[8] = rotation[6]
        result[9] = rotation[7]
        result[10] = rotation[8]
        result[11] = 0.0
        result[12] = translation.x
        result[13] = translation.y
        result[14] = translation.z
        result[15] = 1.0
        return result

    @staticmethod
    def fromTranslationQuaternionRotationScale(translation, rotation, scale, result=None):
        """ Computes a Matrix4 instance from a translation, rotation, and scale (TRS)
        representation with the rotation represented as a quaternion.

        Args:
            translation: `Cartesian3`, The translation transformation.
            rotation: `Quaternion`, The rotation transformation.
            scale: `Cartesian3`, The non-uniform scale transformation.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        if result is None:
            result = Matrix4()

        scaleX = scale.x
        scaleY = scale.y
        scaleZ = scale.z

        x2 = rotation.x * rotation.x
        xy = rotation.x * rotation.y
        xz = rotation.x * rotation.z
        xw = rotation.x * rotation.w
        y2 = rotation.y * rotation.y
        yz = rotation.y * rotation.z
        yw = rotation.y * rotation.w
        z2 = rotation.z * rotation.z
        zw = rotation.z * rotation.w
        w2 = rotation.w * rotation.w

        m00 = x2 - y2 - z2 + w2
        m01 = 2.0 * (xy - zw)
        m02 = 2.0 * (xz + yw)

        m10 = 2.0 * (xy + zw)
        m11 = -x2 + y2 - z2 + w2
        m12 = 2.0 * (yz - xw)

        m20 = 2.0 * (xz - yw)
        m21 = 2.0 * (yz + xw)
        m22 = -x2 - y2 + z2 + w2

        result[0] = m00 * scaleX
        result[1] = m10 * scaleX
        result[2] = m20 * scaleX
        result[3] = 0.0
        result[4] = m01 * scaleY
        result[5] = m11 * scaleY
        result[6] = m21 * scaleY
        result[7] = 0.0
        result[8] = m02 * scaleZ
        result[9] = m12 * scaleZ
        result[10] = m22 * scaleZ
        result[11] = 0.0
        result[12] = translation.x
        result[13] = translation.y
        result[14] = translation.z
        result[15] = 1.0

        return result

    @staticmethod
    def fromTranslationRotationScale(translationRotationScale, result=None):
        """ Creates a Matrix4 instance from a dict.

        Args:
            translationRotationScale: `dict`, The translation, rotation, and scale.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        return Matrix4.fromTranslationQuaternionRotationScale(
            translationRotationScale['translation'],
            translationRotationScale['rotation'],
            translationRotationScale['scale'],
            result)

    @staticmethod
    def fromTranslation(translation, result=None):
        """ Creates a Matrix4 instance from a Cartesian3 representing the translation.

        Args:
            translation: `Cartesian3`, The translation transformation.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        return Matrix4.fromRotationTranslation(Matrix3.IDENTITY(), translation, result)

    @staticmethod
    def fromScale(scale, result=None):
        """ Computes a Matrix4 instance representing a non-uniform scale.

        Args:
            scale: `Cartesian3`, The x, y, and z scale factors.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        if result is None:
            return Matrix4(scale.x, 0.0, 0.0, 0.0,
                           0.0, scale.y, 0.0, 0.0,
                           0.0, 0.0, scale.z, 0.0,
                           0.0, 0.0, 0.0, 1.0)

        result[0] = scale.x
        result[1] = 0.0
        result[2] = 0.0
        result[3] = 0.0
        result[4] = 0.0
        result[5] = scale.y
        result[6] = 0.0
        result[7] = 0.0
        result[8] = 0.0
        result[9] = 0.0
        result[10] = scale.z
        result[11] = 0.0
        result[12] = 0.0
        result[13] = 0.0
        result[14] = 0.0
        result[15] = 1.0
        return result

    @staticmethod
    def fromUniformScale(scale, result=None):
        """ Computes a Matrix4 instance representing a uniform scale.

        Args:
            scale: `float`, The uniform scale factor.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        if result is None:
            return Matrix4(scale, 0.0, 0.0, 0.0,
                           0.0, scale, 0.0, 0.0,
                           0.0, 0.0, scale, 0.0,
                           0.0, 0.0, 0.0, 1.0)

        result[0] = scale
        result[1] = 0.0
        result[2] = 0.0
        result[3] = 0.0
        result[4] = 0.0
        result[5] = scale
        result[6] = 0.0
        result[7] = 0.0
        result[8] = 0.0
        result[9] = 0.0
        result[10] = scale
        result[11] = 0.0
        result[12] = 0.0
        result[13] = 0.0
        result[14] = 0.0
        result[15] = 1.0
        return result

    @staticmethod
    def fromRotation(rotation, result=None):
        """ Creates a rotation matrix.

        Args:
            rotation: `Matrix3`, The rotation matrix.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        if result is None:
            result = Matrix4()

        result[0] = rotation[0]
        result[1] = rotation[1]
        result[2] = rotation[2]
        result[3] = 0.0

        result[4] = rotation[3]
        result[5] = rotation[4]
        result[6] = rotation[5]
        result[7] = 0.0

        result[8] = rotation[6]
        result[9] = rotation[7]
        result[10] = rotation[8]
        result[11] = 0.0

        result[12] = 0.0
        result[13] = 0.0
        result[14] = 0.0
        result[15] = 1.0

        return result

    @staticmethod
    def fromCamera(camera, result=None):
        """ Computes a Matrix4 instance from a Camera.

        Args:
            camera: `dict`,  The camera to use.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        position = camera['position']
        direction = camera['direction']
        up = camera['up']

        Cartesian3.normalize(direction, _fromCameraF)
        Cartesian3.normalize(
            Cartesian3.cross(_fromCameraF, up, _fromCameraR),
            _fromCameraR
        )
        Cartesian3.normalize(
            Cartesian3.cross(_fromCameraR, _fromCameraF, _fromCameraU),
            _fromCameraU
        )

        sX = _fromCameraR.x
        sY = _fromCameraR.y
        sZ = _fromCameraR.z
        fX = _fromCameraF.x
        fY = _fromCameraF.y
        fZ = _fromCameraF.z
        uX = _fromCameraU.x
        uY = _fromCameraU.y
        uZ = _fromCameraU.z
        positionX = position.x
        positionY = position.y
        positionZ = position.z
        t0 = sX * -positionX + sY * -positionY + sZ * -positionZ
        t1 = uX * -positionX + uY * -positionY + uZ * -positionZ
        t2 = fX * positionX + fY * positionY + fZ * positionZ

        if result is None:
            return Matrix4(
                sX,
                sY,
                sZ,
                t0,
                uX,
                uY,
                uZ,
                t1,
                -fX,
                -fY,
                -fZ,
                t2,
                0.0,
                0.0,
                0.0,
                1.0
            )

        result[0] = sX
        result[1] = uX
        result[2] = -fX
        result[3] = 0.0
        result[4] = sY
        result[5] = uY
        result[6] = -fY
        result[7] = 0.0
        result[8] = sZ
        result[9] = uZ
        result[10] = -fZ
        result[11] = 0.0
        result[12] = t0
        result[13] = t1
        result[14] = t2
        result[15] = 1.0
        return result

    @staticmethod
    def computePerspectiveFieldOfView(fovY, aspectRatio, near, far, result=None):
        """ Computes a Matrix4 instance representing a perspective transformation matrix.

        Args:
            fovY: `float`, The field of view along the Y axis in radians.
            aspectRatio: `float`, The aspect ratio.
            near: `float`, The distance to the near plane in meters.
            far: `float`, The distance to the far plane in meters.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        if result is None:
            result = Matrix4()

        bottom = math.tan(fovY * 0.5)

        column1Row1 = 1.0 / bottom
        column0Row0 = column1Row1 / aspectRatio
        column2Row2 = (far + near) / (near - far)
        column3Row2 = (2.0 * far * near) / (near - far)

        result[0] = column0Row0
        result[1] = 0.0
        result[2] = 0.0
        result[3] = 0.0
        result[4] = 0.0
        result[5] = column1Row1
        result[6] = 0.0
        result[7] = 0.0
        result[8] = 0.0
        result[9] = 0.0
        result[10] = column2Row2
        result[11] = -1.0
        result[12] = 0.0
        result[13] = 0.0
        result[14] = column3Row2
        result[15] = 0.0
        return result

    @staticmethod
    def computeOrthographicOffCenter(left, right, bottom, top, near, far, result=None):
        """ Computes a Matrix4 instance representing a perspective transformation matrix.

        Args:
            left: `float`, The number of meters to the left of the camera that will be in view.
            right: `float`, The number of meters to the right of the camera that will be in view.
            bottom: `float`, The number of meters below of the camera that will be in view.
            top: `float`, The number of meters above of the camera that will be in view.
            near: `float`, The distance to the near plane in meters.
            far: `float`, The distance to the far plane in meters.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        if result is None:
            result = Matrix4()

        a = 1.0 / (right - left)
        b = 1.0 / (top - bottom)
        c = 1.0 / (far - near)

        tx = -(right + left) * a
        ty = -(top + bottom) * b
        tz = -(far + near) * c
        a *= 2.0
        b *= 2.0
        c *= -2.0

        result[0] = a
        result[1] = 0.0
        result[2] = 0.0
        result[3] = 0.0
        result[4] = 0.0
        result[5] = b
        result[6] = 0.0
        result[7] = 0.0
        result[8] = 0.0
        result[9] = 0.0
        result[10] = c
        result[11] = 0.0
        result[12] = tx
        result[13] = ty
        result[14] = tz
        result[15] = 1.0
        return result

    @staticmethod
    def computePerspectiveOffCenter(left, right, bottom, top, near, far, result=None):
        """ Computes a Matrix4 instance representing an off center perspective transformation.

        Args:
            left: `float`, The number of meters to the left of the camera that will be in view.
            right: `float`, The number of meters to the right of the camera that will be in view.
            bottom: `float`, The number of meters below of the camera that will be in view.
            top: `float`, The number of meters above of the camera that will be in view.
            near: `float`, The distance to the near plane in meters.
            far: `float`, The distance to the far plane in meters.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        if result is None:
            result = Matrix4()

        column0Row0 = (2.0 * near) / (right - left)
        column1Row1 = (2.0 * near) / (top - bottom)
        column2Row0 = (right + left) / (right - left)
        column2Row1 = (top + bottom) / (top - bottom)
        column2Row2 = -(far + near) / (far - near)
        column2Row3 = -1.0
        column3Row2 = (-2.0 * far * near) / (far - near)

        result[0] = column0Row0
        result[1] = 0.0
        result[2] = 0.0
        result[3] = 0.0
        result[4] = 0.0
        result[5] = column1Row1
        result[6] = 0.0
        result[7] = 0.0
        result[8] = column2Row0
        result[9] = column2Row1
        result[10] = column2Row2
        result[11] = column2Row3
        result[12] = 0.0
        result[13] = 0.0
        result[14] = column3Row2
        result[15] = 0.0
        return result

    @staticmethod
    def computeInfinitePerspectiveOffCenter(left, right, bottom, top, near, result=None):
        """ Computes a Matrix4 instance representing an infinite off center perspective transformation.

        Args:
            left: `float`, The number of meters to the left of the camera that will be in view.
            right: `float`, The number of meters to the right of the camera that will be in view.
            bottom: `float`, The number of meters below of the camera that will be in view.
            top: `float`, The number of meters above of the camera that will be in view.
            near: `float`, The distance to the near plane in meters.
            far: `float`, The distance to the far plane in meters.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        if result is None:
            result = Matrix4()

        column0Row0 = (2.0 * near) / (right - left)
        column1Row1 = (2.0 * near) / (top - bottom)
        column2Row0 = (right + left) / (right - left)
        column2Row1 = (top + bottom) / (top - bottom)
        column2Row2 = -1.0
        column2Row3 = -1.0
        column3Row2 = -2.0 * near

        result[0] = column0Row0
        result[1] = 0.0
        result[2] = 0.0
        result[3] = 0.0
        result[4] = 0.0
        result[5] = column1Row1
        result[6] = 0.0
        result[7] = 0.0
        result[8] = column2Row0
        result[9] = column2Row1
        result[10] = column2Row2
        result[11] = column2Row3
        result[12] = 0.0
        result[13] = 0.0
        result[14] = column3Row2
        result[15] = 0.0
        return result

    @staticmethod
    def computeViewportTransformation(viewport={}, nearDepthRange=0.0, farDepthRange=1.0, result=None):
        """ Computes a Matrix4 instance that transforms from normalized device coordinates to window coordinates.

        Args:
            viewport: `dict`, The viewport's corners.
            nearDepthRange: `float`, The near plane distance in window coordinates.
            farDepthRange: `float`, The far plane distance in window coordinates.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        if result is None:
            result = Matrix4()

        x = viewport['x']
        y = viewport['y']
        width = viewport['width']
        height = viewport['height']

        halfWidth = width * 0.5
        halfHeight = height * 0.5
        halfDepth = (farDepthRange - nearDepthRange) * 0.5

        column0Row0 = halfWidth
        column1Row1 = halfHeight
        column2Row2 = halfDepth
        column3Row0 = x + halfWidth
        column3Row1 = y + halfHeight
        column3Row2 = nearDepthRange + halfDepth
        column3Row3 = 1.0

        result[0] = column0Row0
        result[1] = 0.0
        result[2] = 0.0
        result[3] = 0.0
        result[4] = 0.0
        result[5] = column1Row1
        result[6] = 0.0
        result[7] = 0.0
        result[8] = 0.0
        result[9] = 0.0
        result[10] = column2Row2
        result[11] = 0.0
        result[12] = column3Row0
        result[13] = column3Row1
        result[14] = column3Row2
        result[15] = column3Row3

        return result

    @staticmethod
    def computeLookAt(position, target, up, result=None):
        """ Computes look at transform from observer to target.

        Args:
            position: `Cartesian3`, The observer position.
            target: `Cartesian3`, The target position.
            up: `Cartesian3`, The up direction.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        if result is None:
            result = Matrix4()

        direction = Cartesian3()
        direction = Cartesian3.subtract(target, position, direction)
        direction = Cartesian3.normalize(direction, direction)
        right = Cartesian3()
        right = Cartesian3.cross(up, Cartesian3.negate(direction, right), right)
        return Matrix4.computeView(position, direction, up, right, result)

    @staticmethod
    def computeView(position, direction, up, right, result=None):
        """ Computes a Matrix4 instance that transforms from world space to view space.

        Args:
            position: `Cartesian3`, The position of the camera.
            direction: `Cartesian3`, The forward direction.
            up: `Cartesian3`, The up direction.
            right: `Cartesian3`, The right direction.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter or a new Matrix4 instance if one was not provided.
        """
        if result is None:
            result = Matrix4()

        result[0] = right.x
        result[1] = up.x
        result[2] = -direction.x
        result[3] = 0.0
        result[4] = right.y
        result[5] = up.y
        result[6] = -direction.y
        result[7] = 0.0
        result[8] = right.z
        result[9] = up.z
        result[10] = -direction.z
        result[11] = 0.0
        result[12] = -Cartesian3.dot(right, position)
        result[13] = -Cartesian3.dot(up, position)
        result[14] = Cartesian3.dot(direction, position)
        result[15] = 1.0
        return result

    @staticmethod
    def toArray(matrix, result=None):
        """ Creates an Array from the provided Matrix4 instance.
        The array will be in column-major order.

        Args:
            matrix: `Matrix4`, The matrix to use.
            result: `list`, The Array onto which to store the result.

        Returns:
            A `list`, The modified Array parameter or a new Array instance if one was not provided.
        """
        if result is None:
            return [
                matrix[0],
                matrix[1],
                matrix[2],
                matrix[3],
                matrix[4],
                matrix[5],
                matrix[6],
                matrix[7],
                matrix[8],
                matrix[9],
                matrix[10],
                matrix[11],
                matrix[12],
                matrix[13],
                matrix[14],
                matrix[15],
            ]

        result[0] = matrix[0]
        result[1] = matrix[1]
        result[2] = matrix[2]
        result[3] = matrix[3]
        result[4] = matrix[4]
        result[5] = matrix[5]
        result[6] = matrix[6]
        result[7] = matrix[7]
        result[8] = matrix[8]
        result[9] = matrix[9]
        result[10] = matrix[10]
        result[11] = matrix[11]
        result[12] = matrix[12]
        result[13] = matrix[13]
        result[14] = matrix[14]
        result[15] = matrix[15]
        return result

    @staticmethod
    def getElementIndex(column, row):
        """ Computes the array index of the element at the provided row and column.

        Args:
            row: `int`, The zero-based index of the row.
            column: `int`, The zero-based index of the column.

        Returns:
            A `int`, The index of the element at the provided row and column.
        """
        return column * 4 + row

    @staticmethod
    def getColumn(matrix, index, result):
        """ Retrieves a copy of the matrix column at the provided index as a Cartesian3 instance.

        Args:
            matrix: `Matrix4`, The matrix to use.
            index: `int`, The zero-based index of the column to retrieve.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter.
        """
        startIndex = index * 4
        x = matrix[startIndex]
        y = matrix[startIndex + 1]
        z = matrix[startIndex + 2]
        w = matrix[startIndex + 3]

        result.x = x
        result.y = y
        result.z = z
        result.w = w
        return result

    @staticmethod
    def setColumn(matrix, index, cartesian, result):
        """ Computes a new matrix that replaces the specified column in the provided matrix with the provided Cartesian3 instance.

        Args:
            matrix: `Matrix4`, The matrix to use.
            index: `int`, The zero-based index of the column to retrieve.
            cartesian: `Cartesian3`, The Cartesian whose values will be assigned to the specified column.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        result = Matrix4.clone(matrix, result)
        startIndex = index * 4
        result[startIndex] = cartesian.x
        result[startIndex + 1] = cartesian.y
        result[startIndex + 2] = cartesian.z
        result[startIndex + 3] = cartesian.w
        return result

    @staticmethod
    def getRow(matrix, index, result):
        """ Retrieves a copy of the matrix row at the provided index as a Cartesian3 instance.

        Args:
            matrix: `Matrix4`, The matrix to use.
            index: `int`, The zero-based index of the row to retrieve.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `Cartesian4`, The modified result parameter.
        """
        x = matrix[index]
        y = matrix[index + 4]
        z = matrix[index + 8]
        w = matrix[index + 12]

        result.x = x
        result.y = y
        result.z = z
        result.w = w
        return result

    @staticmethod
    def setRow(matrix, index, cartesian, result):
        """ Computes a new matrix that replaces the specified row in the provided matrix with the provided Cartesian3 instance.

        Args:
            matrix: `Matrix4`, The matrix to use.
            index: `int`, The zero-based index of the row to retrieve.
            cartesian: `Cartesian3`, The Cartesian whose values will be assigned to the specified row.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        result = Matrix4.clone(matrix, result)
        result[index] = cartesian.x
        result[index + 4] = cartesian.y
        result[index + 8] = cartesian.z
        result[index + 12] = cartesian.w
        return result

    @staticmethod
    def setTranslation(matrix, translation, result):
        """ Computes a new matrix that replaces the translation in the rightmost column of the provided
            matrix with the provided translation. This assumes the matrix is an affine transformation.

        Args:
            matrix: `Matrix4`, The matrix to use.
            translation: `Cartesian3`, The translation that replaces the translation of the provided matrix.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        result[0] = matrix[0]
        result[1] = matrix[1]
        result[2] = matrix[2]
        result[3] = matrix[3]

        result[4] = matrix[4]
        result[5] = matrix[5]
        result[6] = matrix[6]
        result[7] = matrix[7]

        result[8] = matrix[8]
        result[9] = matrix[9]
        result[10] = matrix[10]
        result[11] = matrix[11]

        result[12] = translation.x
        result[13] = translation.y
        result[14] = translation.z
        result[15] = matrix[15]

        return result

    @staticmethod
    def setScale(matrix, scale, result):
        """ Computes a new matrix that replaces the scale with the provided scale.
        This assumes the matrix is an affine transformation.

        Args:
            matrix: `Matrix4`, The matrix to use.
            scale: `Cartesian3`, The scale that replaces the scale of the provided matrix.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        existingScale = Matrix4.getScale(matrix, _scaleScratch1)
        scaleRatioX = scale.x / existingScale.x
        scaleRatioY = scale.y / existingScale.y
        scaleRatioZ = scale.z / existingScale.z

        result[0] = matrix[0] * scaleRatioX
        result[1] = matrix[1] * scaleRatioX
        result[2] = matrix[2] * scaleRatioX
        result[3] = matrix[3]

        result[4] = matrix[4] * scaleRatioY
        result[5] = matrix[5] * scaleRatioY
        result[6] = matrix[6] * scaleRatioY
        result[7] = matrix[7]

        result[8] = matrix[8] * scaleRatioZ
        result[9] = matrix[9] * scaleRatioZ
        result[10] = matrix[10] * scaleRatioZ
        result[11] = matrix[11]

        result[12] = matrix[12]
        result[13] = matrix[13]
        result[14] = matrix[14]
        result[15] = matrix[15]

        return result

    @staticmethod
    def setUniformScale(matrix, scale, result):
        """ Computes a new matrix that replaces the scale with the provided uniform scale.
        This assumes the matrix is an affine transformation.

        Args:
            matrix: `Matrix4`, The matrix to use.
            scale: `float`, The uniform scale that replaces the scale of the provided matrix.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        existingScale = Matrix4.getScale(matrix, _scaleScratch2)
        scaleRatioX = scale / existingScale.x
        scaleRatioY = scale / existingScale.y
        scaleRatioZ = scale / existingScale.z

        result[0] = matrix[0] * scaleRatioX
        result[1] = matrix[1] * scaleRatioX
        result[2] = matrix[2] * scaleRatioX
        result[3] = matrix[3]

        result[4] = matrix[4] * scaleRatioY
        result[5] = matrix[5] * scaleRatioY
        result[6] = matrix[6] * scaleRatioY
        result[7] = matrix[7]

        result[8] = matrix[8] * scaleRatioZ
        result[9] = matrix[9] * scaleRatioZ
        result[10] = matrix[10] * scaleRatioZ
        result[11] = matrix[11]

        result[12] = matrix[12]
        result[13] = matrix[13]
        result[14] = matrix[14]
        result[15] = matrix[15]

        return result

    @staticmethod
    def getScale(matrix, result):
        """ Extracts the non-uniform scale assuming the matrix is an affine transformation.

        Args:
            matrix: `Matrix4`, The matrix.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter.
        """
        result.x = Cartesian3.magnitude(
            Cartesian3.fromElements(matrix[0], matrix[1], matrix[2], _scratchColumn)
        )
        result.y = Cartesian3.magnitude(
            Cartesian3.fromElements(matrix[4], matrix[5], matrix[6], _scratchColumn)
        )
        result.z = Cartesian3.magnitude(
            Cartesian3.fromElements(matrix[8], matrix[9], matrix[10], _scratchColumn)
        )
        return result

    @staticmethod
    def getMaximumScale(matrix):
        """ Computes the maximum scale assuming the matrix is an affine transformation.
        The maximum scale is the maximum length of the column vectors.

        Args:
            matrix: `Matrix4`, The matrix.

        Returns:
            A `float`, The maximum scale.
        """
        Matrix4.getScale(matrix, _scaleScratch3)
        return Cartesian3.maximumComponent(_scaleScratch3)

    @staticmethod
    def setRotation(matrix, rotation, result):
        """ Sets the rotation assuming the matrix is an affine transformation.

        Args:
            matrix: `Matrix4`, The matrix to use.
            rotation: `Matrix3`, The rotation matrix.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        scale = Matrix4.getScale(matrix, _scaleScratch4)

        result[0] = rotation[0] * scale.x
        result[1] = rotation[1] * scale.x
        result[2] = rotation[2] * scale.x
        result[3] = matrix[3]

        result[4] = rotation[3] * scale.y
        result[5] = rotation[4] * scale.y
        result[6] = rotation[5] * scale.y
        result[7] = matrix[7]

        result[8] = rotation[6] * scale.z
        result[9] = rotation[7] * scale.z
        result[10] = rotation[8] * scale.z
        result[11] = matrix[11]

        result[12] = matrix[12]
        result[13] = matrix[13]
        result[14] = matrix[14]
        result[15] = matrix[15]

        return result

    @staticmethod
    def getRotation(matrix, result):
        """ Extracts the rotation matrix assuming the matrix is an affine transformation.

        Args:
            matrix: `Matrix4`, The matrix.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        scale = Matrix4.getScale(matrix, _scaleScratch5)

        result[0] = matrix[0] / scale.x
        result[1] = matrix[1] / scale.x
        result[2] = matrix[2] / scale.x

        result[3] = matrix[4] / scale.y
        result[4] = matrix[5] / scale.y
        result[5] = matrix[6] / scale.y

        result[6] = matrix[8] / scale.z
        result[7] = matrix[9] / scale.z
        result[8] = matrix[10] / scale.z

        return result

    @staticmethod
    def multiply(left, right, result):
        """ Computes the product of two matrices.

        Args:
            left: `Matrix4`, The first matrix.
            right: `Matrix4`, The second matrix.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        left0 = left[0]
        left1 = left[1]
        left2 = left[2]
        left3 = left[3]
        left4 = left[4]
        left5 = left[5]
        left6 = left[6]
        left7 = left[7]
        left8 = left[8]
        left9 = left[9]
        left10 = left[10]
        left11 = left[11]
        left12 = left[12]
        left13 = left[13]
        left14 = left[14]
        left15 = left[15]

        right0 = right[0]
        right1 = right[1]
        right2 = right[2]
        right3 = right[3]
        right4 = right[4]
        right5 = right[5]
        right6 = right[6]
        right7 = right[7]
        right8 = right[8]
        right9 = right[9]
        right10 = right[10]
        right11 = right[11]
        right12 = right[12]
        right13 = right[13]
        right14 = right[14]
        right15 = right[15]

        column0Row0 = left0 * right0 + left4 * right1 + left8 * right2 + left12 * right3
        column0Row1 = left1 * right0 + left5 * right1 + left9 * right2 + left13 * right3
        column0Row2 = left2 * right0 + left6 * right1 + left10 * right2 + left14 * right3
        column0Row3 = left3 * right0 + left7 * right1 + left11 * right2 + left15 * right3

        column1Row0 = left0 * right4 + left4 * right5 + left8 * right6 + left12 * right7
        column1Row1 = left1 * right4 + left5 * right5 + left9 * right6 + left13 * right7
        column1Row2 = left2 * right4 + left6 * right5 + left10 * right6 + left14 * right7
        column1Row3 = left3 * right4 + left7 * right5 + left11 * right6 + left15 * right7

        column2Row0 = left0 * right8 + left4 * right9 + left8 * right10 + left12 * right11
        column2Row1 = left1 * right8 + left5 * right9 + left9 * right10 + left13 * right11
        column2Row2 = left2 * right8 + left6 * right9 + left10 * right10 + left14 * right11
        column2Row3 = left3 * right8 + left7 * right9 + left11 * right10 + left15 * right11

        column3Row0 = left0 * right12 + left4 * right13 + left8 * right14 + left12 * right15
        column3Row1 = left1 * right12 + left5 * right13 + left9 * right14 + left13 * right15
        column3Row2 = left2 * right12 + left6 * right13 + left10 * right14 + left14 * right15
        column3Row3 = left3 * right12 + left7 * right13 + left11 * right14 + left15 * right15

        result[0] = column0Row0
        result[1] = column0Row1
        result[2] = column0Row2
        result[3] = column0Row3
        result[4] = column1Row0
        result[5] = column1Row1
        result[6] = column1Row2
        result[7] = column1Row3
        result[8] = column2Row0
        result[9] = column2Row1
        result[10] = column2Row2
        result[11] = column2Row3
        result[12] = column3Row0
        result[13] = column3Row1
        result[14] = column3Row2
        result[15] = column3Row3
        return result

    @staticmethod
    def add(left, right, result):
        """ Computes the sum of two matrices.

        Args:
            left: `Matrix4`, The first matrix.
            right: `Matrix4`, The second matrix.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        result[0] = left[0] + right[0]
        result[1] = left[1] + right[1]
        result[2] = left[2] + right[2]
        result[3] = left[3] + right[3]
        result[4] = left[4] + right[4]
        result[5] = left[5] + right[5]
        result[6] = left[6] + right[6]
        result[7] = left[7] + right[7]
        result[8] = left[8] + right[8]
        result[9] = left[9] + right[9]
        result[10] = left[10] + right[10]
        result[11] = left[11] + right[11]
        result[12] = left[12] + right[12]
        result[13] = left[13] + right[13]
        result[14] = left[14] + right[14]
        result[15] = left[15] + right[15]
        return result

    @staticmethod
    def subtract(left, right, result):
        """ Computes the difference of two matrices.

        Args:
            left: `Matrix4`, The first matrix.
            right: `Matrix4`, The second matrix.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        result[0] = left[0] - right[0]
        result[1] = left[1] - right[1]
        result[2] = left[2] - right[2]
        result[3] = left[3] - right[3]
        result[4] = left[4] - right[4]
        result[5] = left[5] - right[5]
        result[6] = left[6] - right[6]
        result[7] = left[7] - right[7]
        result[8] = left[8] - right[8]
        result[9] = left[9] - right[9]
        result[10] = left[10] - right[10]
        result[11] = left[11] - right[11]
        result[12] = left[12] - right[12]
        result[13] = left[13] - right[13]
        result[14] = left[14] - right[14]
        result[15] = left[15] - right[15]
        return result

    @staticmethod
    def multiplyTransformation(left, right, result):
        """ Computes the product of two matrices assuming the matrices are affine transformation matrices,
        where the upper left 3x3 elements are any matrix, and
        the upper three elements in the fourth column are the translation.
        The bottom row is assumed to be [0, 0, 0, 1].
        The matrix is not verified to be in the proper form.
        This method is faster than computing the product for general 4x4 matrices.

        Args:
            left: `Matrix4`, The first matrix.
            right: `Matrix4`, The second matrix.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        left0 = left[0]
        left1 = left[1]
        left2 = left[2]
        left4 = left[4]
        left5 = left[5]
        left6 = left[6]
        left8 = left[8]
        left9 = left[9]
        left10 = left[10]
        left12 = left[12]
        left13 = left[13]
        left14 = left[14]

        right0 = right[0]
        right1 = right[1]
        right2 = right[2]
        right4 = right[4]
        right5 = right[5]
        right6 = right[6]
        right8 = right[8]
        right9 = right[9]
        right10 = right[10]
        right12 = right[12]
        right13 = right[13]
        right14 = right[14]

        column0Row0 = left0 * right0 + left4 * right1 + left8 * right2
        column0Row1 = left1 * right0 + left5 * right1 + left9 * right2
        column0Row2 = left2 * right0 + left6 * right1 + left10 * right2

        column1Row0 = left0 * right4 + left4 * right5 + left8 * right6
        column1Row1 = left1 * right4 + left5 * right5 + left9 * right6
        column1Row2 = left2 * right4 + left6 * right5 + left10 * right6

        column2Row0 = left0 * right8 + left4 * right9 + left8 * right10
        column2Row1 = left1 * right8 + left5 * right9 + left9 * right10
        column2Row2 = left2 * right8 + left6 * right9 + left10 * right10

        column3Row0 = left0 * right12 + left4 * right13 + left8 * right14 + left12
        column3Row1 = left1 * right12 + left5 * right13 + left9 * right14 + left13
        column3Row2 = left2 * right12 + left6 * right13 + left10 * right14 + left14

        result[0] = column0Row0
        result[1] = column0Row1
        result[2] = column0Row2
        result[3] = 0.0
        result[4] = column1Row0
        result[5] = column1Row1
        result[6] = column1Row2
        result[7] = 0.0
        result[8] = column2Row0
        result[9] = column2Row1
        result[10] = column2Row2
        result[11] = 0.0
        result[12] = column3Row0
        result[13] = column3Row1
        result[14] = column3Row2
        result[15] = 1.0
        return result

    @staticmethod
    def multiplyByMatrix3(matrix, rotation, result):
        """ Multiplies a transformation matrix by a 3x3 rotation matrix.  This is an optimization
        with less allocations and arithmetic operations than fromRotationTranslation.

        Args:
            matrix: `Matrix4`, The matrix on the left-hand side.
            rotation: `Matrix4`, The 3x3 rotation matrix on the right-hand side.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        left0 = matrix[0]
        left1 = matrix[1]
        left2 = matrix[2]
        left4 = matrix[4]
        left5 = matrix[5]
        left6 = matrix[6]
        left8 = matrix[8]
        left9 = matrix[9]
        left10 = matrix[10]

        right0 = rotation[0]
        right1 = rotation[1]
        right2 = rotation[2]
        right4 = rotation[3]
        right5 = rotation[4]
        right6 = rotation[5]
        right8 = rotation[6]
        right9 = rotation[7]
        right10 = rotation[8]

        column0Row0 = left0 * right0 + left4 * right1 + left8 * right2
        column0Row1 = left1 * right0 + left5 * right1 + left9 * right2
        column0Row2 = left2 * right0 + left6 * right1 + left10 * right2

        column1Row0 = left0 * right4 + left4 * right5 + left8 * right6
        column1Row1 = left1 * right4 + left5 * right5 + left9 * right6
        column1Row2 = left2 * right4 + left6 * right5 + left10 * right6

        column2Row0 = left0 * right8 + left4 * right9 + left8 * right10
        column2Row1 = left1 * right8 + left5 * right9 + left9 * right10
        column2Row2 = left2 * right8 + left6 * right9 + left10 * right10

        result[0] = column0Row0
        result[1] = column0Row1
        result[2] = column0Row2
        result[3] = 0.0
        result[4] = column1Row0
        result[5] = column1Row1
        result[6] = column1Row2
        result[7] = 0.0
        result[8] = column2Row0
        result[9] = column2Row1
        result[10] = column2Row2
        result[11] = 0.0
        result[12] = matrix[12]
        result[13] = matrix[13]
        result[14] = matrix[14]
        result[15] = matrix[15]
        return result

    @staticmethod
    def multiplyByTranslation(matrix, translation, result):
        """ Multiplies a transformation matrix by an implicit translation matrix defined by a Cartesian3.
        This is an optimization with less allocations and arithmetic operations than fromRotationTranslation.

        Args:
            matrix: `Matrix4`, The matrix on the left-hand side.
            translation: `Matrix4`, The translation on the right-hand side.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        x = translation.x
        y = translation.y
        z = translation.z

        tx = x * matrix[0] + y * matrix[4] + z * matrix[8] + matrix[12]
        ty = x * matrix[1] + y * matrix[5] + z * matrix[9] + matrix[13]
        tz = x * matrix[2] + y * matrix[6] + z * matrix[10] + matrix[14]

        result[0] = matrix[0]
        result[1] = matrix[1]
        result[2] = matrix[2]
        result[3] = matrix[3]
        result[4] = matrix[4]
        result[5] = matrix[5]
        result[6] = matrix[6]
        result[7] = matrix[7]
        result[8] = matrix[8]
        result[9] = matrix[9]
        result[10] = matrix[10]
        result[11] = matrix[11]
        result[12] = tx
        result[13] = ty
        result[14] = tz
        result[15] = matrix[15]
        return result

    @staticmethod
    def multiplyByScale(matrix, scale, result):
        """ Multiplies a transformation matrix by an implicit non-uniform scale matrix.
        This is an optimization with less allocations and arithmetic operations than fromRotationTranslation.

        Args:
            matrix: `Matrix4`, The matrix on the left-hand side.
            scale: `Cartesian3`, The non-uniform scale on the right-hand side.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        scaleX = scale.x
        scaleY = scale.y
        scaleZ = scale.z

        if (scaleX == 1.0 and scaleY == 1.0 and scaleZ == 1.0):
            return Matrix4.clone(matrix, result)

        result[0] = scaleX * matrix[0]
        result[1] = scaleX * matrix[1]
        result[2] = scaleX * matrix[2]
        result[3] = matrix[3]

        result[4] = scaleY * matrix[4]
        result[5] = scaleY * matrix[5]
        result[6] = scaleY * matrix[6]
        result[7] = matrix[7]

        result[8] = scaleZ * matrix[8]
        result[9] = scaleZ * matrix[9]
        result[10] = scaleZ * matrix[10]
        result[11] = matrix[11]

        result[12] = matrix[12]
        result[13] = matrix[13]
        result[14] = matrix[14]
        result[15] = matrix[15]

        return result

    @staticmethod
    def multiplyByUniformScale(matrix, scale, result):
        """ Computes the product of a matrix times a uniform scale, as if the scale were a scale matrix.

        Args:
            matrix: `Matrix4`, The matrix on the left-hand side.
            scale: `float`, The uniform scale on the right-hand side.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        result[0] = matrix[0] * scale
        result[1] = matrix[1] * scale
        result[2] = matrix[2] * scale
        result[3] = matrix[3]

        result[4] = matrix[4] * scale
        result[5] = matrix[5] * scale
        result[6] = matrix[6] * scale
        result[7] = matrix[7]

        result[8] = matrix[8] * scale
        result[9] = matrix[9] * scale
        result[10] = matrix[10] * scale
        result[11] = matrix[11]

        result[12] = matrix[12]
        result[13] = matrix[13]
        result[14] = matrix[14]
        result[15] = matrix[15]

        return result

    @staticmethod
    def multiplyByVector(matrix, cartesian, result):
        """ Computes the product of a matrix and a column vector.

        Args:
            matrix: `Matrix4`, The matrix.
            cartesian: `Cartesian4`, The vector.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `Cartesian4`, The modified result parameter.
        """
        vX = cartesian.x
        vY = cartesian.y
        vZ = cartesian.z
        vW = cartesian.w

        x = matrix[0] * vX + matrix[4] * vY + matrix[8] * vZ + matrix[12] * vW
        y = matrix[1] * vX + matrix[5] * vY + matrix[9] * vZ + matrix[13] * vW
        z = matrix[2] * vX + matrix[6] * vY + matrix[10] * vZ + matrix[14] * vW
        w = matrix[3] * vX + matrix[7] * vY + matrix[11] * vZ + matrix[15] * vW

        result.x = x
        result.y = y
        result.z = z
        result.w = w
        return result

    @staticmethod
    def multiplyByPointAsVector(matrix, cartesian, result):
        """ Computes the product of a matrix and a Cartesian3.

        Args:
            matrix: `Matrix4`, The matrix.
            cartesian: `Cartesian4`, The point.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `Cartesian4`, The modified result parameter.
        """
        vX = cartesian.x
        vY = cartesian.y
        vZ = cartesian.z

        x = matrix[0] * vX + matrix[4] * vY + matrix[8] * vZ
        y = matrix[1] * vX + matrix[5] * vY + matrix[9] * vZ
        z = matrix[2] * vX + matrix[6] * vY + matrix[10] * vZ

        result.x = x
        result.y = y
        result.z = z
        return result

    @staticmethod
    def multiplyByPoint(matrix, cartesian, result):
        """ Computes the product of a matrix and a Cartesian3. This is equivalent to calling multiplyByVector
            with a Cartesian4 with a w component of 1, but returns a Cartesian3 instead of a Cartesian4.

        Args:
            matrix: `Matrix4`, The matrix.
            cartesian: `Cartesian4`, The point.
            result: `Cartesian4`, The object onto which to store the result.

        Returns:
            A `Cartesian4`, The modified result parameter.
        """
        vX = cartesian.x
        vY = cartesian.y
        vZ = cartesian.z

        x = matrix[0] * vX + matrix[4] * vY + matrix[8] * vZ + matrix[12]
        y = matrix[1] * vX + matrix[5] * vY + matrix[9] * vZ + matrix[13]
        z = matrix[2] * vX + matrix[6] * vY + matrix[10] * vZ + matrix[14]

        result.x = x
        result.y = y
        result.z = z
        return result

    @staticmethod
    def multiplyByScalar(matrix, scalar, result):
        """ Computes the product of a matrix and a scalar.

        Args:
            matrix: `Matrix4`, The matrix.
            scalar: `float`, The number to multiply by.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        result[0] = matrix[0] * scalar
        result[1] = matrix[1] * scalar
        result[2] = matrix[2] * scalar
        result[3] = matrix[3] * scalar
        result[4] = matrix[4] * scalar
        result[5] = matrix[5] * scalar
        result[6] = matrix[6] * scalar
        result[7] = matrix[7] * scalar
        result[8] = matrix[8] * scalar
        result[9] = matrix[9] * scalar
        result[10] = matrix[10] * scalar
        result[11] = matrix[11] * scalar
        result[12] = matrix[12] * scalar
        result[13] = matrix[13] * scalar
        result[14] = matrix[14] * scalar
        result[15] = matrix[15] * scalar
        return result

    @staticmethod
    def negate(matrix, result):
        """ Negates the provided matrix.

        Args:
            matrix, `Matrix4`, The matrix to be negated.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        result[0] = -matrix[0]
        result[1] = -matrix[1]
        result[2] = -matrix[2]
        result[3] = -matrix[3]
        result[4] = -matrix[4]
        result[5] = -matrix[5]
        result[6] = -matrix[6]
        result[7] = -matrix[7]
        result[8] = -matrix[8]
        result[9] = -matrix[9]
        result[10] = -matrix[10]
        result[11] = -matrix[11]
        result[12] = -matrix[12]
        result[13] = -matrix[13]
        result[14] = -matrix[14]
        result[15] = -matrix[15]
        return result

    @staticmethod
    def transpose(matrix, result):
        """ Computes the transpose of the provided matrix.

        Args:
            matrix, `Matrix4`, The matrix to be transpose.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        matrix1 = matrix[1]
        matrix2 = matrix[2]
        matrix3 = matrix[3]
        matrix6 = matrix[6]
        matrix7 = matrix[7]
        matrix11 = matrix[11]

        result[0] = matrix[0]
        result[1] = matrix[4]
        result[2] = matrix[8]
        result[3] = matrix[12]
        result[4] = matrix1
        result[5] = matrix[5]
        result[6] = matrix[9]
        result[7] = matrix[13]
        result[8] = matrix2
        result[9] = matrix6
        result[10] = matrix[10]
        result[11] = matrix[14]
        result[12] = matrix3
        result[13] = matrix7
        result[14] = matrix11
        result[15] = matrix[15]
        return result

    @staticmethod
    def abs(matrix, result):
        """ Computes a matrix, which contains the absolute (unsigned) values of the provided matrix's elements.

        Args:
            matrix, `Matrix4`, The matrix with signed elements.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        result[0] = abs(matrix[0])
        result[1] = abs(matrix[1])
        result[2] = abs(matrix[2])
        result[3] = abs(matrix[3])
        result[4] = abs(matrix[4])
        result[5] = abs(matrix[5])
        result[6] = abs(matrix[6])
        result[7] = abs(matrix[7])
        result[8] = abs(matrix[8])
        result[9] = abs(matrix[9])
        result[10] = abs(matrix[10])
        result[11] = abs(matrix[11])
        result[12] = abs(matrix[12])
        result[13] = abs(matrix[13])
        result[14] = abs(matrix[14])
        result[15] = abs(matrix[15])

        return result

    @staticmethod
    def equalsEpsilon(left, right, epsilon=0):
        """ Compares the provided matrices componentwise and return `True`
        if they pass an absolute or relative tolerance test, `False` otherwise.

        Args:
            left, `Matrix4`, The first Cartesian.
            right: `Matrix4`, The second Cartesian.
            epsilon, `float`, The epsilon to use for equality testing.

        Returns:
            A `boolean`, `True` if they pass an absolute or relative tolerance test, `False` otherwise.
        """
        return (
            left is right or
            (left is not None and
                right is not None and
                abs(left[0] - right[0]) <= epsilon and
                abs(left[1] - right[1]) <= epsilon and
                abs(left[2] - right[2]) <= epsilon and
                abs(left[3] - right[3]) <= epsilon and
                abs(left[4] - right[4]) <= epsilon and
                abs(left[5] - right[5]) <= epsilon and
                abs(left[6] - right[6]) <= epsilon and
                abs(left[7] - right[7]) <= epsilon and
                abs(left[8] - right[8]) <= epsilon and
                abs(left[9] - right[9]) <= epsilon and
                abs(left[10] - right[10]) <= epsilon and
                abs(left[11] - right[11]) <= epsilon and
                abs(left[12] - right[12]) <= epsilon and
                abs(left[13] - right[13]) <= epsilon and
                abs(left[14] - right[14]) <= epsilon and
                abs(left[15] - right[15]) <= epsilon)
        )

    @staticmethod
    def getTranslation(matrix, result):
        """ Gets the translation portion of the provided matrix, assuming the matrix is an affine transformation matrix.

        Args:
            matrix, `Matrix4`, The matrix to use.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        result.x = matrix[12]
        result.y = matrix[13]
        result.z = matrix[14]
        return result

    @staticmethod
    def getMatrix3(matrix, result):
        """ Gets the upper left 3x3 matrix of the provided matrix.

        Args:
            matrix, `Matrix4`, The matrix to use.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        result[0] = matrix[0]
        result[1] = matrix[1]
        result[2] = matrix[2]
        result[3] = matrix[4]
        result[4] = matrix[5]
        result[5] = matrix[6]
        result[6] = matrix[8]
        result[7] = matrix[9]
        result[8] = matrix[10]
        return result

    @staticmethod
    def inverse(matrix, result):
        """ Computes the inverse of the provided matrix using Cramers Rule.
        If the determinant is zero, the matrix can not be inverted, and an exception is thrown.
        If the matrix is a proper rigid transformation, it is more efficient
        to invert it with {@link Matrix4.inverseTransformation}.

        Args:
            matrix, `Matrix4`, The matrix to invert.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        src0 = matrix[0]
        src1 = matrix[4]
        src2 = matrix[8]
        src3 = matrix[12]
        src4 = matrix[1]
        src5 = matrix[5]
        src6 = matrix[9]
        src7 = matrix[13]
        src8 = matrix[2]
        src9 = matrix[6]
        src10 = matrix[10]
        src11 = matrix[14]
        src12 = matrix[3]
        src13 = matrix[7]
        src14 = matrix[11]
        src15 = matrix[15]

        tmp0 = src10 * src15
        tmp1 = src11 * src14
        tmp2 = src9 * src15
        tmp3 = src11 * src13
        tmp4 = src9 * src14
        tmp5 = src10 * src13
        tmp6 = src8 * src15
        tmp7 = src11 * src12
        tmp8 = src8 * src14
        tmp9 = src10 * src12
        tmp10 = src8 * src13
        tmp11 = src9 * src12

        dst0 = tmp0 * src5 + tmp3 * src6 + tmp4 * src7 - (tmp1 * src5 + tmp2 * src6 + tmp5 * src7)
        dst1 = tmp1 * src4 + tmp6 * src6 + tmp9 * src7 - (tmp0 * src4 + tmp7 * src6 + tmp8 * src7)
        dst2 = tmp2 * src4 + tmp7 * src5 + tmp10 * src7 - (tmp3 * src4 + tmp6 * src5 + tmp11 * src7)
        dst3 = tmp5 * src4 + tmp8 * src5 + tmp11 * src6 - (tmp4 * src4 + tmp9 * src5 + tmp10 * src6)
        dst4 = tmp1 * src1 + tmp2 * src2 + tmp5 * src3 - (tmp0 * src1 + tmp3 * src2 + tmp4 * src3)
        dst5 = tmp0 * src0 + tmp7 * src2 + tmp8 * src3 - (tmp1 * src0 + tmp6 * src2 + tmp9 * src3)
        dst6 = tmp3 * src0 + tmp6 * src1 + tmp11 * src3 - (tmp2 * src0 + tmp7 * src1 + tmp10 * src3)
        dst7 = tmp4 * src0 + tmp9 * src1 + tmp10 * src2 - (tmp5 * src0 + tmp8 * src1 + tmp11 * src2)

        tmp0 = src2 * src7
        tmp1 = src3 * src6
        tmp2 = src1 * src7
        tmp3 = src3 * src5
        tmp4 = src1 * src6
        tmp5 = src2 * src5
        tmp6 = src0 * src7
        tmp7 = src3 * src4
        tmp8 = src0 * src6
        tmp9 = src2 * src4
        tmp10 = src0 * src5
        tmp11 = src1 * src4

        dst8 = tmp0 * src13 + tmp3 * src14 + tmp4 * src15 - (tmp1 * src13 + tmp2 * src14 + tmp5 * src15)
        dst9 = tmp1 * src12 + tmp6 * src14 + tmp9 * src15 - (tmp0 * src12 + tmp7 * src14 + tmp8 * src15)
        dst10 = tmp2 * src12 + tmp7 * src13 + tmp10 * src15 - (tmp3 * src12 + tmp6 * src13 + tmp11 * src15)
        dst11 = tmp5 * src12 + tmp8 * src13 + tmp11 * src14 - (tmp4 * src12 + tmp9 * src13 + tmp10 * src14)
        dst12 = tmp2 * src10 + tmp5 * src11 + tmp1 * src9 - (tmp4 * src11 + tmp0 * src9 + tmp3 * src10)
        dst13 = tmp8 * src11 + tmp0 * src8 + tmp7 * src10 - (tmp6 * src10 + tmp9 * src11 + tmp1 * src8)
        dst14 = tmp6 * src9 + tmp11 * src11 + tmp3 * src8 - (tmp10 * src11 + tmp2 * src8 + tmp7 * src9)
        dst15 = tmp10 * src10 + tmp4 * src8 + tmp9 * src9 - (tmp8 * src9 + tmp11 * src10 + tmp5 * src8)

        det = src0 * dst0 + src1 * dst1 + src2 * dst2 + src3 * dst3

        if abs(det) < EPSILON21:
            if Matrix3.equalsEpsilon(Matrix4.getMatrix3(matrix, _scratchInverseRotation), _scratchMatrix3Zero, EPSILON7) and Cartesian4.equals(Matrix4.getRow(matrix, 3, _scratchBottomRow), _scratchExpectedBottomRow):
                result[0] = 0.0
                result[1] = 0.0
                result[2] = 0.0
                result[3] = 0.0
                result[4] = 0.0
                result[5] = 0.0
                result[6] = 0.0
                result[7] = 0.0
                result[8] = 0.0
                result[9] = 0.0
                result[10] = 0.0
                result[11] = 0.0
                result[12] = -matrix[12]
                result[13] = -matrix[13]
                result[14] = -matrix[14]
                result[15] = 1.0
                return result

            raise Exception("matrix is not invertible because its determinate is zero")

        det = 1.0 / det

        result[0] = dst0 * det
        result[1] = dst1 * det
        result[2] = dst2 * det
        result[3] = dst3 * det
        result[4] = dst4 * det
        result[5] = dst5 * det
        result[6] = dst6 * det
        result[7] = dst7 * det
        result[8] = dst8 * det
        result[9] = dst9 * det
        result[10] = dst10 * det
        result[11] = dst11 * det
        result[12] = dst12 * det
        result[13] = dst13 * det
        result[14] = dst14 * det
        result[15] = dst15 * det
        return result

    @staticmethod
    def inverseTransformation(matrix, result):
        """ Computes the inverse of the provided matrix assuming it is a proper rigid matrix,
        where the upper left 3x3 elements are a rotation matrix,
        and the upper three elements in the fourth column are the translation.
        The bottom row is assumed to be [0, 0, 0, 1].
        The matrix is not verified to be in the proper form.
        This method is faster than computing the inverse for a general 4x4
        matrix using inverse.

        Args:
            matrix, `Matrix4`, The matrix to invert.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        matrix0 = matrix[0]
        matrix1 = matrix[1]
        matrix2 = matrix[2]
        matrix4 = matrix[4]
        matrix5 = matrix[5]
        matrix6 = matrix[6]
        matrix8 = matrix[8]
        matrix9 = matrix[9]
        matrix10 = matrix[10]

        vX = matrix[12]
        vY = matrix[13]
        vZ = matrix[14]

        x = -matrix0 * vX - matrix1 * vY - matrix2 * vZ
        y = -matrix4 * vX - matrix5 * vY - matrix6 * vZ
        z = -matrix8 * vX - matrix9 * vY - matrix10 * vZ

        result[0] = matrix0
        result[1] = matrix4
        result[2] = matrix8
        result[3] = 0.0
        result[4] = matrix1
        result[5] = matrix5
        result[6] = matrix9
        result[7] = 0.0
        result[8] = matrix2
        result[9] = matrix6
        result[10] = matrix10
        result[11] = 0.0
        result[12] = x
        result[13] = y
        result[14] = z
        result[15] = 1.0
        return result

    @staticmethod
    def inverseTranspose(matrix, result):
        """ Computes the inverse transpose of a matrix.

        Args:
            matrix, `Matrix4`, The matrix to transpose and invert.
            result: `Matrix4`, The object onto which to store the result.

        Returns:
            A `Matrix4`, The modified result parameter.
        """
        return Matrix4.inverse(
            Matrix4.transpose(matrix, _scratchTransposeMatrix),
            result)

    @staticmethod
    def IDENTITY():
        """ A Matrix4 initialized to the identity matrix."""
        return Matrix4(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0)

    @staticmethod
    def ZERO():
        """ A Matrix4 initialized to the zero matrix."""
        return Matrix4(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0)

    COLUMN0ROW0 = 0
    COLUMN0ROW1 = 1
    COLUMN0ROW2 = 2
    COLUMN0ROW3 = 3
    COLUMN1ROW0 = 4
    COLUMN1ROW1 = 5
    COLUMN1ROW2 = 6
    COLUMN1ROW3 = 7
    COLUMN2ROW0 = 8
    COLUMN2ROW1 = 9
    COLUMN2ROW2 = 10
    COLUMN2ROW3 = 11
    COLUMN3ROW0 = 12
    COLUMN3ROW1 = 13
    COLUMN3ROW2 = 14
    COLUMN3ROW3 = 15


_fromCameraF = Cartesian3()
_fromCameraR = Cartesian3()
_fromCameraU = Cartesian3()

_scaleScratch1 = Cartesian3()
_scaleScratch2 = Cartesian3()
_scaleScratch3 = Cartesian3()
_scaleScratch4 = Cartesian3()
_scaleScratch5 = Cartesian3()

_scratchColumn = Cartesian3()
_scratchInverseRotation = Matrix3()
_scratchMatrix3Zero = Matrix3()
_scratchBottomRow = Cartesian4()
_scratchExpectedBottomRow = Cartesian4(0.0, 0.0, 0.0, 1.0)
_scratchTransposeMatrix = Matrix4()
