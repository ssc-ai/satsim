import math
from satsim.vecmath import Cartesian3
from satsim.math.const import EPSILON15, EPSILON20


class Matrix3:
    """ A 3x3 matrix, indexable as a column-major order array.
    Constructor parameters are in row-major order for code readability.
    """

    packedLength = 9

    def __init__(self,
                 column0Row0=0.0,
                 column1Row0=0.0,
                 column2Row0=0.0,
                 column0Row1=0.0,
                 column1Row1=0.0,
                 column2Row1=0.0,
                 column0Row2=0.0,
                 column1Row2=0.0,
                 column2Row2=0.0):
        """ Constructor.

        Args:
            column0Row0: `float`, The value for column 0, row 0.
            column1Row0: `float`, The value for column 1, row 0.
            column2Row0: `float`, The value for column 2, row 0.
            column0Row1: `float`, The value for column 0, row 1.
            column1Row1: `float`, The value for column 1, row 1.
            column2Row1: `float`, The value for column 2, row 1.
            column0Row2: `float`, The value for column 0, row 2.
            column1Row2: `float`, The value for column 1, row 2.
            column2Row2: `float`, The value for column 2, row 2.
        """
        self.m = [column0Row0, column0Row1, column0Row2,
                  column1Row0, column1Row1, column1Row2,
                  column2Row0, column2Row1, column2Row2]

    def __getitem__(self, key):
        return self.m[key]

    def __setitem__(self, key, value):
        self.m[key] = value

    def __len__(self):
        return Matrix3.packedLength

    def __str__(self):
        return f"""[{self[0]}, {self[3]}, {self[6]}]
[{self[1]}, {self[4]}, {self[7]}]
[{self[2]}, {self[5]}, {self[8]}]"""

    def __eq__(self, other):
        return (
            self is other or
            (self is not None and
                other is not None and
                self[0] == other[0] and
                self[1] == other[1] and
                self[2] == other[2] and
                self[3] == other[3] and
                self[4] == other[4] and
                self[5] == other[5] and
                self[6] == other[6] and
                self[7] == other[7] and
                self[8] == other[8])
        )

    @staticmethod
    def clone(matrix=None, result=None):
        """ Creates a Matrix3 from 9 consecutive elements in an array.

        Args:
            array: `list`, The array whose 9 consecutive elements correspond to the positions of the matrix.  Assumes column-major order.
            startingIndex: `int`, The offset into the array of the first element, which corresponds to the x component.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter or a new Matrix3 instance if one was not provided.
        """
        if matrix is None:
            return None

        if result is None:
            return Matrix3(
                matrix[0],
                matrix[3],
                matrix[6],
                matrix[1],
                matrix[4],
                matrix[7],
                matrix[2],
                matrix[5],
                matrix[8])

        result[0] = matrix[0]
        result[1] = matrix[1]
        result[2] = matrix[2]
        result[3] = matrix[3]
        result[4] = matrix[4]
        result[5] = matrix[5]
        result[6] = matrix[6]
        result[7] = matrix[7]
        result[8] = matrix[8]
        return result

    @staticmethod
    def fromColumnMajorArray(values, result=None):
        """ Creates a Matrix3 from a column-major order array.
        The resulting matrix will be in column-major order.

        Args:
            values: `list`, The column-major order array.
            startingIndex: `int`, The offset into the array of the first element, which corresponds to the x component.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter or a new Matrix3 instance if one was not provided.
        """
        return Matrix3.clone(values, result)

    @staticmethod
    def fromRowMajorArray(values, result=None):
        """ Creates a Matrix3 from a row-major order array.
        The resulting matrix will be in column-major order.

        Args:
            values: `list`, The row-major order array.
            startingIndex: `int`, The offset into the array of the first element, which corresponds to the x component.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter or a new Matrix3 instance if one was not provided.
        """
        if result is None:
            return Matrix3(
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                values[6],
                values[7],
                values[8])

        result[0] = values[0]
        result[1] = values[3]
        result[2] = values[6]
        result[3] = values[1]
        result[4] = values[4]
        result[5] = values[7]
        result[6] = values[2]
        result[7] = values[5]
        result[8] = values[8]
        return result

    @staticmethod
    def fromArray(array, startingIndex=0, result=None):
        """ Creates a Matrix3 from 9 consecutive elements in an array.

        Args:
            array: `list`, The array whose 9 consecutive elements correspond to the positions of the matrix.  Assumes column-major order.
            startingIndex: `int`, The offset into the array of the first element, which corresponds to the x component.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter or a new Matrix3 instance if one was not provided.
        """
        if result is None:
            result = Matrix3()

        result[0] = array[startingIndex]
        result[1] = array[startingIndex + 1]
        result[2] = array[startingIndex + 2]
        result[3] = array[startingIndex + 3]
        result[4] = array[startingIndex + 4]
        result[5] = array[startingIndex + 5]
        result[6] = array[startingIndex + 6]
        result[7] = array[startingIndex + 7]
        result[8] = array[startingIndex + 8]
        return result

    @staticmethod
    def fromQuaternion(quaternion, result=None):
        """ Computes a 3x3 rotation matrix from the provided quaternion.

        Args:
            quaternion: `Quaternion`, The quaternion to use.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter or a new Matrix3 instance if one was not provided.
        """
        x2 = quaternion.x * quaternion.x
        xy = quaternion.x * quaternion.y
        xz = quaternion.x * quaternion.z
        xw = quaternion.x * quaternion.w
        y2 = quaternion.y * quaternion.y
        yz = quaternion.y * quaternion.z
        yw = quaternion.y * quaternion.w
        z2 = quaternion.z * quaternion.z
        zw = quaternion.z * quaternion.w
        w2 = quaternion.w * quaternion.w

        m00 = x2 - y2 - z2 + w2
        m01 = 2.0 * (xy - zw)
        m02 = 2.0 * (xz + yw)

        m10 = 2.0 * (xy + zw)
        m11 = -x2 + y2 - z2 + w2
        m12 = 2.0 * (yz - xw)

        m20 = 2.0 * (xz - yw)
        m21 = 2.0 * (yz + xw)
        m22 = -x2 - y2 + z2 + w2

        if result is None:
            return Matrix3(m00, m01, m02, m10, m11, m12, m20, m21, m22)

        result[0] = m00
        result[1] = m10
        result[2] = m20
        result[3] = m01
        result[4] = m11
        result[5] = m21
        result[6] = m02
        result[7] = m12
        result[8] = m22
        return result

    @staticmethod
    def fromHeadingPitchRoll(headingPitchRoll, result=None):
        """ Computes a 3x3 rotation matrix from the provided headingPitchRoll.

        Args:
            headingPitchRoll: `dict`, The headingPitchRoll to use.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter or a new Matrix3 instance if one was not provided.
        """
        cosTheta = math.cos(-headingPitchRoll['pitch'])
        cosPsi = math.cos(-headingPitchRoll['heading'])
        cosPhi = math.cos(headingPitchRoll['roll'])
        sinTheta = math.sin(-headingPitchRoll['pitch'])
        sinPsi = math.sin(-headingPitchRoll['heading'])
        sinPhi = math.sin(headingPitchRoll['roll'])

        m00 = cosTheta * cosPsi
        m01 = -cosPhi * sinPsi + sinPhi * sinTheta * cosPsi
        m02 = sinPhi * sinPsi + cosPhi * sinTheta * cosPsi

        m10 = cosTheta * sinPsi
        m11 = cosPhi * cosPsi + sinPhi * sinTheta * sinPsi
        m12 = -sinPhi * cosPsi + cosPhi * sinTheta * sinPsi

        m20 = -sinTheta
        m21 = sinPhi * cosTheta
        m22 = cosPhi * cosTheta

        if result is None:
            return Matrix3(m00, m01, m02, m10, m11, m12, m20, m21, m22)

        result[0] = m00
        result[1] = m10
        result[2] = m20
        result[3] = m01
        result[4] = m11
        result[5] = m21
        result[6] = m02
        result[7] = m12
        result[8] = m22
        return result

    @staticmethod
    def fromScale(scale, result=None):
        """ Computes a Matrix3 instance representing a non-uniform scale.

        Args:
            scale: `Cartesian3`, The x, y, and z scale factors.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter or a new Matrix3 instance if one was not provided.
        """
        if result is None:
            return Matrix3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z)

        result[0] = scale.x
        result[1] = 0.0
        result[2] = 0.0
        result[3] = 0.0
        result[4] = scale.y
        result[5] = 0.0
        result[6] = 0.0
        result[7] = 0.0
        result[8] = scale.z
        return result

    @staticmethod
    def fromUniformScale(scale, result=None):
        """ Computes a Matrix3 instance representing a uniform scale.

        Args:
            scale: `float`, The uniform scale factor.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter or a new Matrix3 instance if one was not provided.
        """
        if result is None:
            return Matrix3(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, scale)

        result[0] = scale
        result[1] = 0.0
        result[2] = 0.0
        result[3] = 0.0
        result[4] = scale
        result[5] = 0.0
        result[6] = 0.0
        result[7] = 0.0
        result[8] = scale
        return result

    @staticmethod
    def fromCrossProduct(vector, result=None):
        """ Computes a Matrix3 instance representing the cross product equivalent matrix of a Cartesian3 vector.

        Args:
            vector: `Cartesian3`, The vector on the left hand side of the cross product operation.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter or a new Matrix3 instance if one was not provided.
        """
        if result is None:
            return Matrix3(0.0, -vector.z, vector.y,
                           vector.z, 0.0, -vector.x,
                           -vector.y, vector.x, 0.0)

        result[0] = 0.0
        result[1] = vector.z
        result[2] = -vector.y
        result[3] = -vector.z
        result[4] = 0.0
        result[5] = vector.x
        result[6] = vector.y
        result[7] = -vector.x
        result[8] = 0.0
        return result

    @staticmethod
    def fromRotationX(angle, result=None):
        """ Creates a rotation matrix around the x-axis.

        Args:
            angle: `float`, The angle, in radians, of the rotation.  Positive angles are counterclockwise.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter or a new Matrix3 instance if one was not provided.
        """
        cosAngle = math.cos(angle)
        sinAngle = math.sin(angle)

        if result is None:
            return Matrix3(1.0, 0.0, 0.0,
                           0.0, cosAngle,-sinAngle,
                           0.0, sinAngle, cosAngle)

        result[0] = 1.0
        result[1] = 0.0
        result[2] = 0.0
        result[3] = 0.0
        result[4] = cosAngle
        result[5] = sinAngle
        result[6] = 0.0
        result[7] = -sinAngle
        result[8] = cosAngle

        return result

    @staticmethod
    def fromRotationY(angle, result=None):
        """ Creates a rotation matrix around the y-axis.

        Args:
            angle: `float`, The angle, in radians, of the rotation.  Positive angles are counterclockwise.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter or a new Matrix3 instance if one was not provided.
        """
        cosAngle = math.cos(angle)
        sinAngle = math.sin(angle)

        if result is None:
            return Matrix3(cosAngle, 0.0, sinAngle,
                           0.0, 1.0, 0.0,
                           -sinAngle, 0.0, cosAngle)

        result[0] = cosAngle
        result[1] = 0.0
        result[2] = -sinAngle
        result[3] = 0.0
        result[4] = 1.0
        result[5] = 0.0
        result[6] = sinAngle
        result[7] = 0.0
        result[8] = cosAngle

        return result

    @staticmethod
    def fromRotationZ(angle, result=None):
        """ Creates a rotation matrix around the z-axis.

        Args:
            angle: `float`, The angle, in radians, of the rotation.  Positive angles are counterclockwise.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter or a new Matrix3 instance if one was not provided.
        """
        cosAngle = math.cos(angle)
        sinAngle = math.sin(angle)

        if result is None:
            return Matrix3(cosAngle, -sinAngle, 0.0,
                           sinAngle, cosAngle, 0.0,
                           0.0, 0.0, 1.0)

        result[0] = cosAngle
        result[1] = sinAngle
        result[2] = 0.0
        result[3] = -sinAngle
        result[4] = cosAngle
        result[5] = 0.0
        result[6] = 0.0
        result[7] = 0.0
        result[8] = 1.0

        return result

    @staticmethod
    def toArray(matrix, result=None):
        """ Creates an Array from the provided Matrix3 instance.
        The array will be in column-major order.

        Args:
            matrix: `Matrix3`, The matrix to use.
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
        return column * 3 + row

    @staticmethod
    def getColumn(matrix, index, result):
        """ Retrieves a copy of the matrix column at the provided index as a Cartesian3 instance.

        Args:
            matrix: `Matrix3`, The matrix to use.
            index: `int`, The zero-based index of the column to retrieve.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter.
        """
        startIndex = index * 3
        x = matrix[startIndex]
        y = matrix[startIndex + 1]
        z = matrix[startIndex + 2]

        result.x = x
        result.y = y
        result.z = z
        return result

    @staticmethod
    def setColumn(matrix, index, cartesian, result):
        """ Computes a new matrix that replaces the specified column in the provided matrix with the provided Cartesian3 instance.

        Args:
            matrix: `Matrix3`, The matrix to use.
            index: `int`, The zero-based index of the column to retrieve.
            cartesian: `Cartesian3`, The Cartesian whose values will be assigned to the specified column.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
        """
        result = Matrix3.clone(matrix, result)
        startIndex = index * 3
        result[startIndex] = cartesian.x
        result[startIndex + 1] = cartesian.y
        result[startIndex + 2] = cartesian.z
        return result

    @staticmethod
    def getRow(matrix, index, result):
        """ Retrieves a copy of the matrix row at the provided index as a Cartesian3 instance.

        Args:
            matrix: `Matrix3`, The matrix to use.
            index: `int`, The zero-based index of the row to retrieve.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter.
        """
        x = matrix[index]
        y = matrix[index + 3]
        z = matrix[index + 6]

        result.x = x
        result.y = y
        result.z = z
        return result

    @staticmethod
    def setRow(matrix, index, cartesian, result):
        """ Computes a new matrix that replaces the specified row in the provided matrix with the provided Cartesian3 instance.

        Args:
            matrix: `Matrix3`, The matrix to use.
            index: `int`, The zero-based index of the row to retrieve.
            cartesian: `Cartesian3`, The Cartesian whose values will be assigned to the specified row.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
        """
        result = Matrix3.clone(matrix, result)
        result[index] = cartesian.x
        result[index + 3] = cartesian.y
        result[index + 6] = cartesian.z
        return result

    @staticmethod
    def setScale(matrix, scale, result):
        """ Computes a new matrix that replaces the scale with the provided scale.
        This assumes the matrix is an affine transformation.

        Args:
            matrix: `Matrix3`, The matrix to use.
            scale: `Cartesian3`, The scale that replaces the scale of the provided matrix.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
        """
        existingScale = Matrix3.getScale(matrix, _scaleScratch1)
        scaleRatioX = scale.x / existingScale.x
        scaleRatioY = scale.y / existingScale.y
        scaleRatioZ = scale.z / existingScale.z

        result[0] = matrix[0] * scaleRatioX
        result[1] = matrix[1] * scaleRatioX
        result[2] = matrix[2] * scaleRatioX
        result[3] = matrix[3] * scaleRatioY
        result[4] = matrix[4] * scaleRatioY
        result[5] = matrix[5] * scaleRatioY
        result[6] = matrix[6] * scaleRatioZ
        result[7] = matrix[7] * scaleRatioZ
        result[8] = matrix[8] * scaleRatioZ

        return result

    @staticmethod
    def setUniformScale(matrix, scale, result):
        """ Computes a new matrix that replaces the scale with the provided uniform scale.
        This assumes the matrix is an affine transformation.

        Args:
            matrix: `Matrix3`, The matrix to use.
            scale: `float`, The uniform scale that replaces the scale of the provided matrix.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
        """
        existingScale = Matrix3.getScale(matrix, _scaleScratch2)
        scaleRatioX = scale / existingScale.x
        scaleRatioY = scale / existingScale.y
        scaleRatioZ = scale / existingScale.z

        result[0] = matrix[0] * scaleRatioX
        result[1] = matrix[1] * scaleRatioX
        result[2] = matrix[2] * scaleRatioX
        result[3] = matrix[3] * scaleRatioY
        result[4] = matrix[4] * scaleRatioY
        result[5] = matrix[5] * scaleRatioY
        result[6] = matrix[6] * scaleRatioZ
        result[7] = matrix[7] * scaleRatioZ
        result[8] = matrix[8] * scaleRatioZ

        return result

    @staticmethod
    def getScale(matrix, result):
        """ Extracts the non-uniform scale assuming the matrix is an affine transformation.

        Args:
            matrix: `Matrix3`, The matrix.
            result: `Cartesian3`, The object onto which to store the result.

        Returns:
            A `Cartesian3`, The modified result parameter.
        """
        result.x = Cartesian3.magnitude(
            Cartesian3.fromElements(matrix[0], matrix[1], matrix[2], _scratchColumn)
        )
        result.y = Cartesian3.magnitude(
            Cartesian3.fromElements(matrix[3], matrix[4], matrix[5], _scratchColumn)
        )
        result.z = Cartesian3.magnitude(
            Cartesian3.fromElements(matrix[6], matrix[7], matrix[8], _scratchColumn)
        )
        return result

    @staticmethod
    def getMaximumScale(matrix):
        """ Computes the maximum scale assuming the matrix is an affine transformation.
        The maximum scale is the maximum length of the column vectors.

        Args:
            matrix: `Matrix3`, The matrix.

        Returns:
            A `float`, The maximum scale.
        """
        Matrix3.getScale(matrix, _scaleScratch3)
        return Cartesian3.maximumComponent(_scaleScratch3)

    @staticmethod
    def setRotation(matrix, rotation, result):
        """ Sets the rotation assuming the matrix is an affine transformation.

        Args:
            matrix: `Matrix3`, The matrix to use.
            rotation: `Matrix3`, The rotation matrix.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
        """
        scale = Matrix3.getScale(matrix, _scaleScratch4)

        result[0] = rotation[0] * scale.x
        result[1] = rotation[1] * scale.x
        result[2] = rotation[2] * scale.x
        result[3] = rotation[3] * scale.y
        result[4] = rotation[4] * scale.y
        result[5] = rotation[5] * scale.y
        result[6] = rotation[6] * scale.z
        result[7] = rotation[7] * scale.z
        result[8] = rotation[8] * scale.z

        return result

    @staticmethod
    def getRotation(matrix, result):
        """ Extracts the rotation matrix assuming the matrix is an affine transformation.

        Args:
            matrix: `Matrix3`, The matrix.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
        """
        scale = Matrix3.getScale(matrix, _scaleScratch5)

        result[0] = matrix[0] / scale.x
        result[1] = matrix[1] / scale.x
        result[2] = matrix[2] / scale.x
        result[3] = matrix[3] / scale.y
        result[4] = matrix[4] / scale.y
        result[5] = matrix[5] / scale.y
        result[6] = matrix[6] / scale.z
        result[7] = matrix[7] / scale.z
        result[8] = matrix[8] / scale.z

        return result

    @staticmethod
    def multiply(left, right, result):
        """ Computes the product of two matrices.

        Args:
            left: `Matrix3`, The first matrix.
            right: `Matrix3`, The second matrix.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
        """
        column0Row0 = left[0] * right[0] + left[3] * right[1] + left[6] * right[2]
        column0Row1 = left[1] * right[0] + left[4] * right[1] + left[7] * right[2]
        column0Row2 = left[2] * right[0] + left[5] * right[1] + left[8] * right[2]

        column1Row0 = left[0] * right[3] + left[3] * right[4] + left[6] * right[5]
        column1Row1 = left[1] * right[3] + left[4] * right[4] + left[7] * right[5]
        column1Row2 = left[2] * right[3] + left[5] * right[4] + left[8] * right[5]

        column2Row0 = left[0] * right[6] + left[3] * right[7] + left[6] * right[8]
        column2Row1 = left[1] * right[6] + left[4] * right[7] + left[7] * right[8]
        column2Row2 = left[2] * right[6] + left[5] * right[7] + left[8] * right[8]

        result[0] = column0Row0
        result[1] = column0Row1
        result[2] = column0Row2
        result[3] = column1Row0
        result[4] = column1Row1
        result[5] = column1Row2
        result[6] = column2Row0
        result[7] = column2Row1
        result[8] = column2Row2

        return result

    @staticmethod
    def add(left, right, result):
        """ Computes the sum of two matrices.

        Args:
            left: `Matrix3`, The first matrix.
            right: `Matrix3`, The second matrix.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
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
        return result

    @staticmethod
    def subtract(left, right, result):
        """ Computes the difference of two matrices.

        Args:
            left: `Matrix3`, The first matrix.
            right: `Matrix3`, The second matrix.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
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
        return result

    @staticmethod
    def multiplyByVector(matrix, cartesian, result):
        """ Computes the product of a matrix and a column vector.

        Args:
            matrix: `Matrix3`, The matrix.
            cartesian: `Cartesian3`, The column.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
        """
        vX = cartesian.x
        vY = cartesian.y
        vZ = cartesian.z

        x = matrix[0] * vX + matrix[3] * vY + matrix[6] * vZ
        y = matrix[1] * vX + matrix[4] * vY + matrix[7] * vZ
        z = matrix[2] * vX + matrix[5] * vY + matrix[8] * vZ

        result.x = x
        result.y = y
        result.z = z
        return result

    @staticmethod
    def multiplyByScalar(matrix, scalar, result):
        """ Computes the product of a matrix and a scalar.

        Args:
            matrix: `Matrix3`, The matrix.
            scalar: `float`, The number to multiply by.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
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
        return result

    @staticmethod
    def multiplyByScale(matrix, scale, result):
        """ Computes the product of a matrix times a (non-uniform) scale, as if the scale were a scale matrix.

        Args:
            matrix: `Matrix3`, The matrix.
            scale: `Cartesian3`, The non-uniform scale on the right-hand side.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
        """
        result[0] = matrix[0] * scale.x
        result[1] = matrix[1] * scale.x
        result[2] = matrix[2] * scale.x
        result[3] = matrix[3] * scale.y
        result[4] = matrix[4] * scale.y
        result[5] = matrix[5] * scale.y
        result[6] = matrix[6] * scale.z
        result[7] = matrix[7] * scale.z
        result[8] = matrix[8] * scale.z

        return result

    @staticmethod
    def multiplyByUniformScale(matrix, scale, result):
        """ Computes the product of a matrix times a uniform scale, as if the scale were a scale matrix.

        Args:
            matrix: `Matrix3`, The matrix.
            scale: `float`, The uniform scale on the right-hand side.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
        """
        result[0] = matrix[0] * scale
        result[1] = matrix[1] * scale
        result[2] = matrix[2] * scale
        result[3] = matrix[3] * scale
        result[4] = matrix[4] * scale
        result[5] = matrix[5] * scale
        result[6] = matrix[6] * scale
        result[7] = matrix[7] * scale
        result[8] = matrix[8] * scale

        return result

    @staticmethod
    def negate(matrix, result):
        """ Negates the provided matrix.

        Args:
            matrix, `Matrix3`, The matrix to be negated.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
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
        return result

    @staticmethod
    def transpose(matrix, result):
        """ Computes the transpose of the provided matrix.

        Args:
            matrix, `Matrix3`, The matrix to be transpose.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
        """
        column0Row0 = matrix[0]
        column0Row1 = matrix[3]
        column0Row2 = matrix[6]
        column1Row0 = matrix[1]
        column1Row1 = matrix[4]
        column1Row2 = matrix[7]
        column2Row0 = matrix[2]
        column2Row1 = matrix[5]
        column2Row2 = matrix[8]

        result[0] = column0Row0
        result[1] = column0Row1
        result[2] = column0Row2
        result[3] = column1Row0
        result[4] = column1Row1
        result[5] = column1Row2
        result[6] = column2Row0
        result[7] = column2Row1
        result[8] = column2Row2
        return result

    @staticmethod
    def computeFrobeniusNorm(matrix):
        """ Computes the Frobenius norm.

        Args:
            matrix, `Matrix3`, The matrix to be computed.

        Returns:
            A `float`, The Frobenius norm.
        """
        norm = 0.0
        for i in range(0, 9):
            temp = matrix[i]
            norm += temp * temp

        return math.sqrt(norm)

    @staticmethod
    def offDiagonalFrobeniusNorm(matrix):
        """ Computes the "off-diagonal" Frobenius norm. Assumes matrix is symmetric.

        Args:
            matrix, `Matrix3`, The matrix to be computed.

        Returns:
            A `float`, The Frobenius norm.
        """
        norm = 0.0
        for i in range(0, 3):
            temp = matrix[Matrix3.getElementIndex(_colVal[i], _rowVal[i])]
            norm += 2.0 * temp * temp

        return math.sqrt(norm)

    @staticmethod
    def shurDecomposition(matrix, result):
        """ This routine was created based upon Matrix Computations, 3rd ed., by Golub and Van Loan,
        section 8.4.2 The 2by2 Symmetric Schur Decomposition.
        The routine takes a matrix, which is assumed to be symmetric, and
        finds the largest off-diagonal term, and then creates
        a matrix (result) which can be used to help reduce it

        Args:
            matrix, `Matrix3`, The matrix.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
        """
        tolerance = EPSILON15

        maxDiagonal = 0.0
        rotAxis = 1

        # find pivot (rotAxis) based on max diagonal of matrix
        for i in range(0, 3):
            temp = abs(
                matrix[Matrix3.getElementIndex(_colVal[i], _rowVal[i])]
            )
            if temp > maxDiagonal:
                rotAxis = i
                maxDiagonal = temp

        c = 1.0
        s = 0.0

        p = _rowVal[rotAxis]
        q = _colVal[rotAxis]

        if abs(matrix[Matrix3.getElementIndex(q, p)]) > tolerance:
            qq = matrix[Matrix3.getElementIndex(q, q)]
            pp = matrix[Matrix3.getElementIndex(p, p)]
            qp = matrix[Matrix3.getElementIndex(q, p)]

            tau = (qq - pp) / 2.0 / qp
            t = 0.0

            if tau < 0.0:
                t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
            else:
                t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))

            c = 1.0 / math.sqrt(1.0 + t * t)
            s = t * c

        result = Matrix3.clone(Matrix3.IDENTITY(), result)

        result[Matrix3.getElementIndex(p, p)] = result[
            Matrix3.getElementIndex(q, q)
        ] = c
        result[Matrix3.getElementIndex(q, p)] = s
        result[Matrix3.getElementIndex(p, q)] = -s

        return result

    @staticmethod
    def computeEigenDecomposition(matrix, result=None):
        """ Computes the eigenvectors and eigenvalues of a symmetric matrix.

        Args:
            matrix, `Matrix3`, The matrix to decompose into diagonal and unitary matrix. Expected to be symmetric.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
        """
        tolerance = EPSILON20
        maxSweeps = 10

        count = 0
        sweep = 0

        if result is None:
            result = {}

        result['unitary'] = Matrix3.IDENTITY()
        unitaryMatrix = result['unitary']

        result['diagonal'] = Matrix3.IDENTITY()
        diagMatrix = Matrix3.clone(matrix, result['diagonal'])

        epsilon = tolerance * Matrix3.computeFrobeniusNorm(diagMatrix)

        while sweep < maxSweeps and Matrix3.offDiagonalFrobeniusNorm(diagMatrix) > epsilon:
            Matrix3.shurDecomposition(diagMatrix, _jMatrix)
            Matrix3.transpose(_jMatrix, _jMatrixTranspose)
            Matrix3.multiply(diagMatrix, _jMatrix, diagMatrix)
            Matrix3.multiply(_jMatrixTranspose, diagMatrix, diagMatrix)
            Matrix3.multiply(unitaryMatrix, _jMatrix, unitaryMatrix)

            count = count + 1
            if (count > 2):
                sweep = sweep + 1
                count = 0

        return result

    @staticmethod
    def abs(matrix, result):
        """ Computes a matrix, which contains the absolute (unsigned) values of the provided matrix's elements.

        Args:
            matrix, `Matrix3`, The matrix with signed elements.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
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

        return result

    @staticmethod
    def determinant(matrix):
        """ Computes the determinant of the provided matrix.

        Args:
            matrix, `Matrix3`, The matrix to use.

        Returns:
            A `float`, The value of the determinant of the matrix.
        """
        m11 = matrix[0]
        m21 = matrix[3]
        m31 = matrix[6]
        m12 = matrix[1]
        m22 = matrix[4]
        m32 = matrix[7]
        m13 = matrix[2]
        m23 = matrix[5]
        m33 = matrix[8]

        return (
            m11 * (m22 * m33 - m23 * m32) +
            m12 * (m23 * m31 - m21 * m33) +
            m13 * (m21 * m32 - m22 * m31)
        )

    @staticmethod
    def inverse(matrix, result):
        """ Computes the inverse of the provided matrix.

        Args:
            matrix, `Matrix3`, The matrix to invert.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
        """
        m11 = matrix[0]
        m21 = matrix[1]
        m31 = matrix[2]
        m12 = matrix[3]
        m22 = matrix[4]
        m32 = matrix[5]
        m13 = matrix[6]
        m23 = matrix[7]
        m33 = matrix[8]

        determinant = Matrix3.determinant(matrix)

        if (abs(determinant) <= EPSILON15):
            raise Exception("matrix is not invertible")

        result[0] = m22 * m33 - m23 * m32
        result[1] = m23 * m31 - m21 * m33
        result[2] = m21 * m32 - m22 * m31
        result[3] = m13 * m32 - m12 * m33
        result[4] = m11 * m33 - m13 * m31
        result[5] = m12 * m31 - m11 * m32
        result[6] = m12 * m23 - m13 * m22
        result[7] = m13 * m21 - m11 * m23
        result[8] = m11 * m22 - m12 * m21

        scale = 1.0 / determinant
        return Matrix3.multiplyByScalar(result, scale, result)

    @staticmethod
    def inverseTranspose(matrix, result):
        """ Computes the inverse transpose of a matrix.

        Args:
            matrix, `Matrix3`, The matrix to transpose and invert.
            result: `Matrix3`, The object onto which to store the result.

        Returns:
            A `Matrix3`, The modified result parameter.
        """
        return Matrix3.inverse(
            Matrix3.transpose(matrix, _scratchTransposeMatrix),
            result
        )

    @staticmethod
    def equals(left, right):
        """ Compares the provided matrices componentwise and return `True`
        if they pass an absolute or relative tolerance test, `False` otherwise.

        Args:
            left, `Matrix3`, The first Cartesian.
            right: `Matrix3`, The second Cartesian.
            relativeEpsilon, `float`, The relative epsilon tolerance to use for equality testing. default=0
            absoluteEpsilon, `float`, The absolute epsilon tolerance to use for equality testing. default=0

        Returns:
            A `boolean`, `True` if they pass an absolute or relative tolerance test, `False` otherwise.
        """
        return (
            left is right or
            (left is not None and
                right is not None and
                left[0] == right[0] and
                left[1] == right[1] and
                left[2] == right[2] and
                left[3] == right[3] and
                left[4] == right[4] and
                left[5] == right[5] and
                left[6] == right[6] and
                left[7] == right[7] and
                left[8] == right[8])
        )

    @staticmethod
    def equalsEpsilon(left, right, epsilon=0):
        """ Compares the provided matrices componentwise and return `True`
        if they pass an absolute or relative tolerance test, `False` otherwise.

        Args:
            left, `Matrix3`, The first Cartesian.
            right: `Matrix3`, The second Cartesian.
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
                abs(left[8] - right[8]) <= epsilon)
        )

    @staticmethod
    def IDENTITY():
        """ A Matrix3 initialized to the identity matrix."""
        return Matrix3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    @staticmethod
    def ZERO():
        """ A Matrix3 initialized to the zero matrix."""
        return Matrix3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    COLUMN0ROW0 = 0
    COLUMN0ROW1 = 1
    COLUMN0ROW2 = 2
    COLUMN1ROW0 = 3
    COLUMN1ROW1 = 4
    COLUMN1ROW2 = 5
    COLUMN2ROW0 = 6
    COLUMN2ROW1 = 7
    COLUMN2ROW2 = 8


_scaleScratch1 = Cartesian3()
_scaleScratch2 = Cartesian3()
_scaleScratch3 = Cartesian3()
_scaleScratch4 = Cartesian3()
_scaleScratch5 = Cartesian3()
_scratchColumn = Cartesian3()
_scratchTransposeMatrix = Matrix3()
_jMatrix = Matrix3()
_jMatrixTranspose = Matrix3()

_rowVal = [1, 0, 0]
_colVal = [2, 2, 1]
