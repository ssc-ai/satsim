import math
from satsim.vecmath import Cartesian2


class Matrix2:
    """ A 2x2 matrix, indexable as a column-major order array.
    Constructor parameters are in row-major order for code readability.
    """

    packedLength = 4

    def __init__(self, column0Row0=0.0, column1Row0=0.0, column0Row1=0.0, column1Row1=0.0):
        """ Constructor.

        Args:
            column0Row0: `float`, The value for column 0, row 0.
            column1Row0: `float`, The value for column 1, row 0.
            column0Row1: `float`, The value for column 0, row 1.
            column1Row1: `float`, The value for column 1, row 1.
        """
        self.m = [column0Row0, column0Row1, column1Row0, column1Row1]

    def __getitem__(self, key):
        return self.m[key]

    def __setitem__(self, key, value):
        self.m[key] = value

    def __len__(self):
        return Matrix2.packedLength

    def __str__(self):
        return f"""[{self[0]}, {self[2]}]
[{self[1]}, {self[3]}]"""

    def __eq__(self, other):
        return (
            self is other or
            (self is not None and
             other is not None and
             self[0] == other[0] and
             self[1] == other[1] and
             self[2] == other[2] and
             self[3] == other[3])
        )

    @staticmethod
    def fromArray(array, startingIndex=0, result=None):
        """ Creates a Matrix2 from 4 consecutive elements in an array.

        Args:
            array: `list`, The array whose 4 consecutive elements correspond to the positions of the matrix.  Assumes column-major order.
            startingIndex: `int`, The offset into the array of the first element, which corresponds to the x component.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter or a new Matrix2 instance if one was not provided.
        """
        if result is None:
            result = Matrix2()

        result[0] = array[startingIndex]
        result[1] = array[startingIndex + 1]
        result[2] = array[startingIndex + 2]
        result[3] = array[startingIndex + 3]
        return result

    @staticmethod
    def clone(matrix=None, result=None):
        """ Duplicates a Matrix2 instance.

        Args:
            matrix: `Matrix2`, The cartesian to duplicate.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter or a new Matrix2 instance if one was not provided.
        """
        if matrix is None:
            return None

        if result is None:
            return Matrix2(matrix[0], matrix[2], matrix[1], matrix[3])

        result[0] = matrix[0]
        result[1] = matrix[1]
        result[2] = matrix[2]
        result[3] = matrix[3]
        return result

    @staticmethod
    def fromColumnMajorArray(values, result=None):
        """ Creates a Matrix2 from a column-major order array.
        The resulting matrix will be in column-major order.

        Args:
            array: `list`, The column-major order array.
            startingIndex: `int`, The offset into the array of the first element, which corresponds to the x component.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter or a new Matrix2 instance if one was not provided.
        """
        return Matrix2.clone(values, result)

    @staticmethod
    def fromRowMajorArray(values, result=None):
        """ Creates a Matrix2 from a row-major order array.
        The resulting matrix will be in column-major order.

        Args:
            array: `list`, The row-major order array.
            startingIndex: `int`, The offset into the array of the first element, which corresponds to the x component.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter or a new Matrix2 instance if one was not provided.
        """
        if result is None:
            return Matrix2(values[0], values[1], values[2], values[3])

        result[0] = values[0]
        result[1] = values[2]
        result[2] = values[1]
        result[3] = values[3]
        return result

    @staticmethod
    def fromScale(scale, result=None):
        """ Computes a Matrix2 instance representing a non-uniform scale.

        Args:
            scale: `Cartesian2`, The x and y scale factors.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter or a new Matrix2 instance if one was not provided.
        """
        if result is None:
            return Matrix2(scale.x, 0.0, 0.0, scale.y)

        result[0] = scale.x
        result[1] = 0.0
        result[2] = 0.0
        result[3] = scale.y
        return result

    @staticmethod
    def fromUniformScale(scale, result=None):
        """ Computes a Matrix2 instance representing a uniform scale.

        Args:
            scale: `float`, The uniform scale factor.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter or a new Matrix2 instance if one was not provided.
        """
        if result is None:
            return Matrix2(scale, 0.0, 0.0, scale)

        result[0] = scale
        result[1] = 0.0
        result[2] = 0.0
        result[3] = scale
        return result

    @staticmethod
    def fromRotation(angle, result=None):
        """ Creates a rotation matrix.

        Args:
            angle: `float`, The angle, in radians, of the rotation.  Positive angles are counterclockwise.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter or a new Matrix2 instance if one was not provided.
        """
        cosAngle = math.cos(angle)
        sinAngle = math.sin(angle)

        if result is None:
            return Matrix2(cosAngle, -sinAngle, sinAngle, cosAngle)

        result[0] = cosAngle
        result[1] = sinAngle
        result[2] = -sinAngle
        result[3] = cosAngle
        return result

    @staticmethod
    def toArray(matrix, result=None):
        """ Creates an Array from the provided Matrix2 instance.
        The array will be in column-major order.

        Args:
            matrix: `Matrix2`, The matrix to use.
            result: `list`, The Array onto which to store the result.

        Returns:
            A `list`, The modified Array parameter or a new Array instance if one was not provided.
        """
        if result is None:
            return [matrix[0], matrix[1], matrix[2], matrix[3]]

        result[0] = matrix[0]
        result[1] = matrix[1]
        result[2] = matrix[2]
        result[3] = matrix[3]
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
        return column * 2 + row

    @staticmethod
    def getColumn(matrix, index, result):
        """ Retrieves a copy of the matrix column at the provided index as a Cartesian2 instance.

        Args:
            matrix: `Matrix2`, The matrix to use.
            index: `int`, The zero-based index of the column to retrieve.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        startIndex = index * 2
        x = matrix[startIndex]
        y = matrix[startIndex + 1]

        result.x = x
        result.y = y
        return result

    @staticmethod
    def setColumn(matrix, index, cartesian, result):
        """ Computes a new matrix that replaces the specified column in the provided matrix with the provided Cartesian2 instance.

        Args:
            matrix: `Matrix2`, The matrix to use.
            index: `int`, The zero-based index of the column to retrieve.
            cartesian: `Cartesian2`, The Cartesian whose values will be assigned to the specified column.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter.
        """
        result = Matrix2.clone(matrix, result)
        startIndex = index * 2
        result[startIndex] = cartesian.x
        result[startIndex + 1] = cartesian.y
        return result

    @staticmethod
    def getRow(matrix, index, result):
        """ Retrieves a copy of the matrix row at the provided index as a Cartesian2 instance.

        Args:
            matrix: `Matrix2`, The matrix to use.
            index: `int`, The zero-based index of the row to retrieve.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        x = matrix[index]
        y = matrix[index + 2]

        result.x = x
        result.y = y
        return result

    @staticmethod
    def setRow(matrix, index, cartesian, result):
        """ Computes a new matrix that replaces the specified row in the provided matrix with the provided Cartesian2 instance.

        Args:
            matrix: `Matrix2`, The matrix to use.
            index: `int`, The zero-based index of the row to retrieve.
            cartesian: `Cartesian2`, The Cartesian whose values will be assigned to the specified row.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter.
        """
        result = Matrix2.clone(matrix, result)
        result[index] = cartesian.x
        result[index + 2] = cartesian.y
        return result

    @staticmethod
    def setScale(matrix, scale, result):
        """ Computes a new matrix that replaces the scale with the provided scale.
        This assumes the matrix is an affine transformation.

        Args:
            matrix: `Matrix2`, The matrix to use.
            scale: `Cartesian2`, The scale that replaces the scale of the provided matrix.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter.
        """
        existingScale = Matrix2.getScale(matrix, _scaleScratch1)
        scaleRatioX = scale.x / existingScale.x
        scaleRatioY = scale.y / existingScale.y

        result[0] = matrix[0] * scaleRatioX
        result[1] = matrix[1] * scaleRatioX
        result[2] = matrix[2] * scaleRatioY
        result[3] = matrix[3] * scaleRatioY

        return result

    @staticmethod
    def setUniformScale(matrix, scale, result):
        """ Computes a new matrix that replaces the scale with the provided uniform scale.
        This assumes the matrix is an affine transformation.

        Args:
            matrix: `Matrix2`, The matrix to use.
            scale: `float`, The uniform scale that replaces the scale of the provided matrix.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter.
        """
        existingScale = Matrix2.getScale(matrix, _scaleScratch2)
        scaleRatioX = scale / existingScale.x
        scaleRatioY = scale / existingScale.y

        result[0] = matrix[0] * scaleRatioX
        result[1] = matrix[1] * scaleRatioX
        result[2] = matrix[2] * scaleRatioY
        result[3] = matrix[3] * scaleRatioY

        return result

    @staticmethod
    def getScale(matrix, result):
        """ Extracts the non-uniform scale assuming the matrix is an affine transformation.

        Args:
            matrix: `Matrix2`, The matrix.
            result: `Cartesian2`, The object onto which to store the result.

        Returns:
            A `Cartesian2`, The modified result parameter.
        """
        result.x = Cartesian2.magnitude(
            Cartesian2.fromElements(matrix[0], matrix[1], _scratchColumn)
        )
        result.y = Cartesian2.magnitude(
            Cartesian2.fromElements(matrix[2], matrix[3], _scratchColumn)
        )
        return result

    @staticmethod
    def getMaximumScale(matrix):
        """ Computes the maximum scale assuming the matrix is an affine transformation.
        The maximum scale is the maximum length of the column vectors.

        Args:
            matrix: `Matrix2`, The matrix.

        Returns:
            A `float`, The maximum scale.
        """
        Matrix2.getScale(matrix, _scaleScratch3)
        return Cartesian2.maximumComponent(_scaleScratch3)

    @staticmethod
    def setRotation(matrix, rotation, result):
        """ Sets the rotation assuming the matrix is an affine transformation.

        Args:
            matrix: `Matrix2`, The matrix to use.
            rotation: `Matrix2`, The rotation matrix.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter.
        """
        scale = Matrix2.getScale(matrix, _scaleScratch4)

        result[0] = rotation[0] * scale.x
        result[1] = rotation[1] * scale.x
        result[2] = rotation[2] * scale.y
        result[3] = rotation[3] * scale.y

        return result

    @staticmethod
    def getRotation(matrix, result):
        """ Extracts the rotation matrix assuming the matrix is an affine transformation.

        Args:
            matrix: `Matrix2`, The matrix.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter.
        """
        scale = Matrix2.getScale(matrix, _scaleScratch5)

        result[0] = matrix[0] / scale.x
        result[1] = matrix[1] / scale.x
        result[2] = matrix[2] / scale.y
        result[3] = matrix[3] / scale.y

        return result

    @staticmethod
    def multiply(left, right, result):
        """ Computes the product of two matrices.

        Args:
            left: `Matrix2`, The first matrix.
            right: `Matrix2`, The second matrix.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter.
        """
        column0Row0 = left[0] * right[0] + left[2] * right[1]
        column1Row0 = left[0] * right[2] + left[2] * right[3]
        column0Row1 = left[1] * right[0] + left[3] * right[1]
        column1Row1 = left[1] * right[2] + left[3] * right[3]

        result[0] = column0Row0
        result[1] = column0Row1
        result[2] = column1Row0
        result[3] = column1Row1
        return result

    @staticmethod
    def add(left, right, result):
        """ Computes the sum of two matrices.

        Args:
            left: `Matrix2`, The first matrix.
            right: `Matrix2`, The second matrix.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter.
        """
        result[0] = left[0] + right[0]
        result[1] = left[1] + right[1]
        result[2] = left[2] + right[2]
        result[3] = left[3] + right[3]
        return result

    @staticmethod
    def subtract(left, right, result):
        """ Computes the difference of two matrices.

        Args:
            left: `Matrix2`, The first matrix.
            right: `Matrix2`, The second matrix.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter.
        """
        result[0] = left[0] - right[0]
        result[1] = left[1] - right[1]
        result[2] = left[2] - right[2]
        result[3] = left[3] - right[3]
        return result

    @staticmethod
    def multiplyByVector(matrix, cartesian, result):
        """ Computes the product of a matrix and a column vector.

        Args:
            matrix: `Matrix2`, The first matrix.
            cartesian: `Cartesian2`, The column.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter.
        """
        x = matrix[0] * cartesian.x + matrix[2] * cartesian.y
        y = matrix[1] * cartesian.x + matrix[3] * cartesian.y

        result.x = x
        result.y = y
        return result

    @staticmethod
    def multiplyByScalar(matrix, scalar, result):
        """ Computes the product of a matrix and a column vector.

        Args:
            matrix: `Matrix2`, The first matrix.
            scalar: `float`, The number to multiply by.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter.
        """
        result[0] = matrix[0] * scalar
        result[1] = matrix[1] * scalar
        result[2] = matrix[2] * scalar
        result[3] = matrix[3] * scalar
        return result

    @staticmethod
    def multiplyByScale(matrix, scale, result):
        """ Computes the product of a matrix times a (non-uniform) scale, as if the scale were a scale matrix.

        Args:
            matrix: `Matrix2`, The first matrix.
            scale: `Cartesian2`, The non-uniform scale on the right-hand side.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter.
        """
        result[0] = matrix[0] * scale.x
        result[1] = matrix[1] * scale.x
        result[2] = matrix[2] * scale.y
        result[3] = matrix[3] * scale.y

        return result

    @staticmethod
    def multiplyByUniformScale(matrix, scale, result):
        """ Computes the product of a matrix times a uniform scale, as if the scale were a scale matrix.

        Args:
            matrix: `Matrix2`, The first matrix.
            scale: `float`, The uniform scale on the right-hand side.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter.
        """
        result[0] = matrix[0] * scale
        result[1] = matrix[1] * scale
        result[2] = matrix[2] * scale
        result[3] = matrix[3] * scale

        return result

    @staticmethod
    def negate(matrix, result):
        """ Negates the provided matrix.

        Args:
            matrix, `Matrix2`, The matrix to be negated.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter.
        """
        result[0] = -matrix[0]
        result[1] = -matrix[1]
        result[2] = -matrix[2]
        result[3] = -matrix[3]
        return result

    @staticmethod
    def transpose(matrix, result):
        """ Computes the transpose of the provided matrix.

        Args:
            matrix, `Matrix2`, The matrix to be transpose.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter.
        """
        column0Row0 = matrix[0]
        column0Row1 = matrix[2]
        column1Row0 = matrix[1]
        column1Row1 = matrix[3]

        result[0] = column0Row0
        result[1] = column0Row1
        result[2] = column1Row0
        result[3] = column1Row1
        return result

    @staticmethod
    def abs(matrix, result):
        """ Computes a matrix, which contains the absolute (unsigned) values of the provided matrix's elements.

        Args:
            matrix, `Matrix2`, The matrix with signed elements.
            result: `Matrix2`, The object onto which to store the result.

        Returns:
            A `Matrix2`, The modified result parameter.
        """
        result[0] = abs(matrix[0])
        result[1] = abs(matrix[1])
        result[2] = abs(matrix[2])
        result[3] = abs(matrix[3])

        return result

    @staticmethod
    def equalsEpsilon(left, right, epsilon):
        """ Compares the provided matrices componentwise and return `True`
        if they pass an absolute or relative tolerance test, `False` otherwise.

        Args:
            left, `Matrix2`, The first Cartesian.
            right: `Matrix2`, The second Cartesian.
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
             abs(left[3] - right[3]) <= epsilon)
        )

    @staticmethod
    def IDENTITY():
        """ A Matrix2 initialized to the identity matrix."""
        return Matrix2(1.0, 0.0, 0.0, 1.0)

    @staticmethod
    def ZERO():
        """ A Matrix2 initialized to the zero matrix."""
        return Matrix2(0.0, 0.0, 0.0, 0.0)

    COLUMN0ROW0 = 0
    COLUMN0ROW1 = 1
    COLUMN1ROW0 = 2
    COLUMN1ROW1 = 3


_scratchColumn = Cartesian2()
_scaleScratch1 = Cartesian2()
_scaleScratch2 = Cartesian2()
_scaleScratch3 = Cartesian2()
_scaleScratch4 = Cartesian2()
_scaleScratch5 = Cartesian2()
