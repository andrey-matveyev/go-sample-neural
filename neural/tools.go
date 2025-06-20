package neural

import "math"

// --- Утилиты для матричных операций (оставляем без изменений) ---

// AddVectors складывает два вектора поэлементно.
func AddVectors(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

// SubtractVectors вычитает один вектор из другого поэлементно.
func SubtractVectors(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return result
}

// MultiplyVectors поэлементно умножает два вектора.
func MultiplyVectors(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] * b[i]
	}
	return result
}

// MultiplyScalarVector умножает вектор на скаляр.
func MultiplyScalarVector(scalar float64, vector []float64) []float64 {
	result := make([]float64, len(vector))
	for i := range vector {
		result[i] = scalar * vector[i]
	}
	return result
}

// DotProduct выполняет скалярное произведение двух векторов.
func DotProduct(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// OuterProduct вычисляет внешнее произведение двух векторов.
func OuterProduct(a, b []float64) [][]float64 {
	rows := len(a)
	cols := len(b)
	result := make([][]float64, rows)
	for i := range result {
		result[i] = make([]float64, cols)
		for j := range result[i] {
			result[i][j] = a[i] * b[j]
		}
	}
	return result
}

// MultiplyMatrixVector умножает матрицу на вектор.
func MultiplyMatrixVector(matrix [][]float64, vector []float64) []float64 {
	result := make([]float64, len(matrix))
	for i := 0; i < len(matrix); i++ {
		result[i] = DotProduct(matrix[i], vector)
	}
	return result
}

// TransposeMatrix транспонирует матрицу.
func TransposeMatrix(matrix [][]float64) [][]float64 {
	rows := len(matrix)
	if rows == 0 {
		return [][]float64{}
	}
	cols := len(matrix[0])
	transposed := make([][]float64, cols)
	for i := range transposed {
		transposed[i] = make([]float64, rows)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			transposed[j][i] = matrix[i][j]
		}
	}
	return transposed
}

// --- Функции активации (оставляем без изменений) ---

// Sigmoid функция активации.
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidDerivative производная функции Sigmoid.
func SigmoidDerivative(x float64) float64 {
	s := Sigmoid(x)
	return s * (1 - s)
}

// ReLU функция активации.
func ReLU(x float64) float64 {
	return math.Max(0, x)
}

// ReLUDerivative производная функции ReLU.
func ReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1.0
	}
	return 0.0
}
