package neural

import (
	"math"
	"math/rand"
)

// --- Объединенная структура слоя нейронной сети ---

// NeuralNetworkLayer представляет один полносвязный слой с функцией активации.
type NeuralNetworkLayer struct {
	InputSize  int
	OutputSize int
	Weights    [][]float64 // Weights[output_neuron_idx][input_neuron_idx]
	Biases     []float64

	ActivationFunc func(float64) float64
	DerivativeFunc func(float64) float64

	// Временные значения для обратного распространения
	InputVector  []float64 // Входные значения в слой (от предыдущего слоя)
	WeightedSums []float64 // Значения после линейной трансформации (до активации)
	OutputVector []float64 // Выходные значения после активации

	// Градиенты для обновления весов и смещений
	WeightGradients [][]float64
	BiasGradients   []float64
	InputGradient   []float64 // Градиент, передаваемый на предыдущий слой
}

// NewNeuralNetworkLayer создает новый полносвязный слой с функцией активации.
// activationName может быть "relu", "sigmoid" или "none" для линейного слоя.
func NewNeuralNetworkLayer(inputSize, outputSize int, activationName string) *NeuralNetworkLayer {
	weights := make([][]float64, outputSize)
	biases := make([]float64, outputSize)
	for i := range weights {
		weights[i] = make([]float64, inputSize)
		for j := range weights[i] {
			// Инициализация весов случайными величинами
			weights[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(inputSize))
		}
		biases[i] = 0.0
	}

	layer := &NeuralNetworkLayer{
		InputSize:  inputSize,
		OutputSize: outputSize,
		Weights:    weights,
		Biases:     biases,
	}

	switch activationName {
	case "sigmoid":
		layer.ActivationFunc = Sigmoid
		layer.DerivativeFunc = SigmoidDerivative
	case "relu":
		layer.ActivationFunc = ReLU
		layer.DerivativeFunc = ReLUDerivative
	case "none": // Для выходного слоя без активации
		layer.ActivationFunc = func(x float64) float64 { return x }
		layer.DerivativeFunc = func(x float64) float64 { return 1.0 }
	default:
		panic("Неизвестная функция активации: " + activationName)
	}

	return layer
}

// Forward выполняет прямой проход через слой (линейная часть + активация).
func (item *NeuralNetworkLayer) Forward(input []float64) []float64 {
	item.InputVector = input
	// 1. Линейная трансформация
	item.WeightedSums = MultiplyMatrixVector(item.Weights, input)
	item.WeightedSums = AddVectors(item.WeightedSums, item.Biases)

	// 2. Активация
	item.OutputVector = make([]float64, len(item.WeightedSums))
	for i := range item.WeightedSums {
		item.OutputVector[i] = item.ActivationFunc(item.WeightedSums[i])
	}
	return item.OutputVector
}

// Backward выполняет обратный проход через слой.
func (item *NeuralNetworkLayer) Backward(outputGradient []float64) []float64 {
	// 1. Градиент через функцию активации (применяем производную активации к WeightedSums)
	activationGradient := make([]float64, len(item.WeightedSums))
	for i := range item.WeightedSums {
		activationGradient[i] = item.DerivativeFunc(item.WeightedSums[i])
	}
	// Совмещаем градиент от следующего слоя с градиентом активации
	gradientAfterActivation := MultiplyVectors(outputGradient, activationGradient)

	// 2. Градиент по смещениям равен градиенту после активации
	item.BiasGradients = gradientAfterActivation

	// 3. Градиент по весам = внешнее произведение (Input X gradientAfterActivation)
	item.WeightGradients = OuterProduct(gradientAfterActivation, item.InputVector)

	// 4. Градиент по входу = ТранспонированныеВеса * gradientAfterActivation
	transposedWeights := TransposeMatrix(item.Weights)
	item.InputGradient = MultiplyMatrixVector(transposedWeights, gradientAfterActivation)

	return item.InputGradient
}

// Update обновляет веса и смещения слоя.
func (item *NeuralNetworkLayer) Update(learningRate float64) {
	// Обновление весов
	for i := range item.Weights {
		for j := range item.Weights[i] {
			item.Weights[i][j] -= learningRate * item.WeightGradients[i][j]
		}
	}
	// Обновление смещений
	for i := range item.Biases {
		item.Biases[i] -= learningRate * item.BiasGradients[i]
	}
}
