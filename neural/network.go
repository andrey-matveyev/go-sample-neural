package neural

// --- Структуры нейронной сети (обновленная) ---

// NeuralNetwork представляет полную нейронную сеть.
type NeuralNetwork struct {
	Layers []*NeuralNetworkLayer // Теперь содержит объединенные слои
}

// NewNeuralNetwork создает новую нейронную сеть с объединенными слоями.
func NewNeuralNetwork(inputSize int, hiddenSizes []int, outputSize int, activation string) *NeuralNetwork {
	nn := &NeuralNetwork{}

	currentInputSize := inputSize
	for _, hs := range hiddenSizes {
		// Каждый скрытый слой теперь - это один NeuralNetworkLayer
		nn.Layers = append(nn.Layers, NewNeuralNetworkLayer(currentInputSize, hs, activation))
		currentInputSize = hs
	}

	// Выходной слой без активации (или с "none" активацией)
	nn.Layers = append(nn.Layers, NewNeuralNetworkLayer(currentInputSize, outputSize, "none"))

	return nn
}

// Predict выполняет прямой проход для получения предсказаний сети.
func (nn *NeuralNetwork) Predict(input []float64) []float64 {
	output := input
	for _, layer := range nn.Layers {
		output = layer.Forward(output)
	}
	return output
}

// Train выполняет один шаг обучения сети.
// input: входные данные
// targetOutput: целевые выходные данные (Q-значения для обучения)
// learningRate: скорость обучения
func (nn *NeuralNetwork) Train(input []float64, targetOutput []float64, learningRate float64) {
	// Прямой проход (сохранение промежуточных значений)
	predictedOutput := nn.Predict(input)

	// Вычисление градиента по выходу (MSE loss derivative)
	// dLoss/dOutput = 2 * (predicted - target)
	outputGradient := make([]float64, len(predictedOutput))
	for i := range predictedOutput {
		outputGradient[i] = 2 * (predictedOutput[i] - targetOutput[i])
	}

	// Обратный проход
	currentGradient := outputGradient
	for i := len(nn.Layers) - 1; i >= 0; i-- {
		currentGradient = nn.Layers[i].Backward(currentGradient)
	}

	// Обновление весов
	for _, layer := range nn.Layers {
		layer.Update(learningRate)
	}
}

// Clone создает глубокую копию нейронной сети.
func (nn *NeuralNetwork) Clone() *NeuralNetwork {
	clone := &NeuralNetwork{
		Layers: make([]*NeuralNetworkLayer, len(nn.Layers)),
	}
	for i, layer := range nn.Layers {
		newLayer := &NeuralNetworkLayer{
			InputSize:      layer.InputSize,
			OutputSize:     layer.OutputSize,
			Weights:        make([][]float64, len(layer.Weights)),
			Biases:         make([]float64, len(layer.Biases)),
			ActivationFunc: layer.ActivationFunc,
			DerivativeFunc: layer.DerivativeFunc,
		}
		for r := range layer.Weights {
			newLayer.Weights[r] = make([]float64, len(layer.Weights[r]))
			copy(newLayer.Weights[r], layer.Weights[r])
		}
		copy(newLayer.Biases, layer.Biases)
		clone.Layers[i] = newLayer
	}
	return clone
}
