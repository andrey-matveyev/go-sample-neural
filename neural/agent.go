package neural

import (
	"fmt"
	"math"
	"math/rand"
)

// --- Агент DQN (обновленный для использования NeuralNetworkLayer) ---

// DQNAgent представляет агента глубокого Q-обучения.
type DQNAgent struct {
	QNetwork      *NeuralNetwork
	TargetNetwork *NeuralNetwork
	ReplayBuffer  *ReplayBuffer
	Gamma         float64 // Коэффициент дисконтирования
	Epsilon       float64 // Для эпсилон-жадной стратегии
	MinEpsilon    float64 // Минимальное значение эпсилон
	EpsilonDecay  float64 // Скорость уменьшения эпсилон
	LearningRate  float64
	UpdateTarget  int // Интервал обновления целевой сети
	PlayerSymbol  int // Символ, которым играет этот агент (PlayerX или PlayerO)
}

// NewDQNAgent создает нового агента DQN.
func NewDQNAgent(inputSize, outputSize, bufferCapacity int, playerSymbol int) *DQNAgent {
	// Теперь используем NewNeuralNetworkLayer внутри NewNeuralNetwork
	qNet := NewNeuralNetwork(inputSize, []int{64, 64}, outputSize, "relu") // Пример архитектуры
	targetNet := qNet.Clone()                                              // Клонируем для целевой сети

	return &DQNAgent{
		QNetwork:      qNet,
		TargetNetwork: targetNet,
		ReplayBuffer:  NewReplayBuffer(bufferCapacity),
		Gamma:         0.99,  // Дисконтный фактор
		Epsilon:       1.0,   // Начинаем с исследования
		MinEpsilon:    0.01,  // Минимальное значение эпсилон
		EpsilonDecay:  0.995, // Уменьшение эпсилон за эпизод
		LearningRate:  0.001,
		UpdateTarget:  1000, // Обновлять целевую сеть каждые 1000 шагов
		PlayerSymbol:  playerSymbol,
	}
}

// ChooseAction выбирает действие, используя эпсилон-жадную стратегию.
// board: текущее состояние доски.
func (agent *DQNAgent) ChooseAction(board *Board) int {
	emptyCells := board.GetEmptyCells()
	if len(emptyCells) == 0 {
		return -1 // Нет доступных ходов
	}

	// Эпсилон-жадная стратегия: случайный ход или лучший ход по Q-сети
	if rand.Float64() < agent.Epsilon {
		return emptyCells[rand.Intn(len(emptyCells))] // Случайный ход
	}

	// Выбираем лучший ход по Q-сети
	stateVec := board.GetStateVector(agent.PlayerSymbol)
	qValues := agent.QNetwork.Predict(stateVec)

	bestAction := -1
	maxQ := -math.MaxFloat64

	for _, action := range emptyCells {
		if qValues[action] > maxQ {
			maxQ = qValues[action]
			bestAction = action
		}
	}
	return bestAction
}

// Train выполняет один шаг обучения агента.
// batchSize: размер батча для обучения.
// step: текущий шаг (для обновления целевой сети).
func (agent *DQNAgent) Train(batchSize, step int) {
	batch := agent.ReplayBuffer.Sample(batchSize)
	if batch == nil {
		return // Недостаточно опыта
	}

	for _, exp := range batch {
		// Предсказанные Q-значения для текущего состояния
		currentQValues := agent.QNetwork.Predict(exp.State)
		targetQValues := make([]float64, len(currentQValues))
		copy(targetQValues, currentQValues) // Копируем, чтобы изменить только одно значение

		// Вычисляем целевое Q-значение
		var nextMaxQ float64
		if !exp.Done {
			// Предсказания целевой сети для следующего состояния
			nextQValues := agent.TargetNetwork.Predict(exp.NextState)
			// Находим максимальное Q-значение для следующего состояния (среди возможных ходов)
			nextMaxQ = -math.MaxFloat64
			for _, qVal := range nextQValues {
				if qVal > nextMaxQ {
					nextMaxQ = qVal
				}
			}
		}

		// Обновляем целевое Q-значение для выбранного действия
		if exp.Done {
			targetQValues[exp.Action] = exp.Reward // Если игра закончена, целевое значение = награда
		} else {
			targetQValues[exp.Action] = exp.Reward + agent.Gamma*nextMaxQ // Уравнение Беллмана
		}

		// Обучаем Q-сеть
		agent.QNetwork.Train(exp.State, targetQValues, agent.LearningRate)
	}

	// Уменьшаем эпсилон
	if agent.Epsilon > agent.MinEpsilon {
		agent.Epsilon *= agent.EpsilonDecay
	}

	// Обновляем целевую сеть
	if step%agent.UpdateTarget == 0 {
		agent.TargetNetwork = agent.QNetwork.Clone()
		fmt.Printf("--- Целевая сеть обновлена на шаге %d ---\n", step)
	}
}
