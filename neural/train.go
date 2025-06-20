package neural

import (
	"fmt"
	"math/rand"
	"time"
)

// --- Буфер опыта для DQN (оставляем без изменений) ---

// Experience представляет один игровой опыт.
type Experience struct {
	State     []float64
	Action    int
	Reward    float64
	NextState []float64
	Done      bool
}

// ReplayBuffer хранит игровой опыт.
type ReplayBuffer struct {
	Experiences []Experience
	Capacity    int
	Index       int
	Size        int
}

// NewReplayBuffer создает новый буфер опыта.
func NewReplayBuffer(capacity int) *ReplayBuffer {
	return &ReplayBuffer{
		Experiences: make([]Experience, capacity),
		Capacity:    capacity,
	}
}

// Add добавляет новый опыт в буфер.
func (rb *ReplayBuffer) Add(exp Experience) {
	rb.Experiences[rb.Index] = exp
	rb.Index = (rb.Index + 1) % rb.Capacity
	if rb.Size < rb.Capacity {
		rb.Size++
	}
}

// Sample выбирает случайный батч опыта из буфера.
func (rb *ReplayBuffer) Sample(batchSize int) []Experience {
	if rb.Size < batchSize {
		return nil // Недостаточно опыта для выборки батча
	}

	samples := make([]Experience, batchSize)
	for i := 0; i < batchSize; i++ {
		idx := rand.Intn(rb.Size)
		samples[i] = rb.Experiences[idx]
	}
	return samples
}

func MainTrain() {
	rand.Seed(time.Now().UnixNano()) // Инициализация генератора случайных чисел

	// Параметры обучения
	episodes := 10000         // Количество игровых эпизодов для обучения
	maxStepsPerEpisode := 100 // Максимальное количество шагов в эпизоде
	batchSize := 32           // Размер батча для обучения DQN
	bufferCapacity := 50000   // Емкость буфера опыта
	trainStartSize := 1000    // Начинать обучение после накопления достаточного опыта

	// Создаем агента DQN (играет за X)
	dqnAgentX := NewDQNAgent(9, 9, bufferCapacity, PlayerX)
	//dqnAgentO := NewDQNAgent(9, 9, bufferCapacity, PlayerO) // Агент для O (может быть случайным или другим DQN)

	totalSteps := 0
	winsX := 0
	winsO := 0
	draws := 0

	fmt.Println("Начало обучения DQN агента для крестиков-ноликов...")

	for episode := 0; episode < episodes; episode++ {
		board := NewBoard()
		isDone := false
		currentStepInEpisode := 0

		for !isDone && currentStepInEpisode < maxStepsPerEpisode {
			var chosenAction int
			/*
				var agentPlaying *DQNAgent

				// Выбор агента в зависимости от текущего игрока
				if board.CurrentPlayer == PlayerX {
					agentPlaying = dqnAgentX
				} else {
					// Если играет оппонент (PlayerO), он может быть как DQN, так и случайным
					// Для простоты, здесь PlayerO играет случайно, но вы можете сделать его другим DQNAgent
					emptyCells := board.GetEmptyCells()
					if len(emptyCells) > 0 {
						chosenAction = emptyCells[rand.Intn(len(emptyCells))]
					} else {
						chosenAction = -1 // Нет доступных ходов, игра окончена ничьей
					}
				}
			*/

			// Агент X делает ход
			if board.CurrentPlayer == PlayerX {
				// Состояние доски до хода агента
				state := board.GetStateVector(dqnAgentX.PlayerSymbol)
				chosenAction = dqnAgentX.ChooseAction(board)

				if chosenAction == -1 { // Нет доступных ходов, игра окончена
					isDone = true
					break
				}

				// Делаем ход
				board.MakeMove(chosenAction)
				// Состояние доски после хода агента
				nextState := board.GetStateVector(dqnAgentX.PlayerSymbol)
				reward := board.GetReward(dqnAgentX.PlayerSymbol)

				isDone = board.IsGameOver()

				// Добавляем опыт в буфер агента X
				dqnAgentX.ReplayBuffer.Add(Experience{
					State:     state,
					Action:    chosenAction,
					Reward:    reward,
					NextState: nextState,
					Done:      isDone,
				})

				totalSteps++
				// Обучение агента X
				if dqnAgentX.ReplayBuffer.Size >= trainStartSize {
					dqnAgentX.Train(batchSize, totalSteps)
				}

				if isDone {
					break
				}
			} else { // Оппонент (случайный игрок) делает ход
				if chosenAction != -1 {
					board.MakeMove(chosenAction)
				} else {
					isDone = true
					break
				}
				isDone = board.IsGameOver()
				if isDone {
					// Если оппонент завершил игру, его ход тоже приводит к вознаграждению для агента X.
					// Это вознаграждение будет обработано при следующем обучении агента X.
				}
			}

			if !isDone {
				board.SwitchPlayer() // Переключаем игрока только если игра не закончилась
			}
			currentStepInEpisode++
		}

		// Завершение эпизода
		if board.CheckWin() {
			if board.CurrentPlayer == PlayerX {
				winsX++
			} else {
				winsO++
			}
		} else if board.IsBoardFull() {
			draws++
		}

		if (episode+1)%100 == 0 {
			fmt.Printf("Эпизод: %d, Победы X: %d, Победы O: %d, Ничьи: %d, Epsilon: %.4f\n",
				episode+1, winsX, winsO, draws, dqnAgentX.Epsilon)
			winsX = 0
			winsO = 0
			draws = 0
		}
	}

	fmt.Println("\nОбучение завершено.")
	fmt.Println("Тестирование агента...")

	// --- Тестирование обученного агента против случайного оппонента ---
	testGames := 100
	testWinsX := 0
	testDraws := 0
	testLossesX := 0

	// Устанавливаем epsilon на минимальное значение для тестирования
	dqnAgentX.Epsilon = 0.0

	for i := 0; i < testGames; i++ {
		board := NewBoard()
		isDone := false

		for !isDone {
			// Ход агента X
			if board.CurrentPlayer == PlayerX {
				action := dqnAgentX.ChooseAction(board)
				if action == -1 {
					isDone = true
					break // Доска заполнена, ничья
				}
				board.MakeMove(action)
			} else {
				// Ход случайного оппонента (O)
				emptyCells := board.GetEmptyCells()
				if len(emptyCells) == 0 {
					isDone = true
					break // Доска заполнена, ничья
				}
				randomAction := emptyCells[rand.Intn(len(emptyCells))]
				board.MakeMove(randomAction)
			}

			isDone = board.IsGameOver()
			if !isDone {
				board.SwitchPlayer()
			}
		}

		// Подсчет результатов тестовой игры
		if board.CheckWin() {
			if board.CurrentPlayer == PlayerX {
				testWinsX++
			} else {
				testLossesX++
			}
		} else if board.IsBoardFull() {
			testDraws++
		}
	}

	fmt.Printf("\nРезультаты тестирования (%d игр против случайного оппонента):\n", testGames)
	fmt.Printf("Победы агента X: %d\n", testWinsX)
	fmt.Printf("Поражения агента X: %d\n", testLossesX)
	fmt.Printf("Ничьи: %d\n", testDraws)

	// Пример игры после обучения
	fmt.Println("\nПример игры после обучения:")
	board := NewBoard()
	dqnAgentX.Epsilon = 0.0 // Убедиться, что агент играет оптимально

	for !board.IsGameOver() {
		board.PrintBoard()

		var action int
		if board.CurrentPlayer == PlayerX {
			fmt.Println("Ход X (Агент DQN):")
			action = dqnAgentX.ChooseAction(board)
		} else {
			fmt.Println("Ход O (Случайный игрок):")
			emptyCells := board.GetEmptyCells()
			action = emptyCells[rand.Intn(len(emptyCells))]
		}

		if action == -1 {
			fmt.Println("Нет доступных ходов. Ничья.")
			break
		}

		board.MakeMove(action)
		board.SwitchPlayer()
	}
	board.PrintBoard()
	if board.CheckWin() {
		fmt.Printf("Игра окончена! Игрок %s выиграл!\n", map[int]string{PlayerX: "X", PlayerO: "O"}[-board.CurrentPlayer])
	} else {
		fmt.Println("Игра окончена! Ничья!")
	}
}
