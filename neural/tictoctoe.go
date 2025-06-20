package neural

import "fmt"

// --- Игровая логика крестиков-ноликов (оставляем без изменений) ---

const (
	Empty   = 0
	PlayerX = 1
	PlayerO = -1
)

// Board представляет игровое поле крестиков-ноликов.
type Board struct {
	Cells         [9]int // 0: пусто, 1: X, -1: O
	CurrentPlayer int    // 1 для X, -1 для O
}

// NewBoard создает новую пустую доску.
func NewBoard() *Board {
	return &Board{
		Cells:         [9]int{Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty},
		CurrentPlayer: PlayerX, // X всегда начинает
	}
}

// MakeMove пытается сделать ход на указанной позиции.
// Возвращает true, если ход был успешным, false в противном случае.
func (b *Board) MakeMove(pos int) bool {
	if pos < 0 || pos >= 9 || b.Cells[pos] != Empty {
		return false
	}
	b.Cells[pos] = b.CurrentPlayer
	return true
}

// CheckWin проверяет, выиграл ли текущий игрок.
func (b *Board) CheckWin() bool {
	winConditions := [][]int{
		{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // Горизонтали
		{0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // Вертикали
		{0, 4, 8}, {2, 4, 6}, // Диагонали
	}

	for _, cond := range winConditions {
		if b.Cells[cond[0]] == b.CurrentPlayer &&
			b.Cells[cond[1]] == b.CurrentPlayer &&
			b.Cells[cond[2]] == b.CurrentPlayer {
			return true
		}
	}
	return false
}

// IsBoardFull проверяет, заполнена ли доска.
func (b *Board) IsBoardFull() bool {
	for _, cell := range b.Cells {
		if cell == Empty {
			return false
		}
	}
	return true
}

// IsGameOver проверяет, завершена ли игра (выигрыш или ничья).
func (b *Board) IsGameOver() bool {
	return b.CheckWin() || b.IsBoardFull()
}

// GetReward возвращает вознаграждение для агента, который только что сделал ход.
// player: игрок, который только что сделал ход.
func (b *Board) GetReward(agentPlayer int) float64 {
	if b.CheckWin() {
		if b.CurrentPlayer == agentPlayer {
			return 1.0 // Выигрыш
		} else {
			return -1.0 // Проигрыш
		}
	}
	if b.IsBoardFull() {
		return 0.0 // Ничья
	}
	return -0.01 // Небольшое отрицательное вознаграждение за каждый шаг (для стимулирования быстрой игры)
}

// GetStateVector преобразует состояние доски в вектор для нейронной сети.
// Представляем доску 3x3 как плоский вектор из 9 элементов.
// 1.0 для клетки агента, -1.0 для клетки оппонента, 0.0 для пустой.
func (b *Board) GetStateVector(agentPlayer int) []float64 {
	state := make([]float64, 9)
	for i, cell := range b.Cells {
		if cell == agentPlayer {
			state[i] = 1.0
		} else if cell == -agentPlayer { // Оппонент
			state[i] = -1.0
		} else { // Пусто
			state[i] = 0.0
		}
	}
	return state
}

// GetEmptyCells возвращает список индексов пустых клеток.
func (b *Board) GetEmptyCells() []int {
	var emptyCells []int
	for i, cell := range b.Cells {
		if cell == Empty {
			emptyCells = append(emptyCells, i)
		}
	}
	return emptyCells
}

// SwitchPlayer переключает текущего игрока.
func (b *Board) SwitchPlayer() {
	b.CurrentPlayer = -b.CurrentPlayer
}

// PrintBoard выводит доску в консоль.
func (b *Board) PrintBoard() {
	fmt.Println("-------------")
	for i := 0; i < 3; i++ {
		fmt.Print("| ")
		for j := 0; j < 3; j++ {
			val := b.Cells[i*3+j]
			switch val {
			case PlayerX:
				fmt.Print("X")
			case PlayerO:
				fmt.Print("O")
			case Empty:
				fmt.Print(" ")
			}
			fmt.Print(" | ")
		}
		fmt.Println("\n-------------")
	}
}
