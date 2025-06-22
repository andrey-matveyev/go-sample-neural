package main

import (
	"fmt"
	"main/neural"
)

func main() {

	// Define XOR data
	// Inputs: [0,0], [0,1], [1,0], [1,1]
	// Outputs: [0], [1], [1], [0]
	xorInputs := [][]float64{
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 1.0},
	}
	xorOutputs := [][]float64{
		{0.0},
		{1.0},
		{1.0},
		{0.0},
	}

	// Neural Network architecture for XOR
	// 2 inputs, 1 hidden layer with 2 neurons, 1 output, sigmoid activation
	NeuralNetwork := neural.NewNeuralNetwork(2, []int{2}, 1, "sigmoid")
	
	// Test the non-trained network
	fmt.Println("Testing the non-trained network:")
	for i, input := range xorInputs {
		predictedOutput := NeuralNetwork.Predict(input)
		fmt.Printf("Input: %v, Expected: %v, Predicted: %.4f\n", input, xorOutputs[i], predictedOutput)
	}

	fmt.Println("Starting XOR training...")

	learningRate := 0.01
	epochs := 20000 // Number of training epochs

	for i := 0; i < epochs; i++ {
		totalLoss := 0.0
		for j := range xorInputs {
			input := xorInputs[j]
			target := xorOutputs[j]

			// Train the network on one XOR example
			NeuralNetwork.Train(input, target, learningRate)

			// Calculate current loss for monitoring (optional, but good practice)
			predicted := NeuralNetwork.Predict(input)
			loss := 0.0
			for k := range predicted {
				diff := predicted[k] - target[k]
				loss += diff * diff // MSE
			}
			totalLoss += loss
		}

		if (i+1)%1000 == 0 {
			fmt.Printf("Epoch %d, Average Loss: %.6f\n", i+1, totalLoss/float64(len(xorInputs)))
		}
	}

	fmt.Println("XOR training finished.")
	fmt.Println("Testing the trained network:")

	// Test the trained network
	for i, input := range xorInputs {
		predictedOutput := NeuralNetwork.Predict(input)
		fmt.Printf("Input: %v, Expected: %v, Predicted: %.4f\n", input, xorOutputs[i], predictedOutput)
	}
}
