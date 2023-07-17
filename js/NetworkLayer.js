/**
 * @description A generic neuronal network layer
 * @class NetworkLayer
 */
class NetworkLayer {
  #numInputs;

  constructor(numNeurons, numInputs) {
    this.numNeurons = numNeurons;
    this.#numInputs = numInputs;
    this.weights = []; // Weight matrix
    this.biases = []; // Bias vector
    this.activations = []; // Activations vector
    this.gradients = []; // Gradients vector

    // Initialize weights and biases with random values
    for (let i = 0; i < numNeurons; i++) {
      this.weights[i] = [];
      for (let j = 0; j < numInputs; j++) {
        this.weights[i][j] = Math.random() * 2 - 1; // (-1..1), ie. centered at origin
      }

      this.biases[i] = Math.random() * 2 - 1; // (-1..1), ie. centered at origin
    }
  }

  /**
   * @description Computes the weighted sum of the inputs to each neuron in the layer. Introduce non-linearity by squashing/transforming the result by an activation function.
   * @param {Array} inputs The array of input values
   * @param {Function} activationFunction The activation function (see `activationFunctions`)
   * @returns {Array} The resulting output values
   * @memberof NetworkLayer
   */
  activate(inputs, activationFunction) {
    if (inputs.length !== this.#numInputs) {
      throw new Error("Input dimension mismatch");
    }

    const outputs = [];
    for (let i = 0; i < this.numNeurons; i++) {
      let activation = 0;

      for (let j = 0; j < this.#numInputs; j++) {
        activation += this.weights[i][j] * inputs[j];
      }
      activation += this.biases[i];

      // Apply activation function (e.g., sigmoid)
      const output = activationFunction(activation);

      outputs.push(output);
    }

    this.activations = outputs; // Store the activations

    return outputs;
  }
}

module.exports = NetworkLayer;
