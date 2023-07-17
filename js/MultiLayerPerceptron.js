const { activationFunctions, errorFunctions } = require("./utils");
const NetworkLayer = require("./NetworkLayer");

/**
 * @description A multilayer perceptron neural network (MLP)
 * @class MultiLayerPerceptron
 * @see https://en.wikipedia.org/wiki/Multilayer_perceptron
 */
class MultiLayerPerceptron {
  #activationFunction;
  #errorFunction;
  #learningRate;
  #layers = [];

  /**
   *Creates an instance of MultiLayerPerceptron.
   * @param {Array} layerSizes The size (neurons,inputs) of each layer
   * @param {Object} [options={}] The network options ({activationFunction,errorFunction,learningRate})
   * @memberof MultiLayerPerceptron
   */
  constructor(layerSizes, options = {}) {
    this.#activationFunction =
      options.activationFunction || activationFunctions.reLU;
    this.#errorFunction = options.errorFunction || errorFunctions.mse;
    this.#learningRate = options.learningRate || 0.01;

    for (let i = 0; i < layerSizes.length; i++) {
      this.#layers.push(new NetworkLayer(...layerSizes[i]));
    }
  }

  /**
   * @description Feed forward the inputs through the network, activating each layer successively.
   * @param {Array} inputs The input data
   * @returns {Array} Returns the predictions of the network.
   * @memberof MultiLayerPerceptron
   */
  #predict(inputs) {
    let outputs = inputs;

    // Forward pass through each layer
    for (let i = 0; i < this.#layers.length; i++) {
      outputs = this.#layers[i].activate(outputs, this.#activationFunction);
    }

    return outputs;
  }

  /**
   * @description Calculate the loss/cost function value
   * @param {Array} predictions The vector of predicted values
   * @param {Array} targets The vector of observed values
   * @returns {number} Returns the error cost
   * @memberof MultiLayerPerceptron
   */
  #computeError(predictions, targets) {
    const cost = this.#errorFunction(predictions, targets);

    return cost;
  }

  /**
   * @description Optimize the weights/biases based on the error cost then back propagate the optimized values backwards (last to first layer)
   * @param {number} error The error cost
   * @memberof MultiLayerPerceptron
   */
  #backpropagate(targets, predictions) {
    for (let i = this.#layers.length - 1; i >= 0; i--) {
      const currentLayer = this.#layers[i];
      const nextLayer = this.#layers[i + 1];

      // Compute gradients for the current layer
      const gradients = [];
      if (i === this.#layers.length - 1) {
        // Output layer
        for (let j = 0; j < currentLayer.numNeurons; j++) {
          const activation = currentLayer.activations[j];
          const derivative = activation * (1 - activation); // Derivative of sigmoid activation function
          const gradient = (targets[j] - predictions[j]) * derivative;

          gradients.push(gradient);
        }
      } else if (i > 0) {
        // Hidden layers
        for (let j = 0; j < currentLayer.numNeurons; j++) {
          const activation = currentLayer.activations[j];
          const derivative = activation * (1 - activation); // Derivative of sigmoid activation function
          let gradient = 0;
          for (let k = 0; k < nextLayer.numNeurons; k++) {
            gradient += nextLayer.weights[k][j] * nextLayer.gradients[k];
          }
          gradient *= derivative;
          gradients.push(gradient);
        }
      }
      // else {
      // Input layer
      // (no gradient computation or weight/bias update)
      // }

      // Update weights and biases for the current layer (except first/input layer)
      if (i > 0) {
        const prevLayer = this.#layers[i - 1];
        for (let j = 0; j < currentLayer.numNeurons; j++) {
          for (let k = 0; k < prevLayer.numNeurons; k++) {
            const weightUpdate =
              this.#learningRate * gradients[j] * prevLayer.activations[k];
            currentLayer.weights[j][k] += weightUpdate;
          }
          const biasUpdate = this.#learningRate * gradients[j];
          currentLayer.biases[j] += biasUpdate;
        }

        currentLayer.gradients = gradients; // Store the gradients for the next iteration
      }
    }
  }

  /**
   * @description Iterate over each the training sample, predict the output, evaluate the error cost then backpropagate updating the weights/biases.
   * @param {Array} trainingData An array of training data where each element is an array
   * @param {number} epochs How many epochs the network will undergo
   * @param {Function} callback A callback function that takes the current interation number and the average error value. Returns true if training should stop.
   * @memberof MultiLayerPerceptron
   */
  train(trainingData, epochs, callback) {
    const cb = callback || (() => {});

    // Iterate over the training data for the specified number of iterations
    for (let iteration = 1; iteration <= epochs; iteration++) {
      let totalError = 0;

      // Iterate over each training sample

      for (let i = 0; i < trainingData.length; i++) {
        const inputs = trainingData[i].inputs;
        const targets = trainingData[i].targets;

        // Perform feedforward pass to get predictions
        const predictions = this.#predict(inputs);

        // Compute the error between predictions and targets
        const error = this.#computeError(predictions, targets);
        totalError += error;

        // Backpropagation: Update weights and biases based on the error
        this.#backpropagate(targets, predictions);
      }

      // Calculate the average error for the current iteration
      const averageError = totalError / trainingData.length;

      // trigger the callback
      const shoudlStop = cb(iteration, averageError);

      if (shoudlStop) {
        break;
      }
    }
  }
}

module.exports = MultiLayerPerceptron;
