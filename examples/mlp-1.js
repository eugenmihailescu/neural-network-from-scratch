const { activationFunctions, errorFunctions } = require("./js/utils");
const MultiLayerPerceptron = require("./js/MultiLayerPerceptron");

const layerSize = [
  [2, 2],
  [4, 2],
  [4, 4],
  [2, 4]
];
const options = {
  activationFunction: activationFunctions.sigmoid,
  errorFunction: errorFunctions.mse,
  learningRate: 0.01
};

const epochs = 1e6;

// Example training data (inputs and corresponding targets)
const trainingData = [
  { inputs: [0, 0], targets: [0, 0] },
  { inputs: [0, 1], targets: [0, 1] },
  { inputs: [1, 0], targets: [0, 1] },
  { inputs: [1, 1], targets: [1, 1] }
];

const start = new Date();
const callback = (iteration, averageError) => {
  // Print the current iteration and average error
  if (!(iteration % 1000)) {
    const now = new Date();
    const speed = Math.round(iteration / ((now - start) / 1000));
    process.stdout.clearLine();
    process.stdout.cursorTo(0);
    process.stdout.write(
      `| Iteration ${iteration} | Average Error = ${averageError} | Speed ${speed} epoch/s |`
    );

    //return averageError < 0.01;
  }
};

new MultiLayerPerceptron(layerSize, options).train(
  trainingData,
  epochs,
  callback
);

console.log();
