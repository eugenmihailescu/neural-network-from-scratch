const KNNClassifier = require("../js/KNNClassifier");

const trainingFeatures = [
  [5.1, 3.5, 1.4, 0.2],
  [4.9, 3.0, 1.4, 0.2],
  [4.7, 3.2, 1.3, 0.2],
  [7.0, 3.2, 4.7, 1.4],
  [6.4, 3.2, 4.5, 1.5],
  [6.9, 3.1, 4.9, 1.5],
  [6.3, 3.3, 6.0, 2.5],
  [5.8, 2.7, 5.1, 1.9],
  [7.1, 3.0, 5.9, 2.1]
];
const trainingLabels = [
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica"
];

const testFeatures = [6.0, 3.0, 4.8, 1.8];

const prediction = new KNNClassifier(3)
  .withRegression(false)
  .withNormalization(true)
  .train(trainingFeatures, trainingLabels)
  .predict(testFeatures);

console.log(`Predicted label: ${JSON.stringify(prediction)}`);
