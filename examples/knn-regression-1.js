const KNNClassifier = require("../js/KNNClassifier");

const trainingFeatures = [
  [1, 2],
  [2, 1],
  [1, 3],
  [3, 4]
];
const trainingLabels = [5, 10, 7, 12];

const testFeatures = [2, 2];

const prediction = new KNNClassifier(3)
  .withRegression(true)
  .withNormalization(true)
  .train(trainingFeatures, trainingLabels)
  .predict(testFeatures);

console.log(`Predicted label: ${JSON.stringify(prediction)}`);
