const fs = require("fs");
const path = require("path");
const KNNClassifier = require("../js/KNNClassifier");

function scanDirectoryForJSONFiles(directoryPath) {
  const files = fs.readdirSync(directoryPath);
  const jsonFiles = [];

  files.forEach(file => {
    const filePath = path.join(directoryPath, file);
    const stats = fs.statSync(filePath);

    if (stats.isFile() && path.extname(file) === ".json") {
      jsonFiles.push(filePath);
    } else if (stats.isDirectory()) {
      jsonFiles.push(...scanDirectoryForJSONFiles(filePath));
    }
  });

  return jsonFiles;
}

const extractFeatures = dir => {
  const raw_data = scanDirectoryForJSONFiles(dir).map(filename =>
    require(filename)
  );

  const all_drawings = raw_data.map(({ drawings }) => drawings);

  const samples = all_drawings
    .map(sample_drawings => {
      return Object.keys(sample_drawings).map(key => [
        [sample_drawings[key].length, sample_drawings[key].flat().length],
        key
      ]);
    })
    .flat();

  return samples;
};

const features = extractFeatures(
  "/home/eugen/workspace/AI/MachineLearning/drawing-data/data/raw"
);

const trainingFeatures = features.map(row => row[0]);
const trainingLabels = features.map(row => row[1]);

const testFeatures = [4, 188];

const prediction = new KNNClassifier(50)
  .withRegression(false)
  .withNormalization(true)
  .train(trainingFeatures, trainingLabels)
  .predict(testFeatures);

console.log(`Predicted label: ${JSON.stringify(prediction)}`);
