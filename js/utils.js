/**
 * @description Activation function that squashes the input between 0..1. Recommended for binary classification problems, where the output is either 0 or 1.
 * @param {number} x The value to squash
 * @returns {number} Returns the squashed value
 * @see https://en.wikipedia.org/wiki/Sigmoid_function
 */
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/**
 * @description Activation function (rectifier linear unit) that squashes the input either to to 0 or x. Recommended if the problem or data exhibits non-linear relationships and the outputs should be positive.
 * @param {number} x The value to rectify
 * @returns {number} The rectified value
 * @see https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
function reLU(x) {
  return x > 0 ? x : 0;
}

/**
 * @description Activation function that squashes the input between -1..1. Recommended if the problem or data exhibits non-linear relationships and the output should be bounded to some range.
 * @param {number} x The value to squash
 * @returns {number} The squashed value
 * @see https://en.wikipedia.org/wiki/Hyperbolic_functions
 */
function tanh(x) {
  return (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1);
}

/**
 * @description Calculate the mean squared error value. Recommended for regression tasks, where the goal is to predict continuous values, or if the output data is continuous and the magnitude of the errors is important.
 * @param {Array} predictions The vector of predicted values
 * @param {Array} targets The vector of observed values
 * @returns {number} Returns the average of the squares of the errors
 * @see https://en.wikipedia.org/wiki/Mean_squared_error
 */
function mse(predictions, targets) {
  let error = 0;
  for (let i = 0; i < predictions.length; i++) {
    const diff = targets[i] - predictions[i];
    error += Math.pow(diff, 2);
  }
  error /= predictions.length;

  return error;
}

/**
 * @description Calculate the dissimilarity between the predicted probability distribution and the true binary labels. Recommended for binary classification problems, or if the output data is binary or multi-class probabilities.
 * @param {*} yTrue The true binary labels
 * @param {*} yPred The predicted probabilities for the positive class
 * @returns {number} Returns the average cross-entropy over the dataset
 * @see https://en.wikipedia.org/wiki/Cross_entropy
 */
function crossEntropy(yTrue, yPred) {
  const epsilon = 1e-7; // Small constant to prevent log(0) and division by zero

  const loss = yTrue.map((y_true, i) => {
    const y_pred = yPred[i];
    const positiveTerm = y_true * Math.log(y_pred + epsilon);
    const negativeTerm = (1 - y_true) * Math.log(1 - y_pred + epsilon);
    return -(positiveTerm + negativeTerm);
  });

  return loss.reduce((sum, val) => sum + val, 0) / loss.length;
}

/**
 * @description Calculates the distance between two points in a n-dimensional Euclidean space
 * @param {Object|Array} a The first point
 * @param {Object|Array} b The second point
 * @returns {number} Returns the Euclidean distance between the two points
 * @see https://en.wikipedia.org//wiki/Euclidean_distance
 */
function euclideanDistance(a, b) {
  const squared_dist = Object.keys(a).reduce(
    (carry, key) => carry + (a[key] - b[key]) ** 2,
    0
  );

  return Math.sqrt(squared_dist);
}

/**
 * @description Calculates a fraction which represents how far through a value is between two border values.
 * @param {number} from The first input
 * @param {number} to The second input
 * @param {number} value The inverse interpolated value
 * @returns {number} Returns a value between zero and one, representing where the `value` parameter falls within the range defined by `from` and `to`.
 * @see https://en.wikipedia.org/wiki/Linear_interpolation
 */
function inverseLerp(from, to, value) {
  return (value - from) / (to - from);
}

/**
 * @description Calculate the min and the max points for a given serie of points
 * @param {Array} points An array of points of in n-dimensional Euclidiean space
 * @returns {Object} Returns the min|max points of the given serie
 */
function minMaxPoints(points) {
  // Initialize an array to store the minimum and maximum values of each feature
  let minMax = [];
  // Loop through each feature
  for (let i = 0; i < points[0].length; i++) {
    // Initialize the minimum and maximum values to the first value of the feature
    let min = points[0][i];
    let max = points[0][i];

    // Loop through each point in the data
    for (let j = 0; j < points.length; j++) {
      // Update the minimum and maximum values if needed
      if (points[j][i] < min) {
        min = points[j][i];
      }
      if (points[j][i] > max) {
        max = points[j][i];
      }
    }

    // Add the minimum and maximum values to the minMax array
    minMax.push([
      Math.min(...points.map(v => v[i])),
      Math.max(...points.map(v => v[i]))
    ]);
  }

  return minMax;
}

/**
 * @description Normalizes the points between 0..1
 * @param {Array} data An array of points of in n-dimensional Euclidiean space
 * @param {Object} minMax The min|max points for the given points. If not given it is automatically calculated.
 * @returns {Array} Returns the normalized points
 */
function normalizePoints(data, minMax) {
  const result = [];

  // Initialize an array to store the minimum and maximum values of each feature
  let _minMax = minMax || minMaxPoints(data);

  // Loop through each point in the data
  for (let i = 0; i < data.length; i++) {
    // Loop through each feature
    result[i] = [];
    for (let j = 0; j < data[i].length; j++) {
      // Normalize the feature value using the formula: (value - min) / (max - min)
      result[i][j] = inverseLerp(_minMax[j][0], _minMax[j][1], data[i][j]);
    }
  }

  return result;
}

/**
 * @description Calculates the nearest k-points to a given point
 * @param {Array} point The sample point
 * @param {Array} points The list of points to look into
 * @param {number} [k=1] The number of points to return
 * @param {Function} [distanceMetric=euclideanDistance] The function to calculate the distance
 * @returns {Array} Returns the the nearest k-points and their distance
 */
function nearestPoints(point, points, k = 1, distanceMetric) {
  // Calculate distances between features and training set data
  return (
    points
      .map((p, index) => ({
        index,
        point: p,
        distance: (distanceMetric || euclideanDistance)(point, p)
      }))
      // Sort distances in ascending order
      .sort((a, b) => a.distance - b.distance)
      // Retrieve the k nearest neighbors
      .slice(0, k)
  );
}

const activationFunctions = { sigmoid, reLU, tanh };
const errorFunctions = { mse, crossEntropy };

module.exports = {
  activationFunctions,
  errorFunctions,
  nearestPoints,
  minMaxPoints,
  normalizePoints,
  euclideanDistance,
  inverseLerp
};
