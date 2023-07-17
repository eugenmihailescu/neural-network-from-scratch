const { nearestPoints, normalizePoints } = require("./utils");

/**
 * @description k-Nearest neighbors classifier
 * @class KNNClassifier
 * @see https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
 */
class KNNClassifier {
  #distanceMetric;
  #normalization;
  #regression;
  #trainingFeatures;
  #trainingLabels;
  #k;

  constructor(k) {
    this.#k = k;
    this.#distanceMetric;
    this.#regression = false;
    this.#trainingFeatures = [];
    this.#trainingLabels = [];
    this.#normalization = false;
  }

  /**
   * @description Finds the nearest k-points from the given point
   * @param {Array} point The sample point
   * @returns {Array} Returns the the nearest k-points, their labels and distance
   * @memberof KNNClassifier
   */
  #findNearestNeighbors(point) {
    return nearestPoints(
      point,
      this.#trainingFeatures,
      this.#k,
      this.#distanceMetric
    ).map(point => ({
      ...point,
      label: this.#trainingLabels[point.index]
    }));
  }

  /**
   * @description Perform majority voting to determine the predicted class label
   * @param {Array} neighbors The first k-nearest neighbors
   * @returns {Object} Returns the majority voting for each label
   * @memberof KNNClassifier
   */
  #getVotes(neighbors) {
    return neighbors.reduce(
      (carry, neighbor) =>
        Object.assign(carry, {
          [neighbor.label]: (carry[neighbor.label] || 0) + 1
        }),
      {}
    );
  }

  /**
   * @description Calculate the average of target values of nearest neighbors
   * @param {Array} neighbors The first k-nearest neighbors
   * @returns {number} Returns the average of target values
   * @memberof KNNClassifier
   */
  #getAvgValues(neighbors) {
    return (
      neighbors.reduce((sum, neighbor) => sum + neighbor.label, 0) /
      neighbors.length
    );
  }

  /**
   * @description Loads the labeled training data
   * @param {Array} features The input features
   * @param {Array} labels The input features' labels
   * @returns {KNNClassifier}
   * @memberof KNNClassifier
   */
  train(features, labels) {
    // normalize eventually the features
    const _features = this.#normalization
      ? normalizePoints(features)
      : features;

    // Store the training data in the classifier
    for (let i = 0; i < _features.length; i++) {
      this.#trainingFeatures.push(_features[i]);
      this.#trainingLabels.push(labels[i]);
    }

    return this;
  }

  /**
   * @description Predicts the class label (or regression value) for the given data point/feature
   * @param {Array} feature The data point/feature
   * @returns {Object|number} Returns the predicted class label and votes on non-regression task, the average of target values otherwise
   * @memberof KNNClassifier
   */
  predict(feature) {
    // normalize eventually the point
    const point = this.#normalization
      ? normalizePoints([feature, ...this.#trainingFeatures])[0]
      : feature;

    // Find the k nearest neighbors based on distance metric
    const neighbors = this.#findNearestNeighbors(point);

    if (this.#regression) {
      return this.#getAvgValues(neighbors);
    }

    // Perform majority voting to determine the predicted class label
    const votes = this.#getVotes(neighbors);

    let maxVotes = 0;
    let predictedLabel;
    for (const label in votes) {
      if (votes[label] > maxVotes) {
        maxVotes = votes[label];
        predictedLabel = label;
      }
    }

    return { predictedLabel, maxVotes };
  }

  /**
   * @description Set the function to calculate the metric distance (default to `euclideanDistance`)
   * @param {Function} distanceMetric A function that calculates the distance between two points in a n-dimensional Euclidean space
   * @returns {KNNClassifier}
   * @memberof KNNClassifier
   */
  withDistanceMetric(distanceMetric) {
    this.#distanceMetric = distanceMetric;
    return this;
  }

  /**
   * @description Enable/disable the feature inputs normalization
   * @param {Boolean} normalization When true inputs feature are normalized on train/prediction
   * @returns {KNNClassifier}
   * @memberof KNNClassifier
   */
  withNormalization(normalization) {
    this.#normalization = normalization;
    return this;
  }

  /**
   * @description Set the task type (eg. regression/classification)
   * @param {Boolean} regression When true then regression task, otherwise classification task
   * @returns {KNNClassifier}
   * @memberof KNNClassifier
   */
  withRegression(regression) {
    this.#regression = regression;
    return this;
  }
}

module.exports = KNNClassifier;
