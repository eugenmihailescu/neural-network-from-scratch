# Neural Network from scratch (no external dependency)

A set of neural network classes implemented from scratch for educational purpose only. It doesn't require anything but an ES6 Javascript engine (eg. web browser, Node.js 12 engine).

# Algorithms

1. Multi-Layer Perceptron (MLP)

   This algorithm is implemented by the `js/NetworkLayer.js` and `js/MultiLayerPerceptron.js` classes. The only dependencies are internal `js/utils.js`.
   
   It accepts as input parameters the activation and error functions, the learning rate, epochs and a variable number of neurons/inputs and layers. And obviously a training dataset to learn from.

3. K-Nearest Neighbors Classifier (KNN)

   This algorithm is implemented by the `js/KNNClassifier.js` class. The only dependencies are internal `js/utils.js`.

   It accepts as input parameters the distance metric functions, the k-value, the task type (eg. regression, classification), the inputs features normalization. And obviously a training dataset to learn from.

5. Examples

   To see it at work run the examples provided in the `examples` directory.

> Usage example:
> `node examples/mlp-1.js`
