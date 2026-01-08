export class Neuron {
  constructor(weight = Math.random(), bias = Math.random()) {
    this.weight = weight;
    this.bias = bias;
    this.learningRate = 0.01;
  }

  predict(x) {
    return this.weight * x + this.bias;
  }

  loss(yPred, yTrue) {
    return Math.pow(yPred - yTrue, 2);
  }

  train(x, yTrue) {
    const yPred = this.predict(x);
    const error = yPred - yTrue;

    // Gradients
    const dWeight = 2 * error * x;
    const dBias = 2 * error;

    // Mise à jour
    this.weight -= this.learningRate * dWeight;
    this.bias -= this.learningRate * dBias;

    return this.loss(yPred, yTrue);
  }
}
