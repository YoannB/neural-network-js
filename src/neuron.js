import { relu, reluDerivative } from "./activations.js";

export class Neuron {
  constructor(inputSize) {
    this.weights = Array.from(
      { length: inputSize },
      () => Math.random() - 0.5
    );
    this.bias = Math.random() - 0.5;
    this.learningRate = 0.01;
  }

  forward(inputs) {
    this.inputs = inputs;

    this.z =
      this.weights.reduce(
        (sum, w, i) => sum + w * inputs[i],
        0
      ) + this.bias;

    this.output = relu(this.z);
    return this.output;
  }

  backward(error) {
    const dActivation = reluDerivative(this.z);
    const delta = error * dActivation;

    this.weights = this.weights.map(
      (w, i) => w - this.learningRate * delta * this.inputs[i]
    );

    this.bias -= this.learningRate * delta;
  }
}
