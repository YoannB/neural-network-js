import { leakyRelu, leakyReluDerivative, tanh, tanhDerivative, sigmoid, sigmoidDerivative } from "./activations.js";

export class Neuron {
  constructor(inputSize, activation = "leakyrelu", learningRate = 0.003) {
    const scale = Math.sqrt(2 / inputSize);
    this.weights = Array.from(
      { length: inputSize },
      () => (Math.random() * 2 - 1) * scale
    );
    this.bias = Math.random() * 0.5 - 0.25;
    this.learningRate = Number(learningRate) || 0.003;
    this.activation = activation;
  }

  forward(inputs) {
    if (!Array.isArray(inputs)) {
      throw new Error(
        `Neuron.forward attend un tableau, reçu: ${typeof inputs} (${inputs})`
      );
    }

    if (inputs.length !== this.weights.length) {
      throw new Error(
        `Dimension mismatch: inputs=${inputs.length}, weights=${this.weights.length}`
      );
    }

    this.inputs = inputs;

    this.z =
      this.weights.reduce(
        (sum, w, i) => sum + w * inputs[i],
        0
      ) + this.bias;

    if (this.activation === "leakyrelu") {
      this.output = leakyRelu(this.z);
    } else if (this.activation === "tanh") {
      this.output = tanh(this.z);
    } else if (this.activation === "sigmoid") {
      this.output = sigmoid(this.z);
    } else {
      this.output = this.z;
    }

    return this.output;
  }

  backward(error) {
    if (!Number.isFinite(error)) {
      throw new Error("Erreur reçue NaN dans backward: " + error);
    }

    let delta;

    if (this.activation === "leakyrelu") {
      delta = error * leakyReluDerivative(this.z);
    } else if (this.activation === "tanh") {
      delta = error * tanhDerivative(this.z);
    } else if (this.activation === "sigmoid") {
      delta = error * sigmoidDerivative(this.z);
    } else {
      delta = error;
    }

    if (!Number.isFinite(delta)) {
      throw new Error("Delta non valide: " + delta);
    }

    const clip = (v, limit = 1) => Math.max(-limit, Math.min(limit, v));

    this.weights = this.weights.map((w, i) => {
      const input = this.inputs[i];

      if (!Number.isFinite(input)) {
        throw new Error("Input invalide dans backward: " + input);
      }

      const grad = delta * input;
      const clippedGrad = clip(grad, 5);

      const newW = w - this.learningRate * clippedGrad;

      if (!Number.isFinite(newW)) {
        throw new Error("Poids devenu invalide: " + newW);
      }

      return newW;
    });

    this.bias -= this.learningRate * clip(delta, 5);
  }
}
