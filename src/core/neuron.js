import { leakyRelu, leakyReluDerivative, tanh, tanhDerivative, sigmoid, sigmoidDerivative } from "./activations.js";
import { random } from "./random.js";

export class Neuron {
  constructor(inputSize, activation = "leakyrelu", learningRate = 0.0025, optimizer = "sgd") {
    const scale = Math.sqrt(2 / inputSize);
    this.weights = Array.from(
      { length: inputSize },
      () => (random() * 2 - 1) * scale
    );
    this.bias = random() * 0.5 - 0.25;
    this.learningRate = Number(learningRate) || 0.0025;
    this.activation = activation;
    this.optimizer = optimizer;

    // Pour Adam et RMSprop
    this.m = Array(inputSize).fill(0); // Premier moment
    this.v = Array(inputSize).fill(0); // Second moment
    this.mBias = 0;
    this.vBias = 0;

    // Hyperparamètres Adam/RMSprop
    this.beta1 = 0.9;
    this.beta2 = 0.999;
    this.epsilon = 1e-8;

    // Pour la rétropropagation
    this.lastError = 0;
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

    // Protection contre les valeurs extrêmes
    if (!Number.isFinite(this.z)) {
      this.z = 0;
    }

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

  // error : dérivée de la perte par rapport à la sortie du neurone.
  // Si errorIsDelta est vrai, error est déjà la dérivée par rapport à z
  // (cas de la couche de sortie : sigmoïde + entropie croisée binaire,
  // softmax + entropie croisée, sortie linéaire), et la dérivée de
  // l'activation ne doit pas être appliquée une seconde fois.
  backward(error, iteration = 1, errorIsDelta = false) {
    if (!Number.isFinite(error)) {
      throw new Error("Erreur reçue NaN dans backward: " + error);
    }

    let delta;

    if (errorIsDelta) {
      delta = error;
    } else if (this.activation === "leakyrelu") {
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

    if (this.optimizer === "adam") {
      this.updateAdam(delta, iteration);
    } else if (this.optimizer === "rmsprop") {
      this.updateRMSprop(delta);
    } else {
      this.updateSGD(delta, clip);
    }
  }

  updateSGD(delta, clip) {
    this.weights = this.weights.map((w, i) => {
      const input = this.inputs[i];

      if (!Number.isFinite(input)) {
        console.warn("Input invalide dans backward:", input);
        return w;
      }

      const grad = delta * input;
      const clippedGrad = clip(grad, 5);
      const newW = w - this.learningRate * clippedGrad;

      if (!Number.isFinite(newW)) {
        console.warn("Poids devenu invalide:", newW);
        return w;
      }

      return newW;
    });

    this.bias -= this.learningRate * clip(delta, 5);
  }

  updateAdam(delta, iteration) {
    // Mise à jour des poids
    this.weights = this.weights.map((w, i) => {
      const input = this.inputs[i];
      if (!Number.isFinite(input)) return w;

      const grad = delta * input;

      // Clip gradient
      const clippedGrad = Math.max(-5, Math.min(5, grad));

      // Mise à jour des moments
      this.m[i] = this.beta1 * this.m[i] + (1 - this.beta1) * clippedGrad;
      this.v[i] = this.beta2 * this.v[i] + (1 - this.beta2) * clippedGrad * clippedGrad;

      const mHat = this.m[i] / (1 - Math.pow(this.beta1, iteration));
      const vHat = this.v[i] / (1 - Math.pow(this.beta2, iteration));

      const update = this.learningRate * mHat / (Math.sqrt(vHat) + this.epsilon);
      const newW = w - update;

      if (!Number.isFinite(newW)) return w;
      return newW;
    });

    const gradBias = delta;
    const clippedGradBias = Math.max(-5, Math.min(5, gradBias));

    this.mBias = this.beta1 * this.mBias + (1 - this.beta1) * clippedGradBias;
    this.vBias = this.beta2 * this.vBias + (1 - this.beta2) * clippedGradBias * clippedGradBias;

    const mHatBias = this.mBias / (1 - Math.pow(this.beta1, iteration));
    const vHatBias = this.vBias / (1 - Math.pow(this.beta2, iteration));

    const updateBias = this.learningRate * mHatBias / (Math.sqrt(vHatBias) + this.epsilon);
    this.bias -= updateBias;
  }

  updateRMSprop(delta) {
    const decay = 0.9;

    this.weights = this.weights.map((w, i) => {
      const input = this.inputs[i];
      if (!Number.isFinite(input)) return w;

      const grad = delta * input;
      const clippedGrad = Math.max(-5, Math.min(5, grad));

      this.v[i] = decay * this.v[i] + (1 - decay) * clippedGrad * clippedGrad;

      const update = this.learningRate * clippedGrad / (Math.sqrt(this.v[i]) + this.epsilon);
      const newW = w - update;

      if (!Number.isFinite(newW)) return w;
      return newW;
    });

    const gradBias = delta;
    const clippedGradBias = Math.max(-5, Math.min(5, gradBias));

    this.vBias = decay * this.vBias + (1 - decay) * clippedGradBias * clippedGradBias;
    const updateBias = this.learningRate * clippedGradBias / (Math.sqrt(this.vBias) + this.epsilon);
    this.bias -= updateBias;
  }
}
