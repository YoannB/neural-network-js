import { relu, reluDerivative, leakyRelu, leakyReluDerivative, tanh, tanhDerivative, sigmoid, sigmoidDerivative } from "./activations.js";
import { random } from "./random.js";
import type { Activation, Optimizer } from "./types.js";

const GRADIENT_LIMIT = 5;
const clip = (v: number, limit = GRADIENT_LIMIT): number => Math.max(-limit, Math.min(limit, v));

export class Neuron {
  weights: number[];
  bias: number;
  readonly learningRate: number;
  readonly activation: Activation;
  readonly optimizer: Optimizer;
  /**
   * Pénalité sur les poids (weight decay) : ajoute l2 × w au gradient de chaque
   * poids et pousse les poids vers 0. Le biais n'est pas pénalisé.
   */
  readonly l2: number;

  // Moments d'Adam et RMSprop
  m: number[];
  v: number[];
  mBias = 0;
  vBias = 0;

  readonly beta1 = 0.9;
  readonly beta2 = 0.999;
  readonly epsilon = 1e-8;

  // État de la dernière propagation avant
  inputs: number[] = [];
  z = 0;
  output = 0;

  constructor(inputSize: number, activation: Activation = "leakyrelu", learningRate = 0.0025, optimizer: Optimizer = "sgd", l2 = 0) {
    const scale = Math.sqrt(2 / inputSize);
    this.weights = Array.from({ length: inputSize }, () => (random() * 2 - 1) * scale);
    this.bias = random() * 0.5 - 0.25;
    this.learningRate = Number(learningRate) || 0.0025;
    this.activation = activation;
    this.optimizer = optimizer;
    this.l2 = l2;
    this.m = new Array<number>(inputSize).fill(0);
    this.v = new Array<number>(inputSize).fill(0);
  }

  forward(inputs: number[]): number {
    if (inputs.length !== this.weights.length) {
      throw new Error(`Dimension mismatch: inputs=${inputs.length}, weights=${this.weights.length}`);
    }

    this.inputs = inputs;
    this.z = this.weights.reduce((sum, w, i) => sum + w * inputs[i], 0) + this.bias;

    // Protection contre les valeurs extrêmes
    if (!Number.isFinite(this.z)) {
      this.z = 0;
    }

    switch (this.activation) {
      case "relu": this.output = relu(this.z); break;
      case "leakyrelu": this.output = leakyRelu(this.z); break;
      case "tanh": this.output = tanh(this.z); break;
      case "sigmoid": this.output = sigmoid(this.z); break;
      // linear, et softmax (appliqué ensuite par la couche)
      default: this.output = this.z;
    }

    return this.output;
  }

  /**
   * Met à jour le neurone et renvoie son delta (dérivée de la perte par rapport à z).
   *
   * error : dérivée de la perte par rapport à la sortie du neurone. Si errorIsDelta
   * est vrai, error est déjà la dérivée par rapport à z (couche de sortie : sigmoïde
   * + entropie croisée binaire, softmax + entropie croisée, sortie linéaire), et la
   * dérivée de l'activation ne doit pas être appliquée une seconde fois.
   */
  backward(error: number, iteration = 1, errorIsDelta = false): number {
    if (!Number.isFinite(error)) {
      throw new Error("Erreur reçue NaN dans backward: " + error);
    }

    const delta = errorIsDelta ? error : error * this.activationDerivative();

    if (!Number.isFinite(delta)) {
      throw new Error("Delta non valide: " + delta);
    }

    if (this.optimizer === "adam") {
      this.updateAdam(delta, iteration);
    } else if (this.optimizer === "rmsprop") {
      this.updateRMSprop(delta);
    } else {
      this.updateSGD(delta);
    }

    return delta;
  }

  private activationDerivative(): number {
    switch (this.activation) {
      case "relu": return reluDerivative(this.z);
      case "leakyrelu": return leakyReluDerivative(this.z);
      case "tanh": return tanhDerivative(this.z);
      case "sigmoid": return sigmoidDerivative(this.z);
      default: return 1;
    }
  }

  private updateSGD(delta: number): void {
    this.weights = this.weights.map((w, i) => {
      const input = this.inputs[i];

      if (!Number.isFinite(input)) {
        console.warn("Input invalide dans backward:", input);
        return w;
      }

      const grad = delta * input + this.l2 * w;
      const newW = w - this.learningRate * clip(grad);

      if (!Number.isFinite(newW)) {
        console.warn("Poids devenu invalide:", newW);
        return w;
      }

      return newW;
    });

    this.bias -= this.learningRate * clip(delta);
  }

  private updateAdam(delta: number, iteration: number): void {
    this.weights = this.weights.map((w, i) => {
      const input = this.inputs[i];
      if (!Number.isFinite(input)) return w;

      const grad = clip(delta * input + this.l2 * w);

      this.m[i] = this.beta1 * this.m[i] + (1 - this.beta1) * grad;
      this.v[i] = this.beta2 * this.v[i] + (1 - this.beta2) * grad * grad;

      const mHat = this.m[i] / (1 - Math.pow(this.beta1, iteration));
      const vHat = this.v[i] / (1 - Math.pow(this.beta2, iteration));

      const newW = w - this.learningRate * mHat / (Math.sqrt(vHat) + this.epsilon);
      return Number.isFinite(newW) ? newW : w;
    });

    const gradBias = clip(delta);

    this.mBias = this.beta1 * this.mBias + (1 - this.beta1) * gradBias;
    this.vBias = this.beta2 * this.vBias + (1 - this.beta2) * gradBias * gradBias;

    const mHatBias = this.mBias / (1 - Math.pow(this.beta1, iteration));
    const vHatBias = this.vBias / (1 - Math.pow(this.beta2, iteration));

    this.bias -= this.learningRate * mHatBias / (Math.sqrt(vHatBias) + this.epsilon);
  }

  private updateRMSprop(delta: number): void {
    const decay = 0.9;

    this.weights = this.weights.map((w, i) => {
      const input = this.inputs[i];
      if (!Number.isFinite(input)) return w;

      const grad = clip(delta * input + this.l2 * w);

      this.v[i] = decay * this.v[i] + (1 - decay) * grad * grad;

      const newW = w - this.learningRate * grad / (Math.sqrt(this.v[i]) + this.epsilon);
      return Number.isFinite(newW) ? newW : w;
    });

    const gradBias = clip(delta);

    this.vBias = decay * this.vBias + (1 - decay) * gradBias * gradBias;
    this.bias -= this.learningRate * gradBias / (Math.sqrt(this.vBias) + this.epsilon);
  }
}
