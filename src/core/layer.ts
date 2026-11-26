import { softmax } from "./activations.js";
import { Neuron } from "./neuron.js";
import type { Activation, Optimizer } from "./types.js";

export class Layer {
    readonly activation: Activation;
    readonly neurons: Neuron[];
    outputs: number[] = [];

    constructor(inputSize: number, count: number, activation: Activation, learningRate: number, optimizer: Optimizer, l2 = 0) {
        this.activation = activation;
        this.neurons = Array.from(
            { length: count },
            () => new Neuron(inputSize, activation, learningRate, optimizer, l2)
        );
    }

    forward(inputs: number[]): number[] {
        const raw = this.neurons.map(n => n.forward(inputs));

        // Softmax se calcule sur toute la couche.
        if (this.activation === "softmax") {
            this.outputs = softmax(raw);
            this.neurons.forEach((n, i) => {
                n.output = this.outputs[i];
            });
            return this.outputs;
        }

        this.outputs = raw;
        return this.outputs;
    }

    /**
     * Met à jour les neurones et renvoie leurs deltas (dérivées de la perte par
     * rapport à z), qui servent à calculer l'erreur de la couche précédente.
     */
    backward(errors: number[], iteration = 1, { errorsAreDeltas = false } = {}): number[] {
        return this.neurons.map((n, i) => n.backward(errors[i], iteration, errorsAreDeltas));
    }
}
