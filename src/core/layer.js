import { Neuron } from "./neuron.js";

export class Layer {
    constructor(inputSize, count, activation, learningRate, optimizer) {
        this.neurons = Array.from(
            { length: count },
            () => new Neuron(inputSize, activation, learningRate, optimizer)
        );
    }

    forward(inputs) {
        this.outputs = this.neurons.map(n => n.forward(inputs));
        return this.outputs;
    }

    backward(errors, iteration = 1) {
        this.neurons.forEach((n, i) => n.backward(errors[i], iteration));
    }

    // Utile pour visualiser l'état de la couche
    getWeightStats() {
        const allWeights = this.neurons.flatMap(n => n.weights);
        const mean = allWeights.reduce((s, w) => s + w, 0) / allWeights.length;
        const variance = allWeights.reduce((s, w) => s + (w - mean) ** 2, 0) / allWeights.length;

        return {
            count: allWeights.length,
            mean: mean,
            std: Math.sqrt(variance),
            min: Math.min(...allWeights),
            max: Math.max(...allWeights)
        };
    }
}
