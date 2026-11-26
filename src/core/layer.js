import { softmax } from "./activations.js";
import { Neuron } from "./neuron.js";

export class Layer {
    constructor(inputSize, count, activation, learningRate, optimizer) {
        this.activation = activation;

        this.neurons = Array.from(
            { length: count },
            () => new Neuron(inputSize, activation, learningRate, optimizer)
        );
    }

    forward(inputs) {
        // Calcul classique
        const raw = this.neurons.map(n => n.forward(inputs));

        // Cas spécial softmax (au niveau couche)
        if (this.activation === "softmax") {
            this.outputs = softmax(raw);

            // On met à jour output de chaque neurone pour cohérence/debug
            this.neurons.forEach((n, i) => {
                n.output = this.outputs[i];
            });

            return this.outputs;
        }

        // Cas normal inchangé
        this.outputs = raw;
        return this.outputs;
    }

    backward(errors, iteration = 1, { errorsAreDeltas = false } = {}) {
        this.neurons.forEach((n, i) => n.backward(errors[i], iteration, errorsAreDeltas));
    }

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
