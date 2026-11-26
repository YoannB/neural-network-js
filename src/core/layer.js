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

    // Met à jour les neurones et renvoie leurs deltas (dérivées de la perte par
    // rapport à z), qui servent à calculer l'erreur de la couche précédente.
    backward(errors, iteration = 1, { errorsAreDeltas = false } = {}) {
        return this.neurons.map((n, i) => n.backward(errors[i], iteration, errorsAreDeltas));
    }
}
