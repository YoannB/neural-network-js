import { Neuron } from "./neuron.js";

export class Layer {
    constructor(inputSize, neuronCount) {
        this.neurons = Array.from(
            { length: neuronCount },
            () => new Neuron(inputSize)
        );
    }

    forward(inputs) {
        this.outputs = this.neurons.map(
            (n) => n.forward(inputs)
        );
        return this.outputs;
    }

    backward(errors) {
        this.neurons.forEach(
            (n, i) => n.backward(errors[i])
        );
    }
}
