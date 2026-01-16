import { Neuron } from "./neuron.js";

export class Layer {
    constructor(inputSize, count, activation, learningRate) {
        this.neurons = Array.from(
            { length: count },
            () => new Neuron(inputSize, activation, learningRate)
        );
    }

    forward(inputs) {
        this.outputs = this.neurons.map(n => n.forward(inputs));
        return this.outputs;
    }

    backward(errors) {
        this.neurons.forEach((n, i) => n.backward(errors[i]));
    }
}
