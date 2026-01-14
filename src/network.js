import { Layer } from "./layer.js";

export class Network {
    constructor({ hiddenNeurons = 12, learningRate = 0.003, preprocessing = null } = {}) {
        this.preprocessing = preprocessing;
        this.hidden = new Layer(1, hiddenNeurons, "tanh", learningRate);
        this.output = new Layer(hiddenNeurons, 1, "linear", learningRate);
    }

    static load(data) {
        const net = new Network(data.config);

        data.hidden.neurons.forEach((saved, i) => {
            net.hidden.neurons[i].weights = saved.weights;
            net.hidden.neurons[i].bias = saved.bias;
        });

        data.output.neurons.forEach((saved, i) => {
            net.output.neurons[i].weights = saved.weights;
            net.output.neurons[i].bias = saved.bias;
        });

        return net;
    }

    normalizeX(x) {
        // normalize only if x < 1
        return x < 1 ? this.preprocessing ? x / this.preprocessing.xScale : x : x;
    }

    denormalizeY(y) {
        // normalize only if y < 1
        return y < 1 ? this.preprocessing ? y * this.preprocessing.yScale : y : y;
    }


    predict(x) {
        const xNorm = this.normalizeX(x);

        const h = this.hidden.forward([xNorm]);
        const yNorm = this.output.forward(h)[0];

        return this.denormalizeY(yNorm);
    }

    train(x, yTrue) {
        const xNorm = this.normalizeX(x);
        const yNorm = x < 1 ? this.preprocessing ? yTrue / this.preprocessing.yScale : yTrue : yTrue;

        const h = this.hidden.forward([xNorm]);
        const yPred = this.output.forward(h)[0];

        const error = yPred - yNorm;

        this.output.backward([error]);

        const hiddenErrors = this.hidden.neurons.map((_, i) => {
            return error * this.output.neurons[0].weights[i];
        });

        this.hidden.backward(hiddenErrors);

        return error ** 2;
    }

    debugPredict(x) {
        const hiddenOutputs = this.hidden.forward([x]);
        const output = this.output.forward(hiddenOutputs)[0];

        return {
            x,
            hidden: this.hidden.neurons.map((n, i) => ({
                neuron: i,
                z: n.z.toFixed(4),
                activation: n.output.toFixed(4)
            })),
            output: output.toFixed(4),
            hiddenOutputs
        };
    }
}
