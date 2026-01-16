import { Layer } from "./layer.js";

export class Network {
    constructor({
        inputSize = 1,
        hiddenNeurons = 12,
        learningRate = 0.003,
        preprocessing = null,
        task = "regression"
    } = {}) {
        this.inputSize = inputSize;
        this.preprocessing = preprocessing;
        this.task = task;

        this.hidden = new Layer(inputSize, hiddenNeurons, "tanh", learningRate);
        // Activation de sortie selon la task
        if (task === "binary-classification") {
            this.output = new Layer(hiddenNeurons, 1, "sigmoid", learningRate);
        } else {
            this.output = new Layer(hiddenNeurons, 1, "linear", learningRate);
        }
    }

    // -------------------------
    // Chargement depuis modèle
    // -------------------------
    static load(data) {
        const net = new Network({
            ...data.config,
            task: data.meta?.task ?? "regression"
        });

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

    // -------------------------
    // Utils
    // -------------------------
    ensureVector(input) {
        if (!Array.isArray(input)) {
            throw new Error("Network attend un tableau en entrée");
        }

        if (Array.isArray(input[0])) {
            throw new Error("Entrée invalide (tableau imbriqué)");
        }

        if (input.length !== this.inputSize) {
            throw new Error(
                `Dimension incorrecte: reçu ${input.length}, attendu ${this.inputSize}`
            );
        }

        return input;
    }

    normalizeX(xVec) {
        if (!this.preprocessing) return xVec;

        return xVec.map(x =>
            Math.abs(x) < 1 ? x / this.preprocessing.xScale : x
        );
    }

    denormalizeY(y) {
        if (!this.preprocessing) return y;

        return Math.abs(y) < 1 ? y * this.preprocessing.yScale : y;
    }

    // -------------------------
    // Loss helpers
    // -------------------------
    computeLoss(yPred, yTrue) {
        if (this.task === "binary-classification") {
            // Binary Cross Entropy
            const eps = 1e-7;
            return -(yTrue * Math.log(yPred + eps) + (1 - yTrue) * Math.log(1 - yPred + eps));
        } else {
            // MSE
            return (yPred - yTrue) ** 2;
        }
    }

    computeOutputGradient(yPred, yTrue) {
        // Grâce à sigmoid + BCE, le gradient se simplifie :
        // dL/dz = yPred - yTrue
        return yPred - yTrue;
    }

    // -------------------------
    // Inference
    // -------------------------
    predict(input) {
        const x = this.ensureVector(input);
        const xNorm = this.normalizeX(x);

        const h = this.hidden.forward(xNorm);
        const yNorm = this.output.forward(h)[0];

        return this.denormalizeY(yNorm);
    }

    // -------------------------
    // Training
    // -------------------------
    train(input, yTrueVec) {
        const x = this.ensureVector(input);
        const xNorm = this.normalizeX(x);

        const yTrue = yTrueVec[0];

        const yNorm =
            this.preprocessing && Math.abs(yTrue) < 1
                ? yTrue / this.preprocessing.yScale
                : yTrue;

        const h = this.hidden.forward(xNorm);
        const yPred = this.output.forward(h)[0];

        const gradient = this.computeOutputGradient(yPred, yNorm);

        const weights = [...this.output.neurons[0].weights];

        // Backprop output
        this.output.backward([gradient]);

        // Backprop hidden
        const hiddenErrors = this.hidden.neurons.map((_, i) => {
            return gradient * weights[i];
        });

        this.hidden.backward(hiddenErrors);

        return this.computeLoss(yPred, yNorm);
    }

    // -------------------------
    // Debug
    // -------------------------
    debugPredict(input) {
        const x = this.ensureVector(input);
        const h = this.hidden.forward(x);
        const y = this.output.forward(h)[0];

        return {
            input: x,
            hidden: this.hidden.neurons.map((n, i) => ({
                neuron: i,
                z: n.z.toFixed(6),
                activation: n.output.toFixed(6)
            })),
            output: y.toFixed(6)
        };
    }
}
