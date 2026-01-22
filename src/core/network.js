import { Layer } from "./layer.js";

export class Network {
    constructor({
        inputSize = 1,
        hiddenNeurons = 10,
        learningRate = 0.0025,
        preprocessing = null,
        task = "regression",
        optimizer = "sgd" // sgd, adam, rmsprop
    } = {}) {
        this.inputSize = inputSize;
        this.preprocessing = preprocessing;
        this.task = task;
        this.optimizer = optimizer;

        this.hidden = new Layer(inputSize, hiddenNeurons, "tanh", learningRate, optimizer);

        if (task === "binary-classification") {
            this.output = new Layer(hiddenNeurons, 1, "sigmoid", learningRate, optimizer);
        } else {
            this.output = new Layer(hiddenNeurons, 1, "linear", learningRate, optimizer);
        }

        // Compteur d'itérations pour Adam/RMSprop
        this.iteration = 0;
    }

    // -------------------------
    // Chargement depuis modèle
    // -------------------------
    static load(data) {
        const net = new Network({
            ...data.config,
            task: data.meta?.task ?? "regression",
            optimizer: data.config.optimizer || "sgd"
        });

        data.hidden.neurons.forEach((saved, i) => {
            net.hidden.neurons[i].weights = saved.weights;
            net.hidden.neurons[i].bias = saved.bias;
            net.hidden.neurons[i].m = saved.m;
            net.hidden.neurons[i].v = saved.v;
        });

        data.output.neurons.forEach((saved, i) => {
            net.output.neurons[i].weights = saved.weights;
            net.output.neurons[i].bias = saved.bias;
            net.output.neurons[i].m = saved.m;
            net.output.neurons[i].v = saved.v;
        });

        net.iteration = data.config.iteration || 0;

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
        return xVec.map(x => x / this.preprocessing.xScale);
    }

    denormalizeY(y) {
        if (!this.preprocessing) return y;
        return y * this.preprocessing.yScale;
    }

    // -------------------------
    // Loss helpers
    // -------------------------
    computeLoss(yPred, yTrue) {
        if (this.task === "binary-classification") {
            const eps = 1e-7;
            return -(yTrue * Math.log(yPred + eps) + (1 - yTrue) * Math.log(1 - yPred + eps));
        } else {
            return (yPred - yTrue) ** 2;
        }
    }

    computeOutputGradient(yPred, yTrue) {
        return yPred - yTrue;
    }

    // -------------------------
    // Inference (PROPRE)
    // -------------------------
    forward(input) {
        const x = this.ensureVector(input);
        const xNorm = this.normalizeX(x);

        const h = this.hidden.forward(xNorm);
        const yNorm = this.output.forward(h)[0];

        return {
            x,
            xNorm,
            h,
            yNorm,
            y: this.denormalizeY(yNorm)
        };
    }

    predict(input) {
        return this.forward(input).y;
    }

    // -------------------------
    // Training
    // -------------------------
    train(input, yTrueVec) {
        this.iteration++;

        const x = this.ensureVector(input);
        const xNorm = this.normalizeX(x);

        const yTrue = yTrueVec[0];

        const yNorm =
            this.preprocessing
                ? yTrue / this.preprocessing.yScale
                : yTrue;

        const h = this.hidden.forward(xNorm);
        const yPred = this.output.forward(h)[0];

        const gradient = this.computeOutputGradient(yPred, yNorm);

        const weights = [...this.output.neurons[0].weights];

        this.output.backward([gradient], this.iteration);

        const hiddenErrors = this.hidden.neurons.map((_, i) => {
            return gradient * weights[i];
        });

        this.hidden.backward(hiddenErrors, this.iteration);

        return this.computeLoss(yPred, yNorm);
    }

    // -------------------------
    // Debug (MAINTENANT COHÉRENT)
    // -------------------------
    debugPredict(input) {
        const out = this.forward(input);

        return {
            input: out.x,
            inputNorm: out.xNorm,
            hidden: this.hidden.neurons.map((n, i) => ({
                neuron: i,
                z: +n.z.toFixed(6),
                activation: +n.output.toFixed(6)
            })),
            outputNorm: +out.yNorm.toFixed(6),
            output: +out.y.toFixed(6)
        };
    }

    // -------------------------
    // Sauvegarde
    // -------------------------
    toJSON() {
        return {
            meta: {
                inputSize: this.inputSize,
                outputSize: 1,
                task: this.task
            },
            config: {
                inputSize: this.inputSize,
                hiddenNeurons: this.hidden.neurons.length,
                learningRate: this.hidden.neurons[0].learningRate,
                preprocessing: this.preprocessing,
                optimizer: this.optimizer,
                iteration: this.iteration
            },
            hidden: {
                neurons: this.hidden.neurons.map(n => ({
                    weights: n.weights,
                    bias: n.bias,
                    m: n.m,
                    v: n.v
                }))
            },
            output: {
                neurons: this.output.neurons.map(n => ({
                    weights: n.weights,
                    bias: n.bias,
                    m: n.m,
                    v: n.v
                }))
            }
        };
    }
}
