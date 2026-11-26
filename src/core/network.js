import { Layer } from "./layer.js";

export class Network {
    constructor({
        inputSize = 1,
        hiddenNeurons = 10,
        learningRate = 0.0025,
        preprocessing = null,
        task = "regression",
        optimizer = "sgd",
        outputSize = 1
    } = {}) {
        this.inputSize = inputSize;
        this.preprocessing = preprocessing;
        this.task = task;
        this.optimizer = optimizer;
        this.outputSize = outputSize;

        this.hidden = new Layer(inputSize, hiddenNeurons, "tanh", learningRate, optimizer);

        if (task === "binary-classification") {
            this.output = new Layer(hiddenNeurons, 1, "sigmoid", learningRate, optimizer);
        } else if (task === "multiclass-classification") {
            this.output = new Layer(hiddenNeurons, outputSize, "softmax", learningRate, optimizer);
        } else {
            this.output = new Layer(hiddenNeurons, outputSize, "linear", learningRate, optimizer);
        }

        this.iteration = 0;
    }

    // -------------------------
    // Chargement depuis modèle
    // -------------------------
    static load(data) {
        const net = new Network({
            ...data.config,
            task: data.meta?.task ?? "regression",
            optimizer: data.config.optimizer || "sgd",
            outputSize: data.meta?.outputSize ?? 1
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
            // Les probabilités sont bornées pour éviter log(0) ; la perte reste >= 0.
            const eps = 1e-7;
            const p = Math.min(Math.max(yPred, eps), 1 - eps);
            return -(yTrue * Math.log(p) + (1 - yTrue) * Math.log(1 - p));
        } else if (this.task === "multiclass-classification") {
            const eps = 1e-9;
            return -yTrue.reduce((s, v, i) => s + v * Math.log(Math.max(yPred[i], eps)), 0);
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
        const yVec = this.output.forward(h);

        return { x, xNorm, h, yVec };
    }

    predict(input) {
        const out = this.forward(input).yVec;

        if (this.task === "multiclass-classification") {
            return out;
        }

        return out[0];
    }

    // -------------------------
    // Training
    // -------------------------
    train(input, yTrueVec) {
        this.iteration++;

        const x = this.ensureVector(input);
        const xNorm = this.normalizeX(x);

        const h = this.hidden.forward(xNorm);
        const yPred = this.output.forward(h);

        // Gradient de la perte par rapport aux entrées des neurones de sortie (z).
        // Il vaut p - y pour les trois cas : sigmoïde + entropie croisée binaire,
        // softmax + entropie croisée, sortie linéaire + erreur quadratique (à un
        // facteur 2 près, absorbé par le taux d'apprentissage).
        let gradients;

        if (this.task === "multiclass-classification") {
            gradients = yPred.map((p, i) => p - yTrueVec[i]);
        } else {
            gradients = [yPred[0] - yTrueVec[0]];
        }

        const weights = this.output.neurons.map(n => [...n.weights]);

        this.output.backward(gradients, this.iteration, { errorsAreDeltas: true });

        const hiddenErrors = this.hidden.neurons.map((_, i) => {
            return gradients.reduce((sum, g, j) => sum + g * weights[j][i], 0);
        });

        this.hidden.backward(hiddenErrors, this.iteration);

        return this.computeLoss(
            this.task === "multiclass-classification" ? yPred : yPred[0],
            this.task === "multiclass-classification" ? yTrueVec : yTrueVec[0]
        );
    }

    // -------------------------
    // Debug (MAINTENANT COHÉRENT)
    // -------------------------
    debugPredict(input) {
        const out = this.forward(input);

        return {
            input: out.x,
            inputNorm: out.xNorm,
            hidden: out.h.map((n, i) => ({
                neuron: i,
                weight: +n.toFixed(6)
            })),
            output: +out.yVec[0].toFixed(6)
        };
    }

    // -------------------------
    // Save
    // -------------------------
    toJSON() {
        return {
            meta: {
                inputSize: this.inputSize,
                outputSize: this.outputSize,
                task: this.task
            },
            config: {
                inputSize: this.inputSize,
                hiddenNeurons: this.hidden.neurons.length,
                learningRate: this.hidden.neurons[0].learningRate,
                preprocessing: this.preprocessing,
                optimizer: this.optimizer,
                iteration: this.iteration,
                outputSize: this.outputSize
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
