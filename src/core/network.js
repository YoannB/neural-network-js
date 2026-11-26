import { Layer } from "./layer.js";

// Activation de la couche de sortie selon la tâche.
const OUTPUT_ACTIVATIONS = {
    "regression": "linear",
    "binary-classification": "sigmoid",
    "multiclass-classification": "softmax"
};

export const HIDDEN_ACTIVATIONS = ["tanh", "relu", "leakyrelu", "sigmoid"];

export class Network {
    // hiddenLayers : nombre de neurones de chaque couche cachée, par exemple [16, 8].
    // hiddenNeurons reste accepté pour une seule couche cachée.
    constructor({
        inputSize = 1,
        hiddenLayers = null,
        hiddenNeurons = 10,
        activation = "tanh",
        learningRate = 0.0025,
        preprocessing = null,
        task = "regression",
        optimizer = "sgd",
        outputSize = 1
    } = {}) {
        if (!OUTPUT_ACTIVATIONS[task]) {
            throw new Error(`Tâche inconnue: ${task}`);
        }
        if (!HIDDEN_ACTIVATIONS.includes(activation)) {
            throw new Error(`Activation inconnue: ${activation} (disponibles : ${HIDDEN_ACTIVATIONS.join(", ")})`);
        }

        const sizes = hiddenLayers ?? [hiddenNeurons];
        if (sizes.length === 0 || !sizes.every(n => Number.isInteger(n) && n > 0)) {
            throw new Error(`Couches cachées invalides: ${JSON.stringify(sizes)}`);
        }

        this.inputSize = inputSize;
        this.outputSize = task === "binary-classification" ? 1 : outputSize;
        this.hiddenLayers = [...sizes];
        this.activation = activation;
        this.learningRate = learningRate;
        this.preprocessing = preprocessing;
        this.task = task;
        this.optimizer = optimizer;
        this.iteration = 0;

        // Couches cachées puis couche de sortie : chacune reçoit la sortie de la précédente.
        this.layers = [];
        let previousSize = inputSize;
        for (const size of sizes) {
            this.layers.push(new Layer(previousSize, size, activation, learningRate, optimizer));
            previousSize = size;
        }
        this.layers.push(new Layer(previousSize, this.outputSize, OUTPUT_ACTIVATIONS[task], learningRate, optimizer));
    }

    get output() {
        return this.layers[this.layers.length - 1];
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

        // Les modèles des étapes 5 à 8 ont deux couches nommées hidden et output.
        const savedLayers = data.layers ?? [data.hidden, data.output];
        if (savedLayers.length !== net.layers.length) {
            throw new Error(`Modèle incohérent: ${savedLayers.length} couches enregistrées, ${net.layers.length} attendues`);
        }

        savedLayers.forEach((savedLayer, l) => {
            savedLayer.neurons.forEach((saved, i) => {
                const neuron = net.layers[l].neurons[i];
                neuron.weights = saved.weights;
                neuron.bias = saved.bias;
                neuron.m = saved.m;
                neuron.v = saved.v;
                // Moments du biais (Adam, RMSprop) : absents des modèles des étapes 7 et 8.
                neuron.mBias = saved.mBias ?? 0;
                neuron.vBias = saved.vBias ?? 0;
            });
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

    // -------------------------
    // Loss
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

    // -------------------------
    // Inference
    // -------------------------
    // Renvoie l'entrée, l'entrée normalisée et la sortie de chaque couche.
    forward(input) {
        const x = this.ensureVector(input);
        const xNorm = this.normalizeX(x);

        const activations = [];
        let current = xNorm;
        for (const layer of this.layers) {
            current = layer.forward(current);
            activations.push(current);
        }

        return { x, xNorm, hidden: activations.slice(0, -1), yVec: current };
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

        const { yVec: yPred } = this.forward(input);

        // Gradient de la perte par rapport aux entrées des neurones de sortie (z).
        // Il vaut p - y pour les trois cas : sigmoïde + entropie croisée binaire,
        // softmax + entropie croisée, sortie linéaire + erreur quadratique (à un
        // facteur 2 près, absorbé par le taux d'apprentissage).
        let errors = yPred.map((p, i) => p - yTrueVec[i]);
        let errorsAreDeltas = true;

        // Rétropropagation de la sortie vers l'entrée. Les poids de chaque couche
        // sont copiés avant sa mise à jour : l'erreur de la couche précédente se
        // calcule avec les poids utilisés pendant la propagation avant.
        for (let l = this.layers.length - 1; l >= 0; l--) {
            const layer = this.layers[l];
            const weights = layer.neurons.map(n => [...n.weights]);

            const deltas = layer.backward(errors, this.iteration, { errorsAreDeltas });

            if (l > 0) {
                const previousSize = this.layers[l - 1].neurons.length;
                errors = Array.from({ length: previousSize }, (_, i) =>
                    deltas.reduce((sum, d, j) => sum + d * weights[j][i], 0)
                );
                errorsAreDeltas = false;
            }
        }

        return this.computeLoss(
            this.task === "multiclass-classification" ? yPred : yPred[0],
            this.task === "multiclass-classification" ? yTrueVec : yTrueVec[0]
        );
    }

    // -------------------------
    // Debug
    // -------------------------
    debugPredict(input) {
        const out = this.forward(input);
        const round = v => +v.toFixed(6);

        return {
            input: out.x,
            inputNorm: out.xNorm,
            hidden: out.hidden.map(layer => layer.map(round)),
            output: this.task === "multiclass-classification"
                ? out.yVec.map(round)
                : round(out.yVec[0])
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
                hiddenLayers: this.hiddenLayers,
                activation: this.activation,
                learningRate: this.learningRate,
                preprocessing: this.preprocessing,
                optimizer: this.optimizer,
                iteration: this.iteration,
                outputSize: this.outputSize
            },
            layers: this.layers.map(layer => ({
                activation: layer.activation,
                neurons: layer.neurons.map(n => ({
                    weights: n.weights,
                    bias: n.bias,
                    m: n.m,
                    v: n.v,
                    mBias: n.mBias,
                    vBias: n.vBias
                }))
            }))
        };
    }
}
