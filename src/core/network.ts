import { Layer } from "./layer.js";
import { createRandom, random } from "./random.js";
import { trainNetwork, type TrainOptions, type TrainResult } from "./training.js";
import { evaluate, type Evaluation } from "./metrics.js";
import type { HiddenActivation, ModelJSON, Optimizer, OutputActivation, Preprocessing, Sample, SavedLayer, Task } from "./types.js";

// Activation de la couche de sortie selon la tâche.
const OUTPUT_ACTIVATIONS: Record<Task, OutputActivation> = {
    "regression": "linear",
    "binary-classification": "sigmoid",
    "multiclass-classification": "softmax"
};

export const HIDDEN_ACTIVATIONS: readonly HiddenActivation[] = ["tanh", "relu", "leakyrelu", "sigmoid"];

export interface NetworkOptions {
    inputSize?: number;
    /** Nombre de neurones de chaque couche cachée, par exemple [16, 8]. */
    hiddenLayers?: number[] | null;
    /** Une seule couche cachée (utilisé si hiddenLayers est absent). */
    hiddenNeurons?: number;
    activation?: HiddenActivation;
    learningRate?: number;
    preprocessing?: Preprocessing | null;
    task?: Task;
    optimizer?: Optimizer;
    outputSize?: number;
    l2?: number;
    /**
     * Graine de l'initialisation des poids. Sans graine, le générateur partagé
     * est utilisé (fixé par setSeed, ou Math.random par défaut).
     */
    seed?: number;
}

export interface ForwardResult {
    x: number[];
    xNorm: number[];
    /** Sortie de chaque couche cachée. */
    hidden: number[][];
    yVec: number[];
}

export class Network {
    readonly inputSize: number;
    readonly outputSize: number;
    readonly hiddenLayers: number[];
    readonly activation: HiddenActivation;
    readonly learningRate: number;
    readonly preprocessing: Preprocessing | null;
    readonly task: Task;
    readonly optimizer: Optimizer;
    readonly l2: number;
    readonly layers: Layer[];
    iteration = 0;

    constructor({
        inputSize = 1,
        hiddenLayers = null,
        hiddenNeurons = 10,
        activation = "tanh",
        learningRate = 0.0025,
        preprocessing = null,
        task = "regression",
        optimizer = "sgd",
        outputSize = 1,
        l2 = 0,
        seed
    }: NetworkOptions = {}) {
        if (!(task in OUTPUT_ACTIVATIONS)) {
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
        this.l2 = l2;

        // Couches cachées puis couche de sortie : chacune reçoit la sortie de la précédente.
        const rand = seed === undefined ? random : createRandom(seed);
        this.layers = [];
        let previousSize = inputSize;
        for (const size of sizes) {
            this.layers.push(new Layer(previousSize, size, activation, learningRate, optimizer, l2, rand));
            previousSize = size;
        }
        this.layers.push(new Layer(previousSize, this.outputSize, OUTPUT_ACTIVATIONS[task], learningRate, optimizer, l2, rand));
    }

    get output(): Layer {
        return this.layers[this.layers.length - 1];
    }

    // -------------------------
    // Chargement depuis modèle
    // -------------------------
    static load(data: ModelJSON): Network {
        const { hiddenLayers, hiddenNeurons, activation, learningRate, preprocessing, l2, inputSize } = data.config;
        const net = new Network({
            inputSize,
            hiddenLayers: hiddenLayers ?? null,
            ...(hiddenNeurons !== undefined ? { hiddenNeurons } : {}),
            ...(activation !== undefined ? { activation } : {}),
            learningRate,
            preprocessing,
            l2: l2 ?? 0,
            task: data.meta?.task ?? "regression",
            optimizer: data.config.optimizer || "sgd",
            outputSize: data.meta?.outputSize ?? 1
        });

        // Les modèles des étapes 5 à 8 ont deux couches nommées hidden et output.
        const savedLayers: SavedLayer[] | undefined = data.layers
            ?? (data.hidden && data.output ? [data.hidden, data.output] : undefined);
        if (!savedLayers || savedLayers.length !== net.layers.length) {
            throw new Error(`Modèle incohérent: ${savedLayers?.length ?? 0} couches enregistrées, ${net.layers.length} attendues`);
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
    ensureVector(input: number[]): number[] {
        if (!Array.isArray(input)) {
            throw new Error("Network attend un tableau en entrée");
        }
        if (Array.isArray(input[0])) {
            throw new Error("Entrée invalide (tableau imbriqué)");
        }
        if (input.length !== this.inputSize) {
            throw new Error(`Dimension incorrecte: reçu ${input.length}, attendu ${this.inputSize}`);
        }
        return input;
    }

    normalizeX(xVec: number[]): number[] {
        const preprocessing = this.preprocessing;
        if (!preprocessing) return xVec;
        return xVec.map(x => x / preprocessing.xScale);
    }

    // -------------------------
    // Loss
    // -------------------------
    computeLoss(yPred: number | number[], yTrue: number | number[]): number {
        if (this.task === "multiclass-classification") {
            const eps = 1e-9;
            const p = yPred as number[];
            return -(yTrue as number[]).reduce((s, v, i) => s + v * Math.log(Math.max(p[i], eps)), 0);
        }

        const pred = yPred as number;
        const truth = yTrue as number;

        if (this.task === "binary-classification") {
            // Les probabilités sont bornées pour éviter log(0) ; la perte reste >= 0.
            const eps = 1e-7;
            const p = Math.min(Math.max(pred, eps), 1 - eps);
            return -(truth * Math.log(p) + (1 - truth) * Math.log(1 - p));
        }

        return (pred - truth) ** 2;
    }

    // -------------------------
    // Inference
    // -------------------------
    /** Renvoie l'entrée, l'entrée normalisée et la sortie de chaque couche. */
    forward(input: number[]): ForwardResult {
        const x = this.ensureVector(input);
        const xNorm = this.normalizeX(x);

        const activations: number[][] = [];
        let current = xNorm;
        for (const layer of this.layers) {
            current = layer.forward(current);
            activations.push(current);
        }

        return { x, xNorm, hidden: activations.slice(0, -1), yVec: current };
    }

    /**
     * Prédiction : un nombre en régression et en classification binaire
     * (probabilité de la classe 1), un vecteur de probabilités en multiclasse.
     */
    predict(input: number[]): number | number[] {
        const out = this.forward(input).yVec;
        return this.task === "multiclass-classification" ? out : out[0];
    }

    /** Sortie numérique : valeur en régression, probabilité de la classe 1 en binaire. */
    predictValue(input: number[]): number {
        if (this.task === "multiclass-classification") {
            throw new Error("predictValue : utiliser predictProbabilities en classification multiclasse");
        }
        return this.forward(input).yVec[0];
    }

    /** Probabilité de chaque classe (classification multiclasse). */
    predictProbabilities(input: number[]): number[] {
        return this.forward(input).yVec;
    }

    // -------------------------
    // Training
    // -------------------------
    /** Un pas d'apprentissage sur un exemple ; renvoie la perte avant la mise à jour. */
    train(input: number[], yTrueVec: number[]): number {
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

        const multiclass = this.task === "multiclass-classification";
        return this.computeLoss(multiclass ? yPred : yPred[0], multiclass ? yTrueVec : yTrueVec[0]);
    }

    /** Entraîne le réseau sur des exemples (voir trainNetwork pour les options). */
    fit(samples: Sample[], options: TrainOptions = {}): TrainResult {
        return trainNetwork(this, samples, options);
    }

    /** Perte moyenne et part des exemples bien prédits. */
    evaluate(samples: Sample[]): Evaluation {
        return evaluate(this, samples);
    }

    // -------------------------
    // Debug
    // -------------------------
    debugPredict(input: number[]) {
        const out = this.forward(input);
        const round = (v: number) => +v.toFixed(6);

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
    toJSON(): ModelJSON {
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
                l2: this.l2,
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
