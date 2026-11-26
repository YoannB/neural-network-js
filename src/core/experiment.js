import { Network } from "./network.js";
import { setSeed } from "./random.js";
import { trainNetwork } from "./training.js";
import getDataset from "../datasets/index.js";

// Les entrées sont divisées par leur plus grande valeur absolue (sur les
// exemples d'entraînement). Les sorties ne sont pas normalisées.
export function computePreprocessing(samples) {
    let maxX = 0;
    for (const s of samples) {
        for (const v of s.x) {
            maxX = Math.max(maxX, Math.abs(v));
        }
    }
    return { xScale: maxX || 1 };
}

// Charge le jeu de données, construit le réseau et l'entraîne.
// La graine fixe à la fois la séparation entraînement / test et l'initialisation.
// onStart({ dataset, net }) est appelé juste avant la première époque.
export async function runExperiment(datasetName, options, { onStart = null, onEpoch = null } = {}) {
    const dataset = await getDataset(datasetName, { seed: options.seed, ...options.datasetOptions });

    setSeed(options.seed);

    const net = new Network({
        inputSize: dataset.inputSize,
        outputSize: dataset.outputSize,
        hiddenLayers: options.layers,
        activation: options.activation,
        learningRate: options.lr,
        preprocessing: computePreprocessing(dataset.samples),
        task: dataset.task,
        optimizer: options.optimizer
    });

    onStart?.({ dataset, net });

    const result = trainNetwork(net, dataset.samples, {
        epochs: options.epochs,
        noise: options.noise,
        targetLoss: options.targetLoss,
        onEpoch
    });

    const model = net.toJSON();
    model.config.seed = options.seed;
    model.config.dataset = options.datasetOptions;

    return { dataset, net, result, model };
}

// Part des exemples bien prédits : seuil 0,5 en classification binaire,
// classe la plus probable en multiclasse, 5 % d'erreur relative en régression.
export function accuracy(net, samples) {
    if (samples.length === 0) return NaN;

    const argmax = v => v.indexOf(Math.max(...v));
    let correct = 0;

    for (const { x, y } of samples) {
        const p = net.predict(x);
        if (net.task === "multiclass-classification") {
            if (argmax(p) === argmax(y)) correct++;
        } else if (net.task === "binary-classification") {
            if ((p >= 0.5 ? 1 : 0) === y[0]) correct++;
        } else {
            const err = y[0] > 0 ? Math.abs(p - y[0]) / Math.abs(y[0]) * 100 : Math.abs(p) * 100;
            if (err <= 5) correct++;
        }
    }

    return (correct / samples.length) * 100;
}
