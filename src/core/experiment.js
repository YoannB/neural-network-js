import { Network } from "./network.js";
import { setSeed } from "./random.js";
import { trainNetwork } from "./training.js";
import getDataset from "../datasets/index.js";
import { splitSamples } from "../datasets/split.js";

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
//
// Options en plus des hyperparamètres :
//   validation : part des exemples d'entraînement mise de côté pour décider
//                quand s'arrêter (0 = pas de validation) ; patience en époques ;
//   augment    : "flip" ajoute les images miroir des exemples d'entraînement
//                (après la mise de côté de la validation, qui reste intacte) ;
//   l2         : pénalité sur les poids.
// onStart({ dataset, net, fitSamples, validationSamples }) est appelé juste
// avant la première époque.
export async function runExperiment(datasetName, options, { onStart = null, onEpoch = null } = {}) {
    const dataset = await getDataset(datasetName, { seed: options.seed, ...options.datasetOptions });

    let fitSamples = dataset.samples;
    let validationSamples = [];
    if (options.validation > 0) {
        const split = splitSamples(dataset.samples, { testRatio: options.validation, seed: options.seed + 1 });
        fitSamples = split.samples;
        validationSamples = split.testSamples;
    }

    if (options.augment === "flip") {
        if (!dataset.augment) throw new Error(`Le jeu ${datasetName} ne propose pas d'augmentation`);
        fitSamples = [...fitSamples, ...await dataset.augment(fitSamples)];
    } else if (options.augment && options.augment !== "none") {
        throw new Error(`Augmentation inconnue: ${options.augment} (disponibles : flip)`);
    }

    setSeed(options.seed);

    const net = new Network({
        inputSize: dataset.inputSize,
        outputSize: dataset.outputSize,
        hiddenLayers: options.layers,
        activation: options.activation,
        learningRate: options.lr,
        preprocessing: computePreprocessing(fitSamples),
        task: dataset.task,
        optimizer: options.optimizer,
        l2: options.l2 ?? 0
    });

    onStart?.({ dataset, net, fitSamples, validationSamples });

    const result = trainNetwork(net, fitSamples, {
        epochs: options.epochs,
        noise: options.noise,
        targetLoss: options.targetLoss,
        validation: validationSamples,
        patience: options.patience,
        onEpoch
    });

    const model = net.toJSON();
    model.config.seed = options.seed;
    model.config.dataset = options.datasetOptions;
    model.config.training = { validation: options.validation ?? 0, augment: options.augment ?? "none" };

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
