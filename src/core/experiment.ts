import { Network } from "./network.js";
import { setSeed } from "./random.js";
import { trainNetwork, type EpochInfo, type TrainResult } from "./training.js";
import getDataset from "../datasets/index.js";
import { splitSamples } from "../datasets/split.js";
import type { Dataset, HiddenActivation, ModelJSON, Optimizer, Preprocessing, Sample } from "./types.js";

export interface ExperimentOptions {
    epochs: number;
    layers: number[];
    activation: HiddenActivation;
    lr: number;
    noise: number;
    optimizer: Optimizer;
    seed: number;
    targetLoss: number;
    /** Pénalité sur les poids. */
    l2?: number;
    /** Part des exemples d'entraînement mise de côté pour décider quand s'arrêter (0 = aucune). */
    validation?: number;
    /** Époques sans amélioration de la validation avant l'arrêt. */
    patience?: number;
    /** "flip" ajoute les images miroir des exemples d'entraînement. */
    augment?: string;
    /** Options propres au jeu de données (par exemple size et mode pour images). */
    datasetOptions: Record<string, unknown>;
}

export interface ExperimentStart {
    dataset: Dataset;
    net: Network;
    fitSamples: Sample[];
    validationSamples: Sample[];
}

export interface ExperimentResult {
    dataset: Dataset;
    net: Network;
    result: TrainResult;
    model: ModelJSON;
}

/**
 * Les entrées sont divisées par leur plus grande valeur absolue (sur les
 * exemples d'entraînement). Les sorties ne sont pas normalisées.
 */
export function computePreprocessing(samples: Sample[]): Preprocessing {
    let maxX = 0;
    for (const s of samples) {
        for (const v of s.x) {
            maxX = Math.max(maxX, Math.abs(v));
        }
    }
    return { xScale: maxX || 1 };
}

/**
 * Charge le jeu de données, construit le réseau et l'entraîne.
 * La graine fixe à la fois la séparation entraînement / test et l'initialisation.
 * La validation est prise sur les exemples d'entraînement avant l'augmentation,
 * qui ne touche donc ni la validation ni le test.
 * onStart est appelé juste avant la première époque.
 */
export async function runExperiment(
    datasetName: string,
    options: ExperimentOptions,
    { onStart = null, onEpoch = null }: {
        onStart?: ((start: ExperimentStart) => void) | null;
        onEpoch?: ((info: EpochInfo) => void) | null;
    } = {}
): Promise<ExperimentResult> {
    const dataset = await getDataset(datasetName, { seed: options.seed, ...options.datasetOptions });

    let fitSamples = dataset.samples;
    let validationSamples: Sample[] = [];
    if ((options.validation ?? 0) > 0) {
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
        ...(options.patience !== undefined ? { patience: options.patience } : {}),
        onEpoch
    });

    const model = net.toJSON();
    model.config.seed = options.seed;
    model.config.dataset = options.datasetOptions;
    model.config.training = { validation: options.validation ?? 0, augment: options.augment ?? "none" };

    return { dataset, net, result, model };
}

const argmax = (v: number[]): number => v.indexOf(Math.max(...v));

/**
 * Part des exemples bien prédits, en % : seuil 0,5 en classification binaire,
 * classe la plus probable en multiclasse, 5 % d'erreur relative en régression.
 */
export function accuracy(net: Network, samples: Sample[]): number {
    if (samples.length === 0) return NaN;

    let correct = 0;

    for (const { x, y } of samples) {
        const p = net.predict(x);
        if (Array.isArray(p)) {
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
