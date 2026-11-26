// Point d'entrée de la bibliothèque.
//
// Le cœur (réseau, entraînement, métriques, générateur aléatoire) n'utilise ni
// les modules de Node.js ni sharp : il fonctionne aussi dans un navigateur.
// Les jeux de données et les scripts (train, table, exec, evaluate) servent
// d'exemples et d'outils en ligne de commande ; ils ne sont pas exportés ici.

export { Network, HIDDEN_ACTIVATIONS } from "./core/network.js";
export type { NetworkOptions, ForwardResult } from "./core/network.js";

export { trainNetwork, meanLoss } from "./core/training.js";
export type { TrainOptions, TrainResult, EpochInfo } from "./core/training.js";

export { accuracy, evaluate } from "./core/metrics.js";
export type { Evaluation } from "./core/metrics.js";

export { computePreprocessing } from "./core/preprocessing.js";
export { createRandom, setSeed, shuffle } from "./core/random.js";
export type { RandomFn } from "./core/random.js";
export { splitSamples } from "./datasets/split.js";

export { Layer } from "./core/layer.js";
export { Neuron } from "./core/neuron.js";
export * as activations from "./core/activations.js";

export type {
    Task,
    HiddenActivation,
    OutputActivation,
    Activation,
    Optimizer,
    Sample,
    Dataset,
    Preprocessing,
    ModelJSON,
    ModelConfig,
    SavedLayer,
    SavedNeuron
} from "./core/types.js";
