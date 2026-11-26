// Types partagés par le réseau, les jeux de données et les scripts.

export type Task = "regression" | "binary-classification" | "multiclass-classification";
export type HiddenActivation = "tanh" | "relu" | "leakyrelu" | "sigmoid";
export type OutputActivation = "linear" | "sigmoid" | "softmax";
export type Activation = HiddenActivation | OutputActivation;
export type Optimizer = "sgd" | "adam" | "rmsprop";

export interface Sample {
    x: number[];
    y: number[];
    label?: string;
}

export interface Dataset {
    name: string;
    inputSize: number;
    outputSize: number;
    task: Task;
    /** Exemples d'entraînement. */
    samples: Sample[];
    /** Exemples réservés au test, jamais utilisés pour apprendre. */
    testSamples?: Sample[];
    /** Exemples supplémentaires construits à partir des exemples donnés (augmentation). */
    augment?: (samples: Sample[]) => Promise<Sample[]>;
}

export interface Preprocessing {
    /** Les entrées sont divisées par cette valeur. */
    xScale: number;
}

export interface SavedNeuron {
    weights: number[];
    bias: number;
    m: number[];
    v: number[];
    mBias?: number;
    vBias?: number;
}

export interface SavedLayer {
    activation?: Activation;
    neurons: SavedNeuron[];
}

export interface ModelConfig {
    inputSize: number;
    hiddenLayers?: number[];
    /** Ancien format : une seule couche cachée. */
    hiddenNeurons?: number;
    activation?: HiddenActivation;
    learningRate: number;
    preprocessing: Preprocessing | null;
    optimizer: Optimizer;
    l2?: number;
    iteration: number;
    outputSize: number;
    seed?: number;
    dataset?: Record<string, unknown>;
    training?: { validation: number; augment: string };
}

export interface ModelJSON {
    meta: { inputSize: number; outputSize: number; task: Task };
    config: ModelConfig;
    layers?: SavedLayer[];
    /** Ancien format (étapes 5 à 8). */
    hidden?: SavedLayer;
    output?: SavedLayer;
}
