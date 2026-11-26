// Lecture des options de la ligne de commande, partagée par train et evaluate.
//   node dist/src/train.js <dataset> --epochs 3000 --layers 16,8 ...
import { HIDDEN_ACTIVATIONS } from "./core/network.js";
import type { ExperimentOptions } from "./experiment.js";
import type { HiddenActivation, Optimizer } from "./core/types.js";

const OPTIMIZERS: readonly Optimizer[] = ["sgd", "adam", "rmsprop"];

/** Valeur qui suit --name : un nombre si elle en est un, sinon le texte. */
export function getArg(args: string[], name: string): string | number | undefined {
    const index = args.indexOf(`--${name}`);
    if (index === -1) return undefined;

    const value = args[index + 1];
    if (value === undefined || value.startsWith("--")) return undefined;

    const numValue = Number(value);
    return Number.isFinite(numValue) ? numValue : value;
}

function getNumber(args: string[], name: string, defaultValue: number): number {
    const value = getArg(args, name);
    if (value === undefined) return defaultValue;
    if (typeof value !== "number") throw new Error(`--${name} attend un nombre, reçu ${value}`);
    return value;
}

function getChoice<T extends string>(args: string[], name: string, choices: readonly T[], defaultValue: T): T {
    const value = getArg(args, name);
    if (value === undefined) return defaultValue;
    if (!choices.includes(value as T)) {
        throw new Error(`--${name} : ${value} inconnu (disponibles : ${choices.join(", ")})`);
    }
    return value as T;
}

export function parseTrainingOptions(args: string[]): ExperimentOptions {
    const neurons = getNumber(args, "neurons", 10);

    return {
        epochs: getNumber(args, "epochs", 10000),
        // --layers 16,8 : deux couches cachées de 16 puis 8 neurones (remplace --neurons).
        layers: String(getArg(args, "layers") ?? neurons).split(",").map(Number),
        activation: getChoice<HiddenActivation>(args, "activation", HIDDEN_ACTIVATIONS, "tanh"),
        lr: getNumber(args, "lr", 0.0025),
        noise: getNumber(args, "noise", 0),
        optimizer: getChoice(args, "optimizer", OPTIMIZERS, "adam"),
        seed: getNumber(args, "seed", 42),
        targetLoss: getNumber(args, "target-loss", 0.0001),
        l2: getNumber(args, "l2", 0),
        // --validation 0.2 : 20 % des exemples d'entraînement servent à décider quand s'arrêter.
        validation: getNumber(args, "validation", 0),
        patience: getNumber(args, "patience", 500),
        augment: getChoice(args, "augment", ["none", "flip"], "none"),
        datasetOptions: parseDatasetOptions(args)
    };
}

// Options du jeu images (--image-size, --image-mode) : seules celles qui sont
// données sont transmises, les autres jeux de données les ignorent.
function parseDatasetOptions(args: string[]): Record<string, unknown> {
    const options: Record<string, unknown> = {};
    const size = getArg(args, "image-size");
    const mode = getArg(args, "image-mode");
    if (size !== undefined) options.size = size;
    if (mode !== undefined) options.mode = mode;
    return options;
}
