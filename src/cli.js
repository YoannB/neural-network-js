// Lecture des options de la ligne de commande, partagée par train et evaluate.
//   node src/train.js <dataset> --epochs 3000 --layers 16,8 ...

export function getArg(args, name, defaultValue) {
    const index = args.indexOf(`--${name}`);
    if (index === -1) return defaultValue;

    const value = args[index + 1];
    if (value === undefined || value.startsWith("--")) return defaultValue;

    const numValue = Number(value);
    return Number.isFinite(numValue) ? numValue : value;
}

export function parseTrainingOptions(args) {
    const neurons = getArg(args, "neurons", 10);

    return {
        epochs: getArg(args, "epochs", 10000),
        // --layers 16,8 : deux couches cachées de 16 puis 8 neurones (remplace --neurons).
        layers: String(getArg(args, "layers", neurons)).split(",").map(Number),
        activation: getArg(args, "activation", "tanh"),
        lr: getArg(args, "lr", 0.0025),
        noise: getArg(args, "noise", 0),
        optimizer: getArg(args, "optimizer", "adam"),
        seed: getArg(args, "seed", 42),
        targetLoss: getArg(args, "target-loss", 0.0001),
        datasetOptions: parseDatasetOptions(args)
    };
}

// Options du jeu images (--image-size, --image-mode) : seules celles qui sont
// données sont transmises, les autres jeux de données les ignorent.
function parseDatasetOptions(args) {
    const options = {};
    const size = getArg(args, "image-size", undefined);
    const mode = getArg(args, "image-mode", undefined);
    if (size !== undefined) options.size = size;
    if (mode !== undefined) options.mode = mode;
    return options;
}
