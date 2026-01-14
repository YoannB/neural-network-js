import fs from "fs";
import { Network } from "./network.js";

// Chargement du modèle
const rawModel = JSON.parse(
    fs.readFileSync("./model/model.json", "utf-8")
);

const net = Network.load(rawModel);

// Jeu de test étendu
const dataset = JSON.parse(
    fs.readFileSync("./data/dataset.json", "utf-8")
);
const xs = dataset.inputs;

// Header
console.log(
    [
        "x".padEnd(6),
        "y_true".padEnd(10),
        "y_pred".padEnd(10),
        //...net.hidden.neurons.map((_, i) => `h${i}`.padEnd(8))
    ].join(" | ")
);

console.log("-".repeat(80));

for (const x of xs) {
    const yTrue = x * x;

    const debug = net.debugPredict(x);

    const row = [
        x.toFixed(2).padEnd(6),
        yTrue.toFixed(2).padEnd(10),
        debug.output.padEnd(10),
        //...debug.hiddenOutputs.map(v => v.toFixed(4).padEnd(8))
    ];

    console.log(row.join(" | "));
}

