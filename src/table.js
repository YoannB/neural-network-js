import fs from "fs";
import { Network } from "./core/network.js";
import getDataset from "./datasets/index.js";


// -------------------------
// Chargement du modèle
// -------------------------
const rawModel = JSON.parse(
    fs.readFileSync("./model/model.json", "utf-8")
);

const net = Network.load(rawModel);

// -------------------------
// Vérification dataset intégré
// -------------------------
if (!rawModel.meta.dataset) {
    console.error("Aucun dataset trouvé dans model.json");
    process.exit(1);
}

const dataset = getDataset(rawModel.meta.dataset)

// -------------------------
// Header
// -------------------------
console.log(
    [
        "Input".padEnd(30),
        "y_true".padEnd(12),
        "y_pred".padEnd(12)
    ].join(" | ")
);

console.log("-".repeat(70));

// -------------------------
// Table
// -------------------------
dataset.samples.forEach(({ x, y }, i) => {
    const yTrue = y;

    const yPred = net.predict(x);

    const inputStr = JSON.stringify(x).padEnd(30);
    const yTrueStr = Number(yTrue).toFixed(6).padEnd(12);
    const yPredStr = Number(yPred).toFixed(6).padEnd(12);

    console.log([inputStr, yTrueStr, yPredStr].join(" | "));
});
