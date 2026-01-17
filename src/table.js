import fs from "fs";
import { Network } from "./core/network.js";
import getDataset from "./datasets/index.js";

// -------------------------
// Config
// -------------------------
const THRESHOLD = 0.5;      // classification binaire
const REG_TOLERANCE = 5;    // % d’erreur tolérée pour régression

// -------------------------
// Chargement du modèle
// -------------------------
const rawModel = JSON.parse(
    fs.readFileSync("./model/model.json", "utf-8")
);

const net = Network.load(rawModel);

// -------------------------
// Dataset
// -------------------------
if (!rawModel.meta.dataset) {
    console.error("Aucun dataset trouvé dans model.json");
    process.exit(1);
}

const dataset = getDataset(rawModel.meta.dataset);
const task = dataset.task ?? "regression";

// -------------------------
// Header
// -------------------------
console.log(
    [
        "Input".padEnd(40),
        "y_true".padEnd(12),
        "y_pred".padEnd(12),
        "err_%".padEnd(10),
        "ok?".padEnd(6)
    ].join(" | ")
);

console.log("-".repeat(105));

// -------------------------
// Stats
// -------------------------
let correct = 0;
let falsePositives = 0;
let falseNegatives = 0;

// -------------------------
// Table
// -------------------------
dataset.samples.forEach(({ x, y, label }) => {
    const yTrue = y[0];
    const yPred = net.predict(x);

    const errPct = Math.abs(yPred - yTrue) / (Math.abs(yTrue) + 1e-9) * 100;

    let isCorrect = false;

    if (task === "binary-classification") {
        const predictedClass = yPred >= THRESHOLD ? 1 : 0;
        isCorrect = predictedClass === yTrue;

        if (isCorrect) correct++;
        if (predictedClass === 1 && yTrue === 0) falsePositives++;
        if (predictedClass === 0 && yTrue === 1) falseNegatives++;
    } else {
        // Régression → correct si erreur < tolérance
        isCorrect = errPct <= REG_TOLERANCE;
        if (isCorrect) correct++;
    }

    const inputStr = label
        ? label.padEnd(40)
        : JSON.stringify(x).padEnd(40);

    const yTrueStr = yTrue.toFixed(6).padEnd(12);
    const yPredStr = yPred.toFixed(6).padEnd(12);
    const errStr = errPct.toFixed(2).padEnd(10);
    const okStr = (isCorrect ? "OK" : "KO").padEnd(6);

    console.log([inputStr, yTrueStr, yPredStr, errStr, okStr].join(" | "));
});

// -------------------------
// Résumé
// -------------------------
const total = dataset.samples.length;
const accuracy = (correct / total) * 100;

console.log("\nRésumé:");
console.log(`Task           : ${task}`);
console.log(`Samples        : ${total}`);
console.log(`Accuracy       : ${accuracy.toFixed(2)} %`);

if (task === "binary-classification") {
    console.log(`False positives: ${falsePositives}`);
    console.log(`False negatives: ${falseNegatives}`);
}
