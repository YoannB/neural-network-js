import fs from "fs";
import { Network } from "./core/network.js";
import getDataset from "./datasets/index.js";

// -------------------------
const THRESHOLD = 0.5;
const REG_TOLERANCE = 5;

// -------------------------
const datasetName = process.argv[2] || "square";
const modelName = datasetName;

// -------------------------
const rawModel = JSON.parse(
    fs.readFileSync(`./models/${modelName}.json`, "utf-8")
);

const net = Network.load(rawModel);

// -------------------------
const dataset = getDataset(datasetName);
const samples = dataset.testSamples ?? dataset.samples;
const LABEL_WIDTH = Math.max(
    40,
    ...samples.map(s => (s.label?.length ?? 0))
);
const task = dataset.task ?? "regression";

// -------------------------
console.log(
    [
        "Input".padEnd(LABEL_WIDTH),
        "y_true".padEnd(12),
        "y_pred".padEnd(12),
        "err_%".padEnd(10),
        "ok?".padEnd(6)
    ].join(" | ")
);

console.log("-".repeat(LABEL_WIDTH + 12 + 12 + 10 + 6 + 12));

// -------------------------
let correct = 0;
let falsePositives = 0;
let falseNegatives = 0;

// -------------------------
samples.forEach(({ x, y, label }) => {
    const yTrue = y[0];
    const yPred = net.predict(x);

    const errPct = yTrue > 0 ? Math.abs(yPred - yTrue) / (Math.abs(yTrue)) * 100 : Math.abs(yPred) * 100;

    let isCorrect = false;

    if (task === "binary-classification") {
        const predictedClass = yPred >= THRESHOLD ? 1 : 0;
        isCorrect = predictedClass === yTrue;

        if (isCorrect) correct++;
        if (predictedClass === 1 && yTrue === 0) falsePositives++;
        if (predictedClass === 0 && yTrue === 1) falseNegatives++;
    } else {
        isCorrect = errPct <= REG_TOLERANCE;
        if (isCorrect) correct++;
    }

    const inputStr = label
        ? label.padEnd(LABEL_WIDTH)
        : JSON.stringify(x).slice(0, 37).padEnd(LABEL_WIDTH);

    console.log([
        inputStr,
        yTrue.toFixed(6).padEnd(12),
        yPred.toFixed(6).padEnd(12),
        errPct.toFixed(2).padEnd(10),
        (isCorrect ? "OK" : "KO").padEnd(6)
    ].join(" | "));
});

// -------------------------
const total = samples.length;
const accuracy = (correct / total) * 100;

console.log("\nRésumé:");
console.log(`Task           : ${task}`);
console.log(`Samples        : ${total}`);
console.log(`Accuracy       : ${accuracy.toFixed(2)} %`);

if (task === "binary-classification") {
    console.log(`False positives: ${falsePositives}`);
    console.log(`False negatives: ${falseNegatives}`);
}
