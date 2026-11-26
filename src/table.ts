import fs from "fs";
import { Network } from "./core/network.js";
import getDataset from "./datasets/index.js";
import type { ModelJSON, Sample } from "./core/types.js";

// -------------------------
const THRESHOLD = 0.5;
const REG_TOLERANCE = 5;

// -------------------------
const datasetName = process.argv[2] || "square";
const modelName = datasetName;

// -------------------------
const rawModel: ModelJSON = JSON.parse(
    fs.readFileSync(`./models/${modelName}.json`, "utf-8")
);

const net = Network.load(rawModel);

// -------------------------
// Même graine et mêmes options qu'à l'entraînement : on retrouve la même
// séparation train/test et la même préparation des données.
const dataset = await getDataset(datasetName, { seed: rawModel.config.seed, ...rawModel.config.dataset });
const samples = dataset.samples;
const task = dataset.task ?? "regression";

const trainSamples = samples;
const testSamples = dataset.testSamples ?? [];
const hasSplit = testSamples.length > 0;

// -------------------------
function evaluate(samples: Sample[], title: string): number {
    const LABEL_WIDTH = Math.max(
        40,
        ...samples.map(s => (s.label?.length ?? 0))
    );

    console.log(`\n=== ${title} ===`);
    console.log(
        [
            "Input".padEnd(LABEL_WIDTH),
            "y_true".padEnd(16),
            "y_pred".padEnd(16),
            "err_%".padEnd(10),
            "ok?".padEnd(6)
        ].join(" | ")
    );

    console.log("-".repeat(LABEL_WIDTH + 16 + 16 + 10 + 6 + 12 + 1));

    let correct = 0;
    let falsePositives = 0;
    let falseNegatives = 0;

    // >>> ajout pour le streak initial
    let streak = 0;
    let streakBroken = false;

    for (const { x, y, label } of samples) {
        let yTrueDisplay = "";
        let yPredDisplay = "";
        let errPct = 0;
        let isCorrect = false;

        if (task === "multiclass-classification") {
            const yTrueVec = y;
            const yPredVec = net.predictProbabilities(x);

            const trueClass = yTrueVec.indexOf(Math.max(...yTrueVec));
            const predClass = yPredVec.indexOf(Math.max(...yPredVec));

            yTrueDisplay = trueClass.toString();
            yPredDisplay = predClass.toString();

            isCorrect = trueClass === predClass;
            errPct = isCorrect ? 0 : 100;
        }

        else if (task === "binary-classification") {
            const yTrue = y[0];
            const yPred = net.predictValue(x);

            errPct = yTrue > 0
                ? Math.abs(yPred - yTrue) / Math.abs(yTrue) * 100
                : Math.abs(yPred) * 100;

            const predictedClass = yPred >= THRESHOLD ? 1 : 0;
            isCorrect = predictedClass === yTrue;

            if (predictedClass === 1 && yTrue === 0) falsePositives++;
            if (predictedClass === 0 && yTrue === 1) falseNegatives++;

            yTrueDisplay = yTrue.toFixed(6);
            yPredDisplay = yPred.toFixed(6);
        }

        else {
            const yTrue = y[0];
            const yPred = net.predictValue(x);

            errPct = yTrue > 0
                ? Math.abs(yPred - yTrue) / Math.abs(yTrue) * 100
                : Math.abs(yPred) * 100;

            isCorrect = errPct <= REG_TOLERANCE;

            yTrueDisplay = yTrue.toFixed(6);
            yPredDisplay = yPred.toFixed(6);
        }

        // >>> logique commune centralisée
        if (isCorrect) {
            correct++;
            if (!streakBroken) streak++;
        } else {
            streakBroken = true;
        }

        const inputStr = label
            ? label.padEnd(LABEL_WIDTH)
            : JSON.stringify(x).slice(0, 37).padEnd(LABEL_WIDTH);

        if (title !== "TRAIN")
            console.log([
                inputStr,
                yTrueDisplay.padEnd(16),
                yPredDisplay.padEnd(16),
                errPct.toFixed(2).padEnd(10),
                (isCorrect ? "OK" : "KO").padEnd(6)
            ].join(" | "));
    }

    const accuracy = (correct / samples.length) * 100;

    console.log(`\nRésumé (${title}) :`);
    console.log(`Task           : ${task}`);
    console.log(`Samples        : ${samples.length}`);
    console.log(`Accuracy       : ${accuracy.toFixed(2)} %`);

    if (title === "TEST") {
        console.log(`Initial streak : ${streak}`);
    }

    if (task === "binary-classification") {
        console.log(`False positives: ${falsePositives}`);
        console.log(`False negatives: ${falseNegatives}`);
    }

    return accuracy;
}

// -------------------------
if (hasSplit) {
    const trainAcc = evaluate(trainSamples, "TRAIN");
    const testAcc = evaluate(testSamples, "TEST");

    console.log("\n=== Global ===");
    console.log(`Train accuracy : ${trainAcc.toFixed(2)} %`);
    console.log(`Test accuracy  : ${testAcc.toFixed(2)} %`);
} else {
    evaluate(samples, "DATASET");
}
