import { test } from "node:test";
import assert from "node:assert/strict";
import { Network } from "../src/core/network.js";
import { setSeed } from "../src/core/random.js";
import { trainNetwork, meanLoss } from "../src/core/training.js";
import { runExperiment } from "../src/experiment.js";
import getDataset from "../src/datasets/index.js";
import type { ExperimentOptions } from "../src/experiment.js";

const sumOfSquares = (net: Network) => net.layers.flatMap(l => l.neurons.flatMap(n => n.weights)).reduce((s, w) => s + w * w, 0);

test("la pénalité L2 garde des poids plus petits", async () => {
    const xor = await getDataset("xor");
    const train = (l2: number) => {
        setSeed(3);
        const net = new Network({ inputSize: 2, hiddenLayers: [8], task: "binary-classification", optimizer: "sgd", learningRate: 0.05, l2 });
        trainNetwork(net, xor.samples, { epochs: 2000 });
        return sumOfSquares(net);
    };
    assert.ok(train(0.01) < train(0));
});

test("arrêt sur validation : revient aux poids de la meilleure mesure", async () => {
    const xor = await getDataset("xor");
    // Validation aux étiquettes inversées : plus le réseau apprend XOR, plus elle se dégrade.
    const inverted = xor.samples.map(({ x, y }) => ({ x, y: [1 - y[0]] }));

    setSeed(1);
    const net = new Network({ inputSize: 2, hiddenLayers: [8], task: "binary-classification", optimizer: "adam" });
    const result = trainNetwork(net, xor.samples, { epochs: 10000, validation: inverted, patience: 200, evaluateEvery: 10 });

    assert.ok(result.stoppedOnValidation);
    assert.ok(result.epochs < 10000);
    assert.ok(result.epochs - (result.bestEpoch ?? 0) >= 200);
    assert.ok(Math.abs(meanLoss(net, inverted) - (result.validationLoss ?? NaN)) < 1e-12);
});

test("runExperiment : validation prise sur les exemples d'entraînement", async () => {
    const options: ExperimentOptions = { epochs: 50, layers: [6], activation: "tanh", lr: 0.0025, noise: 0, optimizer: "adam", seed: 1, targetLoss: 0, validation: 0.25, patience: 1000, datasetOptions: {} };
    let counts: number[] = [];
    await runExperiment("rps", options, {
        onStart: ({ dataset, fitSamples, validationSamples }) => {
            counts = [dataset.samples.length, fitSamples.length, validationSamples.length, dataset.testSamples?.length ?? 0];
        }
    });
    assert.deepEqual(counts, [480, 360, 120, 120]);
});

test("runExperiment : augmentation refusée pour un jeu qui ne la propose pas", async () => {
    const options: ExperimentOptions = { epochs: 1, layers: [2], activation: "tanh", lr: 0.0025, noise: 0, optimizer: "sgd", seed: 1, targetLoss: 0, augment: "flip", datasetOptions: {} };
    await assert.rejects(runExperiment("xor", options), /ne propose pas d'augmentation/);
});
