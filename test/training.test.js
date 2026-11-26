import { test } from "node:test";
import assert from "node:assert/strict";
import { Network } from "../src/core/network.js";
import { setSeed } from "../src/core/random.js";
import { trainNetwork } from "../src/core/training.js";
import getDataset from "../src/datasets/index.js";

function buildNetwork(dataset, seed) {
    setSeed(seed);
    return new Network({
        inputSize: dataset.inputSize,
        outputSize: dataset.outputSize,
        hiddenNeurons: 10,
        task: dataset.task,
        optimizer: "adam"
    });
}

const argmax = v => v.indexOf(Math.max(...v));

test("XOR est appris (classification binaire)", async () => {
    const xor = await getDataset("xor");
    const net = buildNetwork(xor, 1);
    const result = trainNetwork(net, xor.samples, { epochs: 10000, targetLoss: 0.001 });

    assert.ok(result.reachedTarget, `perte finale ${result.loss}`);
    for (const { x, y } of xor.samples) {
        assert.equal(net.predict(x) >= 0.5 ? 1 : 0, y[0], `XOR(${x})`);
    }
});

test("pierre-feuille-ciseaux : 100 % sur les exemples de test", async () => {
    const rps = await getDataset("rps", { seed: 1 });
    const net = buildNetwork(rps, 1);
    trainNetwork(net, rps.samples, { epochs: 500, targetLoss: 0.001 });

    for (const { x, y } of rps.testSamples) {
        assert.equal(argmax(net.predict(x)), argmax(y));
    }
});

test("même graine, même modèle", async () => {
    const xor = await getDataset("xor");
    const models = [1, 2].map(() => {
        const net = buildNetwork(xor, 7);
        trainNetwork(net, xor.samples, { epochs: 200, noise: 0.1 });
        return JSON.stringify(net.toJSON());
    });
    assert.equal(models[0], models[1]);
});

test("l'entraînement s'arrête à la perte cible", async () => {
    const xor = await getDataset("xor");
    const net = buildNetwork(xor, 1);
    const epochsSeen = [];
    const result = trainNetwork(net, xor.samples, {
        epochs: 10000,
        targetLoss: 0.01,
        onEpoch: ({ epoch }) => epochsSeen.push(epoch)
    });

    assert.ok(result.reachedTarget);
    assert.ok(result.loss <= 0.01);
    assert.ok(result.epochs < 10000);
    assert.equal(epochsSeen.length, result.epochs);
});

test("les pertes d'entropie croisée ne sont jamais négatives", () => {
    const binary = new Network({ inputSize: 1, task: "binary-classification" });
    assert.ok(binary.computeLoss(1, 1) >= 0);
    assert.ok(binary.computeLoss(0, 0) >= 0);

    const multi = new Network({ inputSize: 1, outputSize: 3, task: "multiclass-classification" });
    assert.ok(multi.computeLoss([1, 0, 0], [1, 0, 0]) >= 0);
});

test("un modèle sauvegardé puis rechargé donne les mêmes prédictions", async () => {
    const rps = await getDataset("rps");
    const net = buildNetwork(rps, 3);
    trainNetwork(net, rps.samples, { epochs: 20 });

    const loaded = Network.load(JSON.parse(JSON.stringify(net.toJSON())));
    for (const { x } of rps.testSamples.slice(0, 10)) {
        assert.deepEqual(loaded.predict(x), net.predict(x));
    }
});

for (const optimizer of ["adam", "rmsprop"]) {
    test(`reprendre un entraînement ${optimizer} sauvegardé donne le même modèle`, async () => {
        const xor = await getDataset("xor");
        setSeed(4);
        const options = { inputSize: 2, hiddenLayers: [6], task: "binary-classification", optimizer };

        const continuous = new Network(options);
        trainNetwork(continuous, xor.samples, { epochs: 100 });

        setSeed(4);
        const first = new Network(options);
        trainNetwork(first, xor.samples, { epochs: 50 });
        const resumed = Network.load(JSON.parse(JSON.stringify(first.toJSON())));
        trainNetwork(resumed, xor.samples, { epochs: 50 });

        assert.deepEqual(resumed.toJSON().layers, continuous.toJSON().layers);
    });
}
