// Tests de la bibliothèque telle qu'un utilisateur la voit : uniquement src/index.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import * as lib from "../src/index.js";
import { Network, random, setSeed } from "./api-helpers.js";

test("le point d'entrée exporte l'API publique", () => {
    const expected = [
        "Network", "HIDDEN_ACTIVATIONS", "trainNetwork", "meanLoss", "accuracy", "evaluate",
        "computePreprocessing", "createRandom", "setSeed", "shuffle", "splitSamples",
        "Layer", "Neuron", "activations"
    ];
    for (const name of expected) {
        assert.ok(name in lib, `export manquant : ${name}`);
    }
});

test("seed : même graine, mêmes poids ; le générateur partagé n'est pas touché", () => {
    const options = { inputSize: 3, hiddenLayers: [5], task: "regression" } as const;
    const a = new Network({ ...options, seed: 1 });
    const b = new Network({ ...options, seed: 1 });
    const c = new Network({ ...options, seed: 2 });
    assert.deepEqual(a.toJSON().layers, b.toJSON().layers);
    assert.notDeepEqual(a.toJSON().layers, c.toJSON().layers);

    setSeed(9);
    const expected = [random(), random()];
    setSeed(9);
    new Network({ ...options, seed: 1 });
    assert.deepEqual([random(), random()], expected);
    setSeed(null);
});

test("fit puis evaluate sur XOR", () => {
    const xor: lib.Sample[] = [
        { x: [0, 0], y: [0] }, { x: [0, 1], y: [1] },
        { x: [1, 0], y: [1] }, { x: [1, 1], y: [0] }
    ];
    const net = new lib.Network({ inputSize: 2, hiddenLayers: [8], task: "binary-classification", optimizer: "adam", seed: 1 });
    const result = net.fit(xor, { epochs: 10000, targetLoss: 0.001 });

    assert.ok(result.reachedTarget);
    const evaluation = net.evaluate(xor);
    assert.equal(evaluation.accuracy, 100);
    assert.equal(evaluation.samples, 4);
    assert.ok(evaluation.loss <= 0.001 + 1e-9);
    assert.ok(Number.isNaN(net.evaluate([]).loss));
});

test("predictValue refuse la classification multiclasse", () => {
    const net = new lib.Network({ inputSize: 3, outputSize: 3, task: "multiclass-classification", seed: 1 });
    assert.equal(net.predictProbabilities([1, 0, 0]).length, 3);
    assert.throws(() => net.predictValue([1, 0, 0]), /predictProbabilities/);
});

test("un modèle sérialisé en JSON se recharge à l'identique", () => {
    const net = new lib.Network({ inputSize: 2, hiddenLayers: [4, 3], task: "regression", seed: 3 });
    const copy = lib.Network.load(JSON.parse(JSON.stringify(net)));
    assert.equal(copy.predictValue([0.3, -0.2]), net.predictValue([0.3, -0.2]));
});

test("le cœur n'importe ni module Node.js ni sharp", () => {
    // Les tests s'exécutent depuis dist/test : le code compilé est dans dist/src.
    const srcDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "src");
    const files = [
        "index.js",
        "datasets/split.js",
        ...fs.readdirSync(path.join(srcDir, "core")).filter(f => f.endsWith(".js")).map(f => `core/${f}`)
    ];
    for (const file of files) {
        const code = fs.readFileSync(path.join(srcDir, file), "utf8");
        const imports = [...code.matchAll(/from\s+"([^"]+)"/g)].map(m => m[1]);
        for (const spec of imports) {
            assert.ok(spec.startsWith("."), `${file} importe ${spec}`);
        }
    }
});
