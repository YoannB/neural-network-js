import { test } from "node:test";
import assert from "node:assert/strict";
import { Network } from "../src/core/network.js";
import { setSeed } from "../src/core/random.js";
import type { HiddenActivation, ModelJSON, Task } from "../src/core/types.js";

test("couches cachées configurables", () => {
    const net = new Network({ inputSize: 3, hiddenLayers: [16, 8], outputSize: 2, task: "multiclass-classification" });
    assert.deepEqual(net.layers.map(l => l.neurons.length), [16, 8, 2]);
    assert.deepEqual(net.layers.map(l => l.neurons[0].weights.length), [3, 16, 8]);
    assert.deepEqual(net.layers.map(l => l.activation), ["tanh", "tanh", "softmax"]);
});

test("hiddenNeurons reste accepté pour une seule couche cachée", () => {
    const net = new Network({ inputSize: 2, hiddenNeurons: 6, task: "binary-classification" });
    assert.deepEqual(net.layers.map(l => l.neurons.length), [6, 1]);
    assert.equal(net.output.activation, "sigmoid");
});

test("configurations invalides refusées", () => {
    assert.throws(() => new Network({ hiddenLayers: [] }), /Couches cachées invalides/);
    assert.throws(() => new Network({ hiddenLayers: [4, 0] }), /Couches cachées invalides/);
    assert.throws(() => new Network({ activation: "inconnue" as HiddenActivation }), /Activation inconnue/);
    assert.throws(() => new Network({ task: "inconnue" as Task }), /Tâche inconnue/);
});

test("un modèle à l'ancien format (hidden / output) se charge encore", () => {
    setSeed(2);
    const net = new Network({ inputSize: 2, hiddenNeurons: 3, task: "binary-classification" });
    const { layers = [], ...saved } = net.toJSON();
    const legacy: ModelJSON = {
        ...saved,
        config: { ...saved.config, hiddenLayers: undefined, hiddenNeurons: 3 },
        hidden: layers[0],
        output: layers[1]
    };

    const loaded = Network.load(JSON.parse(JSON.stringify(legacy)));
    assert.equal(loaded.predict([0.2, 0.7]), net.predict([0.2, 0.7]));
});
