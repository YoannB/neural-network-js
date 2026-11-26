// Vérification numérique de la rétropropagation : après un pas de descente
// de gradient simple (SGD), chaque poids a bougé de -lr × gradient. On compare
// ce gradient à celui obtenu par différences finies sur la perte.
import { test } from "node:test";
import assert from "node:assert/strict";
import { Network } from "../src/core/network.js";
import { setSeed } from "../src/core/random.js";
import type { HiddenActivation, Task } from "../src/core/types.js";

const LR = 1e-3;
const H = 1e-5;

function loss(net: Network, x: number[], y: number[]): number {
    const multiclass = net.task === "multiclass-classification";
    const value = net.computeLoss(net.predict(x), multiclass ? y : y[0]);
    // En régression, le gradient p - y correspond à la perte (p - y)² / 2.
    return net.task === "regression" ? value / 2 : value;
}

function clone(net: Network): Network {
    return Network.load(JSON.parse(JSON.stringify(net.toJSON())));
}

function checkGradients(
    task: Task, outputSize: number, x: number[], y: number[],
    { hiddenLayers = [4], activation = "tanh" }: { hiddenLayers?: number[]; activation?: HiddenActivation } = {}
): void {
    setSeed(5);
    const net = new Network({ inputSize: x.length, hiddenLayers, activation, outputSize, task, learningRate: LR, optimizer: "sgd" });

    const trained = clone(net);
    trained.train(x, y);

    net.layers.forEach((netLayer, layer) => {
        netLayer.neurons.forEach((neuron, n) => {
            neuron.weights.forEach((w, i) => {
                neuron.weights[i] = w + H;
                const plus = loss(net, x, y);
                neuron.weights[i] = w - H;
                const minus = loss(net, x, y);
                neuron.weights[i] = w;

                const numerical = (plus - minus) / (2 * H);
                const analytic = (w - trained.layers[layer].neurons[n].weights[i]) / LR;

                assert.ok(
                    Math.abs(numerical - analytic) < 1e-6 + 1e-4 * Math.abs(numerical),
                    `${task}, couche ${layer}, neurone ${n}, poids ${i} : numérique ${numerical}, rétropropagation ${analytic}`
                );
            });
        });
    });
}

test("gradients corrects : régression", () => {
    checkGradients("regression", 1, [0.3, -0.8], [0.5]);
});

test("gradients corrects : classification binaire (sigmoïde)", () => {
    checkGradients("binary-classification", 1, [0.3, -0.8], [1]);
    checkGradients("binary-classification", 1, [-0.6, 0.2], [0]);
});

test("gradients corrects : classification multiclasse (softmax)", () => {
    checkGradients("multiclass-classification", 3, [0.3, -0.8, 0.5], [0, 1, 0]);
});

test("gradients corrects avec plusieurs couches cachées", () => {
    const deep = { hiddenLayers: [5, 4, 3] };
    checkGradients("regression", 1, [0.3, -0.8], [0.5], deep);
    checkGradients("binary-classification", 1, [0.3, -0.8], [1], deep);
    checkGradients("multiclass-classification", 3, [0.3, -0.8, 0.5], [0, 0, 1], deep);
});

test("gradients corrects pour chaque activation cachée", () => {
    for (const activation of ["tanh", "relu", "leakyrelu", "sigmoid"] as const) {
        checkGradients("binary-classification", 1, [0.3, -0.8], [1], { hiddenLayers: [4, 3], activation });
    }
});
