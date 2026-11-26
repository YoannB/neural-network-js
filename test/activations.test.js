import { test } from "node:test";
import assert from "node:assert/strict";
import { sigmoid, sigmoidDerivative, tanhDerivative, softmax } from "../src/core/activations.js";

const numericalDerivative = (f, x, h = 1e-6) => (f(x + h) - f(x - h)) / (2 * h);

test("dérivées de la sigmoïde et de tanh", () => {
    for (const x of [-3, -0.5, 0, 0.7, 2.5]) {
        assert.ok(Math.abs(sigmoidDerivative(x) - numericalDerivative(sigmoid, x)) < 1e-8);
        assert.ok(Math.abs(tanhDerivative(x) - numericalDerivative(Math.tanh, x)) < 1e-8);
    }
});

test("softmax : probabilités positives dont la somme vaut 1", () => {
    const p = softmax([2, 1, -1]);
    assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
    assert.ok(p.every(v => v > 0));
    assert.ok(p[0] > p[1] && p[1] > p[2]);
});

test("softmax reste stable avec de grandes valeurs", () => {
    const p = softmax([1000, 1000]);
    assert.deepEqual(p, [0.5, 0.5]);
});
