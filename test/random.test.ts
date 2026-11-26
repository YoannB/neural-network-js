import { test } from "node:test";
import assert from "node:assert/strict";
import { createRandom, setSeed, random, shuffle } from "../src/core/random.js";

test("même graine, même suite de nombres", () => {
    const a = createRandom(123);
    const b = createRandom(123);
    for (let i = 0; i < 100; i++) assert.equal(a(), b());
});

test("graines différentes, suites différentes", () => {
    const a = createRandom(1);
    const b = createRandom(2);
    assert.notEqual(a(), b());
});

test("les nombres sont dans [0, 1[", () => {
    const rand = createRandom(7);
    for (let i = 0; i < 10000; i++) {
        const v = rand();
        assert.ok(v >= 0 && v < 1);
    }
});

test("setSeed rend le générateur partagé reproductible", () => {
    setSeed(42);
    const first = [random(), random(), random()];
    setSeed(42);
    assert.deepEqual([random(), random(), random()], first);
    setSeed(null);
});

test("shuffle garde les mêmes éléments et ne modifie pas l'original", () => {
    const original = [1, 2, 3, 4, 5, 6, 7, 8];
    const shuffled = shuffle(original, createRandom(3));
    assert.deepEqual(original, [1, 2, 3, 4, 5, 6, 7, 8]);
    assert.deepEqual([...shuffled].sort(), original);
});
