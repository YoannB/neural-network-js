import { test } from "node:test";
import assert from "node:assert/strict";
import sharp from "sharp";
import { imageToVector, imageInputSize, type ImageMode } from "../src/datasets/image-features.js";

// Image carrée de côté n : moitié gauche noire, moitié droite rouge pur.
async function makeImage(n = 2) {
    const pixels = Buffer.alloc(n * n * 3);
    for (let row = 0; row < n; row++) {
        for (let col = n / 2; col < n; col++) pixels[(row * n + col) * 3] = 255;
    }
    return sharp(pixels, { raw: { width: n, height: n, channels: 3 } }).png().toBuffer();
}

test("tailles d'entrée selon le mode", () => {
    assert.equal(imageInputSize({ size: 32, mode: "gray" }), 1024);
    assert.equal(imageInputSize({ size: 16, mode: "rgb" }), 768);
    assert.equal(imageInputSize({ mode: "hist" }), 20);
    assert.throws(() => imageInputSize({ mode: "inconnu" as ImageMode }), /Mode d'image inconnu/);
});

test("mode rgb : trois valeurs par pixel, entre 0 et 1", async () => {
    const x = await imageToVector(await makeImage(), { size: 2, mode: "rgb" });
    assert.deepEqual(x, [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]);
});

test("miroir horizontal", async () => {
    const x = await imageToVector(await makeImage(), { size: 2, mode: "rgb", flop: true });
    assert.deepEqual(x, [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]);
});

test("mode hist : la moitié des pixels dans la teinte rouge", async () => {
    // 32 × 32 : la taille de travail du mode hist, donc sans redimensionnement.
    const x = await imageToVector(await makeImage(32), { mode: "hist" });
    assert.equal(x.length, 20);
    assert.ok(Math.abs(x[0] - 0.5) < 1e-9, `teinte rouge : ${x[0]}`);
    assert.ok(Math.abs(x.slice(1, 12).reduce((a, b) => a + b, 0)) < 1e-9);
    // saturation et luminosité : chaque groupe d'intervalles couvre tous les pixels
    assert.ok(Math.abs(x.slice(12, 16).reduce((a, b) => a + b, 0) - 1) < 1e-9);
    assert.ok(Math.abs(x.slice(16, 20).reduce((a, b) => a + b, 0) - 1) < 1e-9);
});
