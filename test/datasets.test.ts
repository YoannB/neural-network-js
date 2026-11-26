import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import getDataset, { datasetNames } from "../src/datasets/index.js";

const withoutImages = datasetNames.filter(name => name !== "images");

test("les jeux de données se chargent sans le dossier d'images", async () => {
    for (const name of withoutImages) {
        const dataset = await getDataset(name);
        assert.equal(dataset.name, name);
        assert.ok(dataset.samples.length > 0);
        for (const { x, y } of dataset.samples) {
            assert.equal(x.length, dataset.inputSize, `${name} : taille d'entrée`);
            assert.equal(y.length, dataset.outputSize, `${name} : taille de sortie`);
        }
    }
});

test("un jeu inconnu donne une erreur claire", async () => {
    await assert.rejects(getDataset("inconnu"), /Dataset inconnu: inconnu/);
});

test("rps : 80 % entraînement, 20 % test, séparation reproductible", async () => {
    const a = await getDataset("rps", { seed: 5 });
    const b = await getDataset("rps", { seed: 5 });
    const c = await getDataset("rps", { seed: 6 });

    assert.equal(a.samples.length, 480);
    assert.equal(a.testSamples?.length, 120);
    assert.deepEqual(a.testSamples, b.testSamples);
    assert.notDeepEqual(a.testSamples, c.testSamples);
});

test("images : message explicite si le dossier est absent", { skip: fs.existsSync("./tmp/oranges") }, async () => {
    await assert.rejects(getDataset("images"), /tmp\/oranges introuvable/);
});
