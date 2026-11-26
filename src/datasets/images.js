// src/datasets/images.js
import fs from "fs";
import path from "path";
import sharp from "sharp";
import { splitSamples } from "./split.js";

const IMAGE_DIR = "./tmp/oranges";
const SIZE = 32;

// 80 % des images pour l'entraînement, 20 % pour le test.
export default async function loadImages({ seed } = {}) {
    if (!fs.existsSync(IMAGE_DIR)) {
        throw new Error(
            `Dossier ${IMAGE_DIR} introuvable : place les images .jpg du jeu "images" dans ce dossier ` +
            `(voir scripts/download-oranges.js).`
        );
    }

    const files = fs.readdirSync(IMAGE_DIR).filter(f => f.endsWith(".jpg")).sort();

    const samples = [];
    let skipped = 0;

    for (const file of files) {
        const isOk = file.includes("_ok.");
        const y = [isOk ? 1 : 0];

        const fullPath = path.join(IMAGE_DIR, file);

        try {
            const buffer = await sharp(fullPath)
                .resize(SIZE, SIZE, { fit: "fill" })
                .grayscale()
                .raw()
                .toBuffer();

            if (buffer.length !== SIZE * SIZE) {
                skipped++;
                continue;
            }

            const x = Array.from(buffer, v => v / 255);

            if (x.some(v => !Number.isFinite(v))) {
                skipped++;
                continue;
            }

            samples.push({
                x, y,
                label: file
            });

        } catch (err) {
            skipped++;
        }
    }

    // console.log(`Images chargées: ${samples.length}, ignorées: ${skipped}`);

    return {
        name: "images",
        inputSize: SIZE * SIZE,
        outputSize: 1,
        task: "binary-classification",
        ...splitSamples(samples, { testRatio: 0.2, seed })
    };
}

