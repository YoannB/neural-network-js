// src/datasets/images.js
import fs from "fs";
import path from "path";
import { splitSamples } from "./split.js";
import { imageInputSize, imageToVector } from "./image-features.js";

const IMAGE_DIR = "./tmp/oranges";

// Options : size (32 par défaut) et mode (gray, rgb ou hist), voir image-features.js.
// 80 % des images pour l'entraînement, 20 % pour le test.
export default async function loadImages({ seed, size = 32, mode = "gray" } = {}) {
    if (!fs.existsSync(IMAGE_DIR)) {
        throw new Error(
            `Dossier ${IMAGE_DIR} introuvable : place les images .jpg du jeu "images" dans ce dossier ` +
            `(voir scripts/download-oranges.js).`
        );
    }

    const files = fs.readdirSync(IMAGE_DIR).filter(f => f.endsWith(".jpg")).sort();
    const samples = [];

    for (const file of files) {
        const y = [file.includes("_ok.") ? 1 : 0];

        try {
            const x = await imageToVector(path.join(IMAGE_DIR, file), { size, mode });
            if (x.every(Number.isFinite)) {
                samples.push({ x, y, label: file });
            }
        } catch {
            // image illisible : ignorée
        }
    }

    return {
        name: "images",
        inputSize: imageInputSize({ size, mode }),
        outputSize: 1,
        task: "binary-classification",
        ...splitSamples(samples, { testRatio: 0.2, seed })
    };
}
