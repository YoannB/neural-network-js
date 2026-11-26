import type { Preprocessing, Sample } from "./types.js";

/**
 * Les entrées sont divisées par leur plus grande valeur absolue (calculée sur
 * les exemples d'entraînement). Les sorties ne sont pas normalisées.
 */
export function computePreprocessing(samples: Sample[]): Preprocessing {
    let maxX = 0;
    for (const s of samples) {
        for (const v of s.x) {
            maxX = Math.max(maxX, Math.abs(v));
        }
    }
    return { xScale: maxX || 1 };
}
