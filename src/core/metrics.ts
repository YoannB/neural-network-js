import { meanLoss } from "./training.js";
import type { Network } from "./network.js";
import type { Sample } from "./types.js";

export interface Evaluation {
    /** Perte moyenne (la même que celle minimisée pendant l'entraînement). */
    loss: number;
    /** Part des exemples bien prédits, en %. */
    accuracy: number;
    samples: number;
}

const argmax = (v: number[]): number => v.indexOf(Math.max(...v));

/**
 * Part des exemples bien prédits, en % : seuil 0,5 en classification binaire,
 * classe la plus probable en multiclasse, 5 % d'erreur relative en régression.
 */
export function accuracy(net: Network, samples: Sample[]): number {
    if (samples.length === 0) return NaN;

    let correct = 0;

    for (const { x, y } of samples) {
        const p = net.predict(x);
        if (Array.isArray(p)) {
            if (argmax(p) === argmax(y)) correct++;
        } else if (net.task === "binary-classification") {
            if ((p >= 0.5 ? 1 : 0) === y[0]) correct++;
        } else {
            const err = y[0] > 0 ? Math.abs(p - y[0]) / Math.abs(y[0]) * 100 : Math.abs(p) * 100;
            if (err <= 5) correct++;
        }
    }

    return (correct / samples.length) * 100;
}

/** Perte moyenne et part des exemples bien prédits. */
export function evaluate(net: Network, samples: Sample[]): Evaluation {
    return {
        loss: samples.length > 0 ? meanLoss(net, samples) : NaN,
        accuracy: accuracy(net, samples),
        samples: samples.length
    };
}
