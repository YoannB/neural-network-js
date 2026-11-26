import { createRandom, shuffle } from "../core/random.js";

// Mélange les exemples avec une graine puis en réserve une part pour le test.
// Renvoie { samples, testSamples } : le réseau n'apprend que sur `samples`.
export function splitSamples(all, { testRatio = 0.2, seed = 42 } = {}) {
    const shuffled = shuffle(all, createRandom(seed));
    const testCount = Math.round(shuffled.length * testRatio);

    return {
        samples: shuffled.slice(testCount),
        testSamples: shuffled.slice(0, testCount)
    };
}
