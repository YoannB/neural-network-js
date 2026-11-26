import { createRandom, shuffle } from "../core/random.js";
import type { Sample } from "../core/types.js";

/**
 * Mélange les exemples avec une graine puis en réserve une part pour le test.
 * Renvoie { samples, testSamples } : le réseau n'apprend que sur `samples`.
 */
export function splitSamples(
    all: Sample[],
    { testRatio = 0.2, seed = 42 }: { testRatio?: number | undefined; seed?: number | undefined } = {}
): { samples: Sample[]; testSamples: Sample[] } {
    const shuffled = shuffle(all, createRandom(seed));
    const testCount = Math.round(shuffled.length * testRatio);

    return {
        samples: shuffled.slice(testCount),
        testSamples: shuffled.slice(0, testCount)
    };
}
