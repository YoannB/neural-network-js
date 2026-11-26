import type { Dataset, Sample } from "../core/types.js";

function isPrime(n: number): boolean {
    if (n < 2) return false;
    for (let i = 2; i <= Math.sqrt(n); i++) {
        if (n % i === 0) return false;
    }
    return true;
}

function toBinaryVector(n: number, bits = 5): number[] {
    return n
        .toString(2)
        .padStart(bits, "0")
        .split("")
        .map(b => Number(b));
}

const samples: Sample[] = [];

for (let i = 0; i <= 13; i++) {
    samples.push({
        x: toBinaryVector(i, 5),  // passage en vecteur
        y: [isPrime(i) ? 1 : 0]
    });
}

export default {
    name: "primes",
    inputSize: 5,
    outputSize: 1,
    task: "binary-classification",
    samples
} satisfies Dataset;
