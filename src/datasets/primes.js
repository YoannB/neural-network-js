function isPrime(n) {
    if (n < 2) return false;
    for (let i = 2; i <= Math.sqrt(n); i++) {
        if (n % i === 0) return false;
    }
    return true;
}

function toBinaryVector(n, bits = 5) {
    return n
        .toString(2)
        .padStart(bits, "0")
        .split("")
        .map(b => Number(b));
}

const samples = [];

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
};
