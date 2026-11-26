export const relu = (x: number): number => Math.max(0, x);
export const reluDerivative = (x: number): number => (x > 0 ? 1 : 0);
export const leakyRelu = (x: number): number => (x > 0 ? x : 0.01 * x);
export const leakyReluDerivative = (x: number): number => (x > 0 ? 1 : 0.01);

export function tanh(x: number): number {
    return Math.tanh(x);
}

export function tanhDerivative(x: number): number {
    const t = Math.tanh(x);
    return 1 - t * t;
}

export function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
}

export function sigmoidDerivative(x: number): number {
    const s = sigmoid(x);
    return s * (1 - s);
}

/**
 * Softmax, appliqué à toute la couche de sortie : les sorties deviennent des
 * probabilités dont la somme vaut 1.
 */
export function softmax(vec: number[]): number[] {
    const max = Math.max(...vec); // stabilité numérique
    const exps = vec.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
}
