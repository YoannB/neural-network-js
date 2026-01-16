export const relu = (x) => Math.max(0, x);
export const reluDerivative = (x) => (x > 0 ? 1 : 0);
export const leakyRelu = (x) => (x > 0 ? x : 0.01 * x);
export const leakyReluDerivative = (x) => (x > 0 ? 1 : 0.01);
export function tanh(x) {
    return Math.tanh(x);
}
export function tanhDerivative(x) {
    const t = Math.tanh(x);
    return 1 - t * t;
}
export function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

export function sigmoidDerivative(x) {
    const s = sigmoid(x);
    return s * (1 - s);
}