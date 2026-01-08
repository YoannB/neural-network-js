export const relu = (x) => Math.max(0, x);
export const reluDerivative = (x) => (x > 0 ? 1 : 0);
