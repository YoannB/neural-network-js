// XOR : classification binaire, le plus petit problème qu'un seul neurone ne
// sait pas résoudre. Lancer : npm run example:xor
import { Network, type Sample } from "../src/index.js";

const xor: Sample[] = [
    { x: [0, 0], y: [0] },
    { x: [0, 1], y: [1] },
    { x: [1, 0], y: [1] },
    { x: [1, 1], y: [0] }
];

const net = new Network({
    inputSize: 2,
    hiddenLayers: [8],
    task: "binary-classification",
    optimizer: "adam",
    seed: 1
});

const result = net.fit(xor, { epochs: 10000, targetLoss: 0.001 });
console.log(`Perte ${result.loss.toFixed(6)} après ${result.epochs} époques`);

for (const { x, y } of xor) {
    console.log(`${x.join(" XOR ")} = ${y[0]}   prédit : ${net.predictValue(x).toFixed(4)}`);
}

const { accuracy } = net.evaluate(xor);
console.log(`Exactitude : ${accuracy} %`);
