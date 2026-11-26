// Régression : apprendre y = sin(x) sur [-π, π], puis vérifier sur des points
// que le réseau n'a jamais vus. Lancer : npm run example:sine
import { Network, computePreprocessing, splitSamples, meanLoss, type Sample } from "../src/index.js";

const samples: Sample[] = Array.from({ length: 200 }, (_, i) => {
    const x = -Math.PI + (2 * Math.PI * i) / 199;
    return { x: [x], y: [Math.sin(x)] };
});

// 80 % pour apprendre, 20 % pour vérifier.
const { samples: train, testSamples: test } = splitSamples(samples, { testRatio: 0.2, seed: 7 });

const net = new Network({
    inputSize: 1,
    hiddenLayers: [16],
    activation: "tanh",
    task: "regression",
    optimizer: "adam",
    learningRate: 0.005,
    preprocessing: computePreprocessing(train),
    seed: 7
});

const result = net.fit(train, {
    epochs: 2000,
    onEpoch: ({ epoch, loss }) => {
        if (epoch % 500 === 0) console.log(`Époque ${epoch} - perte ${loss.toFixed(6)}`);
    }
});

// Écart moyen entre prédiction et valeur vraie (racine de l'erreur quadratique moyenne).
const rmse = (set: Sample[]) => Math.sqrt(meanLoss(net, set));
console.log(`\n${result.epochs} époques`);
console.log(`Écart moyen sur les ${train.length} points d'entraînement : ${rmse(train).toFixed(4)}`);
console.log(`Écart moyen sur les ${test.length} points jamais vus      : ${rmse(test).toFixed(4)}`);

console.log("\nx        sin(x)    prédit");
for (const { x, y } of test.slice(0, 5)) {
    const p = net.predictValue(x);
    console.log(`${x[0].toFixed(3).padStart(6)}   ${y[0].toFixed(4).padStart(7)}   ${p.toFixed(4).padStart(7)}`);
}
