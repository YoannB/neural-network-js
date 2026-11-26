import { random } from "./random.js";

// Boucle d'entraînement : une époque = un passage sur tous les exemples.
// S'arrête après `epochs` époques, ou plus tôt si la perte moyenne de l'époque
// descend sous `targetLoss` (0 = jamais d'arrêt anticipé).
export function trainNetwork(net, samples, {
    epochs = 10000,
    noise = 0,
    targetLoss = 0,
    onEpoch = null
} = {}) {
    let loss = Infinity;
    let epoch = 0;
    let reachedTarget = false;

    while (epoch < epochs && !reachedTarget) {
        epoch++;
        let totalLoss = 0;

        for (const { x, y } of samples) {
            const input = noise > 0 ? x.map(v => v + (random() - 0.5) * noise) : x;
            totalLoss += net.train(input, y);
        }

        loss = totalLoss / samples.length;
        reachedTarget = targetLoss > 0 && loss <= targetLoss;

        onEpoch?.({ epoch, loss, last: epoch === epochs || reachedTarget });
    }

    return { epochs: epoch, loss, reachedTarget };
}
