import { random } from "./random.js";

// Perte moyenne du réseau sur des exemples, sans entraînement.
export function meanLoss(net, samples) {
    const multiclass = net.task === "multiclass-classification";
    const total = samples.reduce((sum, { x, y }) =>
        sum + net.computeLoss(net.predict(x), multiclass ? y : y[0]), 0);
    return total / samples.length;
}

// Copie des poids et biais de toutes les couches (pour revenir au meilleur état).
function snapshot(net) {
    return net.layers.map(layer => layer.neurons.map(n => ({ weights: [...n.weights], bias: n.bias })));
}

function restore(net, saved) {
    saved.forEach((layer, l) => layer.forEach((s, i) => {
        net.layers[l].neurons[i].weights = [...s.weights];
        net.layers[l].neurons[i].bias = s.bias;
    }));
}

// Boucle d'entraînement : une époque = un passage sur tous les exemples.
// S'arrête après `epochs` époques, ou plus tôt :
//   - si la perte moyenne de l'époque descend sous `targetLoss` (0 = désactivé) ;
//   - si des exemples de `validation` sont donnés : leur perte est mesurée toutes
//     les `evaluateEvery` époques, l'entraînement s'arrête quand elle ne s'est pas
//     améliorée depuis `patience` époques, et le réseau revient aux poids de la
//     meilleure mesure.
export function trainNetwork(net, samples, {
    epochs = 10000,
    noise = 0,
    targetLoss = 0,
    validation = null,
    patience = 500,
    evaluateEvery = 50,
    onEpoch = null
} = {}) {
    let loss = Infinity;
    let epoch = 0;
    let reachedTarget = false;
    let stoppedOnValidation = false;

    const useValidation = validation?.length > 0;
    let best = useValidation ? { loss: meanLoss(net, validation), epoch: 0, weights: snapshot(net) } : null;

    while (epoch < epochs && !reachedTarget && !stoppedOnValidation) {
        epoch++;
        let totalLoss = 0;

        for (const { x, y } of samples) {
            const input = noise > 0 ? x.map(v => v + (random() - 0.5) * noise) : x;
            totalLoss += net.train(input, y);
        }

        loss = totalLoss / samples.length;
        reachedTarget = targetLoss > 0 && loss <= targetLoss;

        let validationLoss;
        if (useValidation && (epoch % evaluateEvery === 0 || epoch === epochs)) {
            validationLoss = meanLoss(net, validation);
            if (validationLoss < best.loss) {
                best = { loss: validationLoss, epoch, weights: snapshot(net) };
            } else if (epoch - best.epoch >= patience) {
                stoppedOnValidation = true;
            }
        }

        onEpoch?.({ epoch, loss, validationLoss, last: epoch === epochs || reachedTarget || stoppedOnValidation });
    }

    if (useValidation) {
        restore(net, best.weights);
    }

    return {
        epochs: epoch,
        loss,
        reachedTarget,
        stoppedOnValidation,
        bestEpoch: best?.epoch,
        validationLoss: best?.loss
    };
}
