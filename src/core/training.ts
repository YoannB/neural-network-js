import { random } from "./random.js";
import type { Network } from "./network.js";
import type { Sample } from "./types.js";

export interface EpochInfo {
    epoch: number;
    loss: number;
    validationLoss?: number;
    last: boolean;
}

export interface TrainOptions {
    epochs?: number;
    /** Amplitude du bruit ajouté aux entrées à chaque passage (0 = aucun). */
    noise?: number;
    /** Arrêt quand la perte moyenne d'une époque descend sous cette valeur (0 = désactivé). */
    targetLoss?: number;
    /** Exemples de validation : arrêt quand leur perte ne s'améliore plus. */
    validation?: Sample[] | null;
    /** Nombre d'époques sans amélioration de la validation avant l'arrêt. */
    patience?: number;
    /** La perte de validation est mesurée toutes les evaluateEvery époques. */
    evaluateEvery?: number;
    onEpoch?: ((info: EpochInfo) => void) | null;
}

export interface TrainResult {
    epochs: number;
    loss: number;
    reachedTarget: boolean;
    stoppedOnValidation: boolean;
    bestEpoch: number | undefined;
    validationLoss: number | undefined;
}

/** Perte moyenne du réseau sur des exemples, sans entraînement. */
export function meanLoss(net: Network, samples: Sample[]): number {
    const multiclass = net.task === "multiclass-classification";
    const total = samples.reduce((sum, { x, y }) =>
        sum + net.computeLoss(net.predict(x), multiclass ? y : y[0]), 0);
    return total / samples.length;
}

type WeightsSnapshot = { weights: number[]; bias: number }[][];

// Copie des poids et biais de toutes les couches (pour revenir au meilleur état).
function snapshot(net: Network): WeightsSnapshot {
    return net.layers.map(layer => layer.neurons.map(n => ({ weights: [...n.weights], bias: n.bias })));
}

function restore(net: Network, saved: WeightsSnapshot): void {
    saved.forEach((layer, l) => layer.forEach((s, i) => {
        net.layers[l].neurons[i].weights = [...s.weights];
        net.layers[l].neurons[i].bias = s.bias;
    }));
}

/**
 * Boucle d'entraînement : une époque = un passage sur tous les exemples.
 * S'arrête après `epochs` époques, ou plus tôt :
 *   - si la perte moyenne de l'époque descend sous `targetLoss` (0 = désactivé) ;
 *   - si des exemples de `validation` sont donnés : leur perte est mesurée toutes
 *     les `evaluateEvery` époques, l'entraînement s'arrête quand elle ne s'est pas
 *     améliorée depuis `patience` époques, et le réseau revient aux poids de la
 *     meilleure mesure.
 */
export function trainNetwork(net: Network, samples: Sample[], {
    epochs = 10000,
    noise = 0,
    targetLoss = 0,
    validation = null,
    patience = 500,
    evaluateEvery = 50,
    onEpoch = null
}: TrainOptions = {}): TrainResult {
    let loss = Infinity;
    let epoch = 0;
    let reachedTarget = false;
    let stoppedOnValidation = false;

    const validationSamples = validation ?? [];
    const useValidation = validationSamples.length > 0;
    let best = useValidation
        ? { loss: meanLoss(net, validationSamples), epoch: 0, weights: snapshot(net) }
        : null;

    while (epoch < epochs && !reachedTarget && !stoppedOnValidation) {
        epoch++;
        let totalLoss = 0;

        for (const { x, y } of samples) {
            const input = noise > 0 ? x.map(v => v + (random() - 0.5) * noise) : x;
            totalLoss += net.train(input, y);
        }

        loss = totalLoss / samples.length;
        reachedTarget = targetLoss > 0 && loss <= targetLoss;

        let validationLoss: number | undefined;
        if (best && (epoch % evaluateEvery === 0 || epoch === epochs)) {
            validationLoss = meanLoss(net, validationSamples);
            if (validationLoss < best.loss) {
                best = { loss: validationLoss, epoch, weights: snapshot(net) };
            } else if (epoch - best.epoch >= patience) {
                stoppedOnValidation = true;
            }
        }

        onEpoch?.({
            epoch,
            loss,
            ...(validationLoss !== undefined ? { validationLoss } : {}),
            last: epoch === epochs || reachedTarget || stoppedOnValidation
        });
    }

    if (best) {
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
