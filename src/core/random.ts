// Générateur pseudo-aléatoire à graine (mulberry32).
// Avec la même graine, l'initialisation des poids, le mélange des données
// et le bruit d'entraînement sont identiques d'une exécution à l'autre.

export type RandomFn = () => number;

export function createRandom(seed: number): RandomFn {
    let state = seed >>> 0;
    return function random() {
        state = (state + 0x6d2b79f5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// Générateur partagé : non déterministe tant qu'aucune graine n'est fixée.
let current: RandomFn = Math.random;

export function setSeed(seed: number | null | undefined): void {
    current = seed === undefined || seed === null ? Math.random : createRandom(seed);
}

export function random(): number {
    return current();
}

// Mélange de Fisher-Yates, sur une copie.
export function shuffle<T>(array: readonly T[], rand: RandomFn = random): T[] {
    const copy = [...array];
    for (let i = copy.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
}
