// Pierre-feuille-ciseaux : pour chaque coup, le réseau doit donner le coup qui le bat.
import { splitSamples } from "./split.js";
import type { DatasetOptions } from "./index.js";
import type { Dataset, Sample } from "../core/types.js";

const ROCK = [1, 0, 0];
const PAPER = [0, 1, 0];
const SCISSORS = [0, 0, 1];

// Ce que le modèle doit apprendre :
// Input  -> Output (coup gagnant)
const RULES = [
    { x: ROCK, y: PAPER, label: "rock -> paper" },     // feuille bat pierre
    { x: PAPER, y: SCISSORS, label: "paper -> scissors" }, // ciseau bat feuille
    { x: SCISSORS, y: ROCK, label: "scissors -> rock" }   // pierre bat ciseau
];

// On duplique les données pour faciliter l'entraînement
function buildSamples(repeat = 200): Sample[] {
    const samples: Sample[] = [];

    for (let i = 0; i < repeat; i++) {
        for (const r of RULES) {
            samples.push({
                x: [...r.x],
                y: [...r.y],
                label: r.label
            });
        }
    }

    return samples;
}

// 80 % des exemples pour l'entraînement, 20 % pour le test.
export default function rps({ seed }: DatasetOptions = {}): Dataset {
    return {
        name: "rps",
        inputSize: 3,
        outputSize: 3,
        task: "multiclass-classification",
        ...splitSamples(buildSamples(200), { testRatio: 0.2, seed })
    };
}
