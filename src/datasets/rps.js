// datasets/rps.js

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
function buildSamples(repeat = 200) {
    const samples = [];

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

const all = buildSamples(200);

// Shuffle simple
for (let i = all.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [all[i], all[j]] = [all[j], all[i]];
}

const split = Math.floor(all.length * 0.8);

export default {
    name: "rps",
    inputSize: 3,
    outputSize: 3,
    task: "multiclass-classification",

    samples: all.slice(0, split),
    testSamples: all.slice(split)
};
