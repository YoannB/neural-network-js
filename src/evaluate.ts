import { getArg, parseTrainingOptions } from "./cli.js";
import { runExperiment } from "./experiment.js";
import { accuracy } from "./core/metrics.js";

// node dist/src/evaluate.js <dataset> --seeds 1-10 [options d'entraînement]
// Entraîne une fois par graine. Chaque graine change la séparation
// entraînement / test et l'initialisation des poids ; on regarde la moyenne
// et la dispersion du score sur les exemples de test plutôt qu'un seul essai.

const datasetName = process.argv[2] || "square";
const args = process.argv.slice(3);
const options = parseTrainingOptions(args);
const seeds = parseSeeds(String(getArg(args, "seeds") ?? "1-5"));

// "1-5" -> [1, 2, 3, 4, 5] ; "1,3,8" -> [1, 3, 8]
function parseSeeds(spec: string): number[] {
    return spec.split(",").flatMap(part => {
        const [from, to] = part.split("-").map(Number);
        return to === undefined ? [from] : Array.from({ length: to - from + 1 }, (_, i) => from + i);
    });
}

const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;
const std = (values: number[]) => Math.sqrt(mean(values.map(v => (v - mean(values)) ** 2)));
const pct = (v: number) => (Number.isFinite(v) ? `${v.toFixed(1)} %` : "-").padStart(8);

console.log(`Dataset : ${datasetName}`);
console.log(`Options : ${args.filter((a, i) => a !== "--seeds" && args[i - 1] !== "--seeds").join(" ") || "(défaut)"}`);
console.log("");
console.log("Graine | Époques | Train    | Test");

const trainScores: number[] = [];
const testScores: number[] = [];

for (const seed of seeds) {
    const { dataset, net, result } = await runExperiment(datasetName, { ...options, seed });
    const train = accuracy(net, dataset.samples);
    const test = accuracy(net, dataset.testSamples ?? []);
    trainScores.push(train);
    testScores.push(test);
    console.log(`${String(seed).padStart(6)} | ${String(result.epochs).padStart(7)} | ${pct(train)} | ${pct(test)}`);
}

console.log("");
console.log(`Train : ${mean(trainScores).toFixed(1)} % ± ${std(trainScores).toFixed(1)}`);
if (testScores.every(Number.isFinite)) {
    console.log(`Test  : ${mean(testScores).toFixed(1)} % ± ${std(testScores).toFixed(1)} (min ${Math.min(...testScores).toFixed(1)}, max ${Math.max(...testScores).toFixed(1)})`);
} else {
    console.log("Test  : ce jeu de données n'a pas d'exemples de test");
}
