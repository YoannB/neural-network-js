import fs from "fs";
import { Network } from "./core/network.js";

// -------------------------
// Chargement du modèle
// -------------------------
const rawModel = JSON.parse(
  fs.readFileSync("./model/model.json", "utf-8")
);

const net = Network.load(rawModel);

const { inputSize, dataset } = rawModel.meta;

console.log(`Model dataset: ${dataset}`);
console.log(`Expected input size: ${inputSize}`);

// -------------------------
// Lecture entrée CLI
// -------------------------
const args = process.argv.slice(2);

if (args.length === 0) {
  console.error("Veuillez fournir une entrée");
  process.exit(1);
}

let input;

// Cas 1: tableau JSON
if (args.length === 1 && args[0].startsWith("[")) {
  try {
    input = JSON.parse(args[0]);
  } catch {
    console.error("Entrée JSON invalide");
    process.exit(1);
  }
}
// Cas 2: valeurs séparées (1 0)
else {
  input = args.map(Number);
}

if (input.some(v => !Number.isFinite(v))) {
  console.error("Entrée invalide :", input);
  process.exit(1);
}

// -------------------------
// Validation dimensions
// -------------------------
if (input.length !== inputSize) {
  console.error(
    `Dimension incorrecte: reçu ${input.length}, attendu ${inputSize}`
  );
  process.exit(1);
}

// -------------------------
// Prédiction
// -------------------------
const debug = net.debugPredict(input);

console.log("\nInput:", input);
console.log("Activations cachées :");

debug.hidden.forEach(h =>
  console.log(
    `  Neurone ${h.neuron} | z=${h.z} | activation=${h.activation}`
  )
);

console.log(`\nSortie finale y ≈ ${debug.output}`);
