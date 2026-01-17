import fs from "fs";
import sharp from "sharp";
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

// ================================
// CAS 1 : fichier image
// ================================
if (fs.existsSync(args[0])) {
  const file = args[0];

  const size = Math.sqrt(inputSize);
  if (!Number.isInteger(size)) {
    console.error("InputSize incompatible avec une image carrée:", inputSize);
    process.exit(1);
  }

  try {
    const buffer = await sharp(file)
      .resize(size, size)
      .grayscale()
      .raw()
      .toBuffer();

    input = Array.from(buffer).map(v => v / 255);
  } catch (err) {
    console.error("Erreur lecture image:", err.message);
    process.exit(1);
  }
}

// ================================
// CAS 2 : tableau JSON
// ================================
else if (args.length === 1 && args[0].startsWith("[")) {
  try {
    input = JSON.parse(args[0]);
  } catch {
    console.error("Entrée JSON invalide");
    process.exit(1);
  }
}

// ================================
// CAS 3 : valeurs séparées
// ================================
else {
  input = args.map(Number);
}

if (input.some(v => !Number.isFinite(v))) {
  console.error("Entrée invalide :", input.slice(0, 10));
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

console.log("\nActivations cachées :");

debug.hidden.forEach(h =>
  console.log(
    `  Neurone ${h.neuron} | z=${h.z} | activation=${h.activation}`
  )
);

console.log(`\nSortie finale y ≈ ${debug.output}`);
