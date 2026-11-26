import fs from "fs";
import { imageToVector } from "./datasets/image-features.js";
import { Network } from "./core/network.js";

const dataset = process.argv[2];
if (!dataset) {
  console.error("Usage: exec.js <dataset> <input>");
  process.exit(1);
}

let inputStartIndex = 3;

const modelName = dataset;

console.log(`Read ./models/${modelName}.json`);

const rawModel = JSON.parse(
  fs.readFileSync(`./models/${modelName}.json`, "utf-8")
);

const net = Network.load(rawModel);
const { inputSize } = rawModel.meta;

const args = process.argv.slice(inputStartIndex);
if (!args.length) {
  console.error("Veuillez fournir une entrée");
  process.exit(1);
}

let input;

// image : même préparation qu'à l'entraînement. Les options sont enregistrées
// dans le modèle ; un ancien modèle est en niveaux de gris, de côté sqrt(inputSize).
if (fs.existsSync(args[0])) {
  const imageOptions = rawModel.config.dataset ?? { size: Math.sqrt(inputSize), mode: "gray" };
  input = await imageToVector(args[0], imageOptions);
}

// json
else if (args.length === 1 && args[0].startsWith("[")) {
  input = JSON.parse(args[0]);
}

// numbers
else {
  input = args.map(Number);
}

if (input.length !== inputSize) {
  console.error(`Dimension incorrecte: ${input.length} vs ${inputSize}`);
  process.exit(1);
}

console.log("Debug predict :", input);

const debug = net.debugPredict(input);

debug.hidden.forEach((layer, l) => {
  console.log(`\nHidden layer ${l + 1}:`);
  layer.forEach((a, i) => console.log(`Neuron ${i} | a=${a}`));
});

const output = Array.isArray(debug.output) ? JSON.stringify(debug.output) : debug.output;
console.log(`\nOutput: ${output}`);
