import fs from "fs";
import sharp from "sharp";
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

// image
if (fs.existsSync(args[0])) {
  const size = Math.sqrt(inputSize);
  const buffer = await sharp(args[0])
    .resize(size, size)
    .grayscale()
    .raw()
    .toBuffer();

  input = Array.from(buffer).map(v => v / 255);
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

console.log("\nHidden:");
debug.hidden.forEach(h =>
  console.log(`Neuron ${h.neuron} | a=${h.weight}`)
);

const output = Array.isArray(debug.output) ? JSON.stringify(debug.output) : debug.output;
console.log(`\nOutput: ${output}`);
