import fs from "fs";
import { Network } from "./core/network.js";
import getDataset from "./datasets/index.js";

// -------------------------
// CLI parsing (cohérent avec exec.js)
// -------------------------

const dataset = process.argv[2] || "square";

let argsStartIndex = 3;

const modelName = dataset;

const datasetObj = getDataset(dataset);

console.log(`Dataset    : ${datasetObj.name}`);
console.log(`Model name : ${modelName}`);
console.log(`Samples    : ${datasetObj.samples.length}`);
console.log(`Input size : ${datasetObj.inputSize}`);
console.log(`Output size: ${datasetObj.outputSize}`);
console.log(`Task       : ${datasetObj.task}`);

// -------------------------
// Hyperparams
// -------------------------

const args = process.argv.slice(argsStartIndex);


function getArg(name, defaultValue) {
  const index = args.indexOf(`--${name}`);
  if (index !== -1) {
    const value = args[index + 1];
    if (value === undefined) return defaultValue;

    // Support valeurs numériques et strings
    const numValue = Number(value);
    if (Number.isFinite(numValue)) return numValue;
    return value;
  }
  return defaultValue;
}

const EPOCHS = getArg("epochs", 10000);
const NEURONS = getArg("neurons", 10);
const LR = getArg("lr", 0.0025);
const NOISE = getArg("noise", 0.0);
const OPTIMIZER = getArg("optimizer", "adam"); // sgd, adam, rmsprop

console.log("Training config:", { EPOCHS, NEURONS, LR, NOISE, OPTIMIZER });

// -------------------------
// Preprocessing
// -------------------------

function computePreprocessing(samples) {
  let maxX = 0;
  let maxY = 0;

  for (const s of samples) {
    for (const v of s.x) {
      const a = Math.abs(v);
      if (a > maxX) maxX = a;
    }
    for (const v of s.y) {
      const a = Math.abs(v);
      if (a > maxY) maxY = a;
    }
  }

  return {
    xScale: maxX || 1,
    yScale: maxY || 1
  };
}

const preprocessing = computePreprocessing(datasetObj.samples);

// -------------------------
// Network
// -------------------------

const net = new Network({
  inputSize: datasetObj.inputSize,
  hiddenNeurons: NEURONS,
  learningRate: LR,
  preprocessing,
  task: datasetObj.task,
  optimizer: OPTIMIZER
});

// -------------------------
// Training
// -------------------------

console.time("Training");
for (let epoch = 1; epoch <= EPOCHS; epoch++) {
  let totalLoss = 0;

  for (const { x, y } of datasetObj.samples) {
    const noisyX = x.map(v => v + (Math.random() - 0.5) * NOISE);
    const loss = net.train(noisyX, y);
    totalLoss += loss;
  }

  if (epoch % 500 === 0) {
    console.log(
      `Epoch ${epoch}/${EPOCHS} - Loss: ${(totalLoss / datasetObj.samples.length).toFixed(6)}`
    );
  }
}

console.timeEnd("Training");
// -------------------------
// Save model
// -------------------------

const model = net.toJSON();

fs.mkdirSync("./models", { recursive: true });
fs.writeFileSync(`./models/${modelName}.json`, JSON.stringify(model, null, 2));

console.log(`Model saved to models/${modelName}.json`);
