import fs from "fs";
import { Network } from "./core/network.js";
import getDataset from "./datasets/index.js";
import { setSeed } from "./core/random.js";
import { trainNetwork } from "./core/training.js";

// -------------------------
// CLI parsing (cohérent avec exec.js)
// -------------------------

const dataset = process.argv[2] || "square";

let argsStartIndex = 3;

const modelName = dataset;

// -------------------------
// Hyperparams
// -------------------------

const args = process.argv.slice(argsStartIndex);

function getArg(name, defaultValue) {
  const index = args.indexOf(`--${name}`);
  if (index !== -1) {
    const value = args[index + 1];
    if (value === undefined) return defaultValue;

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
const OPTIMIZER = getArg("optimizer", "adam");
const SEED = getArg("seed", 42);
const TARGET_LOSS = getArg("target-loss", 0.0001);

const datasetObj = await getDataset(dataset, { seed: SEED });

// -------------------------
console.log(`Dataset    : ${datasetObj.name}`);
console.log(`Model name : ${modelName}`);
console.log(`Samples    : ${datasetObj.samples.length + (datasetObj.testSamples?.length ?? 0)}`);
console.log(`Input size : ${datasetObj.inputSize}`);
console.log(`Output size: ${datasetObj.outputSize}`);
console.log(`Task       : ${datasetObj.task}`);

console.log("Training config:", { EPOCHS, NEURONS, LR, NOISE, OPTIMIZER, SEED, TARGET_LOSS });

setSeed(SEED);

// -------------------------
// Dataset split : seuls les exemples d'entraînement servent à apprendre
// -------------------------

const trainSamples = datasetObj.samples;
const testCount = datasetObj.testSamples?.length ?? 0;

console.log(`Training samples: ${trainSamples.length}`);
if (testCount > 0) {
  console.log(`Test samples    : ${testCount} (non utilisés pour l'entraînement)`);
}

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

const preprocessing = computePreprocessing(trainSamples);

// -------------------------
// Network
// -------------------------

const net = new Network({
  inputSize: datasetObj.inputSize,
  outputSize: datasetObj.outputSize,
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

const result = trainNetwork(net, trainSamples, {
  epochs: EPOCHS,
  noise: NOISE,
  targetLoss: TARGET_LOSS,
  onEpoch: ({ epoch, loss, last }) => {
    if (epoch === 1 || epoch % 500 === 0 || last) {
      console.log(`Epoch ${epoch}/${EPOCHS} - Loss: ${loss.toFixed(6)}`);
    }
  }
});

if (result.reachedTarget) {
  console.log(`Perte cible ${TARGET_LOSS} atteinte à l'époque ${result.epochs}`);
}

console.timeEnd("Training");

// -------------------------
// Save model
// -------------------------

const model = net.toJSON();
model.config.seed = SEED;

fs.mkdirSync("./models", { recursive: true });
fs.writeFileSync(`./models/${modelName}.json`, JSON.stringify(model, null, 2));

console.log(`Model saved to models/${modelName}.json`);
