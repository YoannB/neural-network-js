import fs from "fs";
import { Network } from "./core/network.js";
import getDataset from "./datasets/index.js";

// -------------------------
// Lecture dataset
// -------------------------
const datasetName = process.argv[2] || "square";
const dataset = getDataset(datasetName);

console.log(`Dataset: ${dataset.name}`);
console.log(`Samples: ${dataset.samples.length}`);
console.log(`Input size: ${dataset.inputSize}`);
console.log(`Output size: ${dataset.outputSize}`);
console.log(`Task: ${dataset.task}`);

// -------------------------
// Lecture des paramètres CLI
// -------------------------
const args = process.argv.slice(3);

function getArg(name, defaultValue) {
  const index = args.indexOf(`--${name}`);
  if (index !== -1) {
    const value = Number(args[index + 1]);
    if (Number.isFinite(value)) return value;
  }
  return defaultValue;
}

const EPOCHS = getArg("epochs", 10000);
const NEURONS = getArg("neurons", 12);
const LR = getArg("lr", 0.003);
const NOISE = getArg("noise", 0.0);

console.log("Training config:");
console.log({ EPOCHS, NEURONS, LR, NOISE });

// -------------------------
// Préprocessing (depuis les samples)
// -------------------------
function computePreprocessing(samples) {
  const allX = samples.flatMap(s => s.x);
  const allY = samples.flatMap(s => s.y);

  return {
    xScale: Math.max(...allX.map(v => Math.abs(v))) || 1,
    yScale: Math.max(...allY.map(v => Math.abs(v))) || 1
  };
}

const preprocessing = computePreprocessing(dataset.samples);

// -------------------------
// Création du réseau
// -------------------------
const net = new Network({
  inputSize: dataset.inputSize,
  hiddenNeurons: NEURONS,
  learningRate: LR,
  preprocessing,
  task: dataset.task
});

// -------------------------
// Entraînement
// -------------------------
for (let epoch = 1; epoch <= EPOCHS; epoch++) {
  let totalLoss = 0;

  for (const { x, y } of dataset.samples) {
    const noisyX = x.map(v => v + (Math.random() - 0.5) * NOISE);
    const loss = net.train(noisyX, y);
    totalLoss += loss;
  }

  if (epoch % 500 === 0) {
    console.log(
      `Epoch ${epoch}/${EPOCHS} - Loss: ${(totalLoss / dataset.samples.length).toFixed(6)}`
    );
  }
}

// -------------------------
// Sauvegarde du modèle
// -------------------------
const model = {
  meta: {
    dataset: dataset.name,
    inputSize: dataset.inputSize,
    outputSize: dataset.outputSize,
    task: dataset.task
  },
  config: {
    inputSize: dataset.inputSize,
    hiddenNeurons: net.hidden.neurons.length,
    learningRate: net.hidden.neurons[0].learningRate,
    preprocessing
  },
  hidden: {
    neurons: net.hidden.neurons.map(n => ({
      weights: n.weights,
      bias: n.bias
    }))
  },
  output: {
    neurons: net.output.neurons.map(n => ({
      weights: n.weights,
      bias: n.bias
    }))
  }
};

fs.mkdirSync("./model", { recursive: true });
fs.writeFileSync("./model/model.json", JSON.stringify(model, null, 2));

console.log("Model saved to model/model.json");
