import fs from "fs";
import { Network } from "./network.js";

// -------------------------
// Lecture des paramètres CLI
// -------------------------

const args = process.argv.slice(2);

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
// Création du réseau
// -------------------------

const dataset = JSON.parse(
  fs.readFileSync("./data/dataset.json", "utf-8")
);

function computePreprocessing(X, Y) {
  return {
    xScale: Math.max(...X.map(v => Math.abs(v))),
    yScale: Math.max(...Y.map(v => Math.abs(v)))
  };
}

const preprocessing = computePreprocessing(dataset.inputs, dataset.outputs);

const net = new Network({
  hiddenNeurons: NEURONS,
  learningRate: LR,
  preprocessing
});

const data = [];
dataset.inputs.map((x, i) => {
  data.push({ x, y: dataset.outputs[i] });
})

// -------------------------
// Entraînement
// -------------------------

for (let epoch = 1; epoch <= EPOCHS; epoch++) {
  let totalLoss = 0;

  for (const { x, y } of data) {
    const noisyX = x + (Math.random() - 0.5) * NOISE;
    const loss = net.train(noisyX, y);
    totalLoss += loss;
  }

  if (epoch % 500 === 0) {
    console.log(
      `Epoch ${epoch}/${EPOCHS} - Loss: ${(totalLoss / data.length).toFixed(6)}`
    );
  }
}

// -------------------------
// Sauvegarde du modèle
// -------------------------

const model = {
  config: {
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
