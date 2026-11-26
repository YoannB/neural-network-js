import fs from "fs";
import { parseTrainingOptions } from "./cli.js";
import { runExperiment } from "./core/experiment.js";

// node src/train.js <dataset> [options] : entraîne et sauvegarde models/<dataset>.json

const datasetName = process.argv[2] || "square";
const modelName = datasetName;
const options = parseTrainingOptions(process.argv.slice(3));

const { result, model } = await runExperiment(datasetName, options, {
  onStart: ({ dataset }) => {
    printHeader(dataset);
    console.time("Training");
  },
  onEpoch: ({ epoch, loss, last }) => {
    if (epoch === 1 || epoch % 500 === 0 || last) {
      console.log(`Epoch ${epoch}/${options.epochs} - Loss: ${loss.toFixed(6)}`);
    }
  }
});

function printHeader(ds) {
  console.log(`Dataset    : ${ds.name}`);
  console.log(`Model name : ${modelName}`);
  console.log(`Samples    : ${ds.samples.length + (ds.testSamples?.length ?? 0)}`);
  console.log(`Input size : ${ds.inputSize}`);
  console.log(`Output size: ${ds.outputSize}`);
  console.log(`Task       : ${ds.task}`);

  const { epochs, layers, activation, lr, noise, optimizer, seed, targetLoss, datasetOptions } = options;
  console.log("Training config:", {
    EPOCHS: epochs, LAYERS: layers, ACTIVATION: activation, LR: lr, NOISE: noise,
    OPTIMIZER: optimizer, SEED: seed, TARGET_LOSS: targetLoss,
    ...(Object.keys(datasetOptions).length > 0 ? { DATASET: datasetOptions } : {})
  });

  console.log(`Training samples: ${ds.samples.length}`);
  const testCount = ds.testSamples?.length ?? 0;
  if (testCount > 0) {
    console.log(`Test samples    : ${testCount} (non utilisés pour l'entraînement)`);
  }
}

if (result.reachedTarget) {
  console.log(`Perte cible ${options.targetLoss} atteinte à l'époque ${result.epochs}`);
}

console.timeEnd("Training");

fs.mkdirSync("./models", { recursive: true });
fs.writeFileSync(`./models/${modelName}.json`, JSON.stringify(model, null, 2));

console.log(`Model saved to models/${modelName}.json`);
