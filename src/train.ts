import fs from "fs";
import { parseTrainingOptions } from "./cli.js";
import { runExperiment } from "./core/experiment.js";
import type { Dataset, Sample } from "./core/types.js";

// node dist/src/train.js <dataset> [options] : entraîne et sauvegarde models/<dataset>.json

const datasetName = process.argv[2] || "square";
const modelName = datasetName;
const options = parseTrainingOptions(process.argv.slice(3));

const { result, model } = await runExperiment(datasetName, options, {
  onStart: ({ dataset, fitSamples, validationSamples }) => {
    printHeader(dataset, fitSamples, validationSamples);
    console.time("Training");
  },
  onEpoch: ({ epoch, loss, validationLoss, last }) => {
    if (epoch === 1 || epoch % 500 === 0 || last) {
      const val = validationLoss !== undefined ? ` - Validation: ${validationLoss.toFixed(6)}` : "";
      console.log(`Epoch ${epoch}/${options.epochs} - Loss: ${loss.toFixed(6)}${val}`);
    }
  }
});

function printHeader(ds: Dataset, fitSamples: Sample[], validationSamples: Sample[]): void {
  console.log(`Dataset    : ${ds.name}`);
  console.log(`Model name : ${modelName}`);
  console.log(`Samples    : ${ds.samples.length + (ds.testSamples?.length ?? 0)}`);
  console.log(`Input size : ${ds.inputSize}`);
  console.log(`Output size: ${ds.outputSize}`);
  console.log(`Task       : ${ds.task}`);

  const { epochs, layers, activation, lr, noise, optimizer, seed, targetLoss, l2, validation, patience, augment, datasetOptions } = options;
  console.log("Training config:", {
    EPOCHS: epochs, LAYERS: layers, ACTIVATION: activation, LR: lr, NOISE: noise,
    OPTIMIZER: optimizer, SEED: seed, TARGET_LOSS: targetLoss,
    ...((l2 ?? 0) > 0 ? { L2: l2 } : {}),
    ...((validation ?? 0) > 0 ? { VALIDATION: validation, PATIENCE: patience } : {}),
    ...(augment !== "none" ? { AUGMENT: augment } : {}),
    ...(Object.keys(datasetOptions).length > 0 ? { DATASET: datasetOptions } : {})
  });

  console.log(`Training samples: ${fitSamples.length}`);
  if (validationSamples.length > 0) {
    console.log(`Validation samples: ${validationSamples.length} (pour décider quand s'arrêter)`);
  }
  const testCount = ds.testSamples?.length ?? 0;
  if (testCount > 0) {
    console.log(`Test samples    : ${testCount} (non utilisés pour l'entraînement)`);
  }
}

if (result.validationLoss !== undefined) {
  console.log(`Meilleure perte de validation ${result.validationLoss.toFixed(6)} à l'époque ${result.bestEpoch} : poids de cette époque conservés`);
}

if (result.reachedTarget) {
  console.log(`Perte cible ${options.targetLoss} atteinte à l'époque ${result.epochs}`);
}

console.timeEnd("Training");

fs.mkdirSync("./models", { recursive: true });
fs.writeFileSync(`./models/${modelName}.json`, JSON.stringify(model, null, 2));

console.log(`Model saved to models/${modelName}.json`);
