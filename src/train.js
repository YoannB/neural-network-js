import fs from "fs";
import { Neuron } from "./neuron.js";

const EPOCH = Number(process.argv[2]) || 1000;

const dataset = JSON.parse(
  fs.readFileSync("./data/dataset.json", "utf-8")
);

const neuron = new Neuron();

for (let epoch = 1; epoch <= EPOCH; epoch++) {
  let totalLoss = 0;

  for (let i = 0; i < dataset.inputs.length; i++) {
    totalLoss += neuron.train(
      dataset.inputs[i],
      dataset.outputs[i]
    );
  }

  if (epoch % Math.floor(EPOCH / 10) === 0) {
    console.log(
      `Epoch ${epoch}/${EPOCH} | Loss=${totalLoss.toFixed(4)} | w=${neuron.weight.toFixed(4)} | b=${neuron.bias.toFixed(4)}`
    );
  }
}

fs.writeFileSync(
  "./model/model.json",
  JSON.stringify(
    {
      weight: neuron.weight,
      bias: neuron.bias
    },
    null,
    2
  )
);

console.log("Entraînement terminé.");
