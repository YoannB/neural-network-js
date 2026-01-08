import fs from "fs";
import { Network } from "./network.js";

const EPOCH = Number(process.argv[2]) || 1000;

const dataset = JSON.parse(
  fs.readFileSync("./data/dataset.json", "utf-8")
);

const net = new Network();

for (let epoch = 1; epoch <= EPOCH; epoch++) {
  let totalLoss = 0;

  for (let i = 0; i < dataset.inputs.length; i++) {
    const x = dataset.inputs[i];
    const y = dataset.outputs[i];

    const loss = net.train(x, y);
    totalLoss += loss;
  }

  if (epoch % Math.floor(EPOCH / 10) === 0) {
    console.log(
      `Epoch ${epoch}/${EPOCH} - Loss: ${totalLoss.toFixed(4)}`
    );
  }
}

// Sauvegarde du modèle
fs.writeFileSync(
  "./model/model.json",
  JSON.stringify(net, null, 2)
);

console.log("Entraînement terminé.");
