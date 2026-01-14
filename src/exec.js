import fs from "fs";
import { Network } from "./network.js";

// Chargement du modèle
const rawModel = JSON.parse(
  fs.readFileSync("./model/model.json", "utf-8")
);

const net = Network.load(rawModel);

const x = Number(process.argv[2]);

if (isNaN(x)) {
  console.error("Entrée invalide");
  process.exit(1);
}

const debug = net.debugPredict(x);

console.log(`x = ${debug.x}`);
console.log("Activations cachées :");

debug.hidden.forEach(h =>
  console.log(
    `  Neurone ${h.neuron} | z=${h.z} | ReLU=${h.activation}`
  )
);

console.log(`Sortie finale y ≈ ${debug.output}`);
