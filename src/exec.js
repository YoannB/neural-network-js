import fs from "fs";
import { Network } from "./network.js";

// Lecture du modèle entraîné
const rawModel = JSON.parse(
  fs.readFileSync("./model/model.json", "utf-8")
);

// Reconstruction du réseau
const net = new Network();

// On restaure les poids manuellement
net.hidden.neurons.forEach((neuron, i) => {
  neuron.weights = rawModel.hidden.neurons[i].weights;
  neuron.bias = rawModel.hidden.neurons[i].bias;
});

net.output.neurons.forEach((neuron, i) => {
  neuron.weights = rawModel.output.neurons[i].weights;
  neuron.bias = rawModel.output.neurons[i].bias;
});

// Lecture de l’entrée
const input = Number(process.argv[2]);

if (isNaN(input)) {
  console.error("Veuillez fournir une valeur numérique en entrée.");
  process.exit(1);
}

// Prédiction
const output = net.predict(input);

console.log(`Input : ${input}`);
console.log(`Output: ${output.toFixed(4)}`);
