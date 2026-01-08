import fs from "fs";
import { Neuron } from "./neuron.js";

const model = JSON.parse(
  fs.readFileSync("./model/model.json", "utf-8")
);

const neuron = new Neuron(model.weight);

const input = Number(process.argv[2]);

if (isNaN(input)) {
  console.error("Veuillez fournir une valeur d'entrée.");
  process.exit(1);
}

const output = neuron.predict(input);

console.log(`Input: ${input}`);
console.log(`Output: ${output.toFixed(2)}`);
