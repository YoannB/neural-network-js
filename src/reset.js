import fs from "fs";
import { Network } from "./network.js";

// Nouveau réseau avec poids aléatoires
const net = new Network();

fs.writeFileSync(
  "./model/model.json",
  JSON.stringify(net, null, 2)
);

console.log("Modèle réinitialisé.");
