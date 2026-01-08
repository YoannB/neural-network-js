import fs from "fs";

fs.writeFileSync(
  "./model/model.json",
  JSON.stringify({ weight: Math.random() }, null, 2)
);

console.log("Modèle réinitialisé.");
