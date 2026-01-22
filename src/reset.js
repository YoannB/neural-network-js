import fs from "fs";

fs.rmSync("./models", { recursive: true, force: true });
fs.mkdirSync("./models");

console.log("Tous les modèles supprimés.");
