// Les jeux de données sont chargés à la demande : seul celui qui est
// demandé est importé (le jeu d'images n'est lu que s'il est utilisé).
const loaders = {
    linear: () => import("./linear.js"),
    xor: () => import("./xor.js"),
    square: () => import("./square.js"),
    primes: () => import("./primes.js"),
    images: () => import("./images.js"),
    rps: () => import("./rps.js")
};

export const datasetNames = Object.keys(loaders);

// Un module de jeu de données exporte soit un objet, soit une fonction
// (éventuellement asynchrone) qui reçoit les options et renvoie cet objet.
export default async function getDataset(name, options = {}) {
    const loader = loaders[name];
    if (!loader) {
        throw new Error(`Dataset inconnu: ${name} (disponibles : ${datasetNames.join(", ")})`);
    }

    const { default: dataset } = await loader();
    return typeof dataset === "function" ? await dataset(options) : dataset;
}
