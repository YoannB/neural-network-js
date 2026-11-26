import type { Dataset } from "../core/types.js";

/** Options transmises aux jeux de données : la graine, et celles propres à chaque jeu. */
export type DatasetOptions = { seed?: number | undefined } & Record<string, unknown>;

/**
 * Un module de jeu de données exporte soit un jeu fixe, soit une fonction
 * (éventuellement asynchrone) qui reçoit les options et renvoie le jeu.
 */
type DatasetModule = { default: Dataset | ((options: DatasetOptions) => Dataset | Promise<Dataset>) };

// Les jeux de données sont chargés à la demande : seul celui qui est
// demandé est importé (le jeu d'images n'est lu que s'il est utilisé).
const loaders: Record<string, () => Promise<DatasetModule>> = {
    linear: () => import("./linear.js"),
    xor: () => import("./xor.js"),
    square: () => import("./square.js"),
    primes: () => import("./primes.js"),
    images: () => import("./images.js"),
    rps: () => import("./rps.js")
};

export const datasetNames = Object.keys(loaders);

export default async function getDataset(name: string, options: DatasetOptions = {}): Promise<Dataset> {
    const loader = loaders[name];
    if (!loader) {
        throw new Error(`Dataset inconnu: ${name} (disponibles : ${datasetNames.join(", ")})`);
    }

    const { default: dataset } = await loader();
    return typeof dataset === "function" ? await dataset(options) : dataset;
}
