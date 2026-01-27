import xor from "./xor.js";
import square from "./square.js";
import primes from "./primes.js";
import images from "./images.js";
import linear from "./linear.js";
import rps from "./rps.js";

export default function getDataset(name) {
    switch (name) {
        case "linear": return linear;
        case "xor": return xor;
        case "square": return square;
        case "primes": return primes;
        case "images": return images;
        case "rps": return rps;
        default:
            throw new Error(`Dataset inconnu: ${name}`);
    }
}
