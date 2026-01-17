import square from './square.js';
import xor from './xor.js';
import primes from './primes.js';
import images from './images.js';

const datasets = {
    square,
    xor,
    primes,
    images
};

export default function getDataset(name) {
    if (!datasets[name]) {
        throw new Error(`Dataset inconnu: ${name}`);
    }
    return datasets[name];
}
