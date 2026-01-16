import square from './square.js';
import xor from './xor.js';
import primes from './primes.js';

const datasets = {
    square,
    xor,
    primes
};

function getDataset(name) {
    if (!datasets[name]) {
        throw new Error(`Dataset inconnu: ${name}`);
    }
    return datasets[name];
}

export default getDataset;