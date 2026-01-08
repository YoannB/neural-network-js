import { Layer } from "./layer.js";

export class Network {
    constructor() {
        this.hidden = new Layer(1, 3); // 1 entrée → 3 neurones
        this.output = new Layer(3, 1); // 3 → 1 sortie
    }

    predict(x) {
        const h = this.hidden.forward([x]);
        return this.output.forward(h)[0];
    }

    train(x, yTrue) {
        const yPred = this.predict(x);
        const error = yPred - yTrue;

        this.output.backward([error]);
        this.hidden.backward(
            this.output.neurons[0].weights.map(
                (w) => error * w
            )
        );

        return error ** 2;
    }
}
