export class Neuron {
    constructor(weight = Math.random()) {
      this.weight = weight;
      this.learningRate = 0.01;
    }
  
    // y = w * x
    predict(x) {
      return this.weight * x;
    }
  
    // Erreur quadratique moyenne (simplifiée)
    loss(yPred, yTrue) {
      return Math.pow(yPred - yTrue, 2);
    }
  
    // Descente de gradient
    train(x, yTrue) {
      const yPred = this.predict(x);
      const error = yPred - yTrue;
  
      // d(loss)/d(weight) = 2 * error * x
      const gradient = 2 * error * x;
  
      this.weight -= this.learningRate * gradient;
  
      return this.loss(yPred, yTrue);
    }
}
  