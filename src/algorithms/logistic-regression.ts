import type { TrainData } from "#/types";

export class LogisticRegression {
  private slope: number;
  private intercept: number;
  private learningRate: number;
  private epochs: number;

  constructor(learningRate = 0.1, epochs = 1000) {
    this.slope = 0;
    this.intercept = 0;
    this.learningRate = learningRate;
    this.epochs = epochs;
  }

  private sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-z));
  }

  public train(data: TrainData) {
    const len = data.length;

    for (let i = 0; i < this.epochs; i++) {
      let gradientSlope = 0;
      let gradientIntercept = 0;

      for (const [x, y] of data) {
        const z = this.slope * x + this.intercept;
        const prediction = this.sigmoid(z);
        const error = prediction - y;

        gradientSlope += error * x;
        gradientIntercept += error;
      }

      this.slope -= this.learningRate * (gradientSlope / len);
      this.intercept -= this.learningRate * (gradientIntercept / len);
    }
  }

  public predict(x: number) {
    const z = this.slope * x + this.intercept;
    const prob = this.sigmoid(z);
    return prob >= 0.5 ? 1 : 0;
  }

  public predictProbability(x: number) {
    const z = this.slope * x + this.intercept;
    return this.sigmoid(z);
  }

  public getEquation() {
    return {
      slope: this.slope,
      intercept: this.intercept,
    };
  }
}
