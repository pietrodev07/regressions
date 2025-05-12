import type { TrainData } from "#/types";

export class LinearRegression {
  private slope: number;
  private intercept: number;

  constructor() {
    this.slope = 0;
    this.intercept = 0;
  }

  public train(data: TrainData) {
    const len = data.length;
    const sumX = data.reduce((sum, point) => sum + point[0], 0);
    const sumY = data.reduce((sum, point) => sum + point[1], 0);
    const sumXY = data.reduce((sum, point) => sum + point[0] * point[1], 0);
    const sumX2 = data.reduce((sum, point) => sum + point[0] ** 2, 0);

    this.slope = (len * sumXY - sumX * sumY) / (len * sumX2 - sumX ** 2);
    this.intercept = (sumY - this.slope * sumX) / len;
  }

  public predict(x: number) {
    return this.slope * x + this.intercept;
  }

  public getEquation() {
    return {
      slope: this.slope,
      intercept: this.intercept,
    };
  }
}
