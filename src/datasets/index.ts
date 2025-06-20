import type { TrainData } from "#/types";

// dataset for the linear regression example
export const housePricesData: TrainData = [
  [900, 200],
  [1200, 250],
  [1500, 300],
  [1800, 350],
  [2000, 400],
  [2200, 450],
  [2500, 500],
];

// dataset for the logistic regression example
export const customerData: TrainData = [
  [18, 0],
  [22, 0],
  [25, 0],
  [28, 1],
  [30, 1],
  [35, 1],
  [40, 1],
];
