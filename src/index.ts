import { housePricesData } from "#/datasets";
import { LinearRegression } from "#/algorihtms";

console.log("\n============= HOUSING PRICES (Linear Regression) =============\n");

const housingModel = new LinearRegression();
housingModel.train(housePricesData);

const housingEquation = housingModel.getEquation();
const predictedPrice = housingModel.predict(1600);

console.table(housePricesData.map(([x, y]) => ({ SquareFeet: x, Price: y })));
console.log(`Equation: y = ${housingEquation.slope}x + ${housingEquation.intercept}`);
console.log(`Predicted price for 1600 sqft: $${predictedPrice}K`);
