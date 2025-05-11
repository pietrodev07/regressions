import { customerData, housePricesData } from "#/datasets";
import { LinearRegression, LogisticRegression } from "#/algorihtms";

console.log("\n============= HOUSING PRICES (Linear Regression) =============\n");

const housingModel = new LinearRegression();
housingModel.train(housePricesData);

const housingEquation = housingModel.getEquation();
const predictedPrice = housingModel.predict(1600);

console.table(housePricesData.map(([x, y]) => ({ SquareFeet: x, Price: y })));
console.log(`Equation: y = ${housingEquation.slope}x + ${housingEquation.intercept}`);
console.log(`Predicted price for 1600 sqft: $${predictedPrice}K`);

console.log("\n============= CUSTOMER PURCHASE (Logistic Regression) =============\n");

const purchaseModel = new LogisticRegression(0.1, 1000);
purchaseModel.train(customerData);

const purchaseEquation = purchaseModel.getEquation();
const predictedProbability = purchaseModel.predictProbability(27);
const predictedClass = purchaseModel.predict(27);

console.table(customerData.map(([x, y]) => ({ Age: x, Purchased: y })));
console.log(`Equation: y = ${purchaseEquation.slope}x + ${purchaseEquation.intercept}`);
console.log(`Predicted probability for age 27: ${predictedProbability}`);
console.log(`Predicted class for age 27 (0 = no, 1 = yes): ${predictedClass}`);
