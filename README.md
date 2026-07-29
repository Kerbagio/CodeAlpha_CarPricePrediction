\# CodeAlpha - Car Price Prediction



\## Overview

This project builds a machine learning model to predict a used car's selling price based on features like its present (showroom) price, age, mileage, fuel type, transmission, and brand.



This is Task 3 of my Data Science Internship at CodeAlpha.



\## Dataset

Sourced from Kaggle's \[Vehicle dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho), originally 301 rows.



\### Data quality discovery

While exploring the data, I found that the file actually mixed \*\*100 motorcycle listings\*\* (Bajaj, Hero, Honda bikes, Royal Enfield, TVS, Yamaha, KTM, etc.) in with real cars — some inconsistently labeled (e.g. `"Honda Activa 125"` vs. just `"Activa 3g"`). Motorcycles sit on a completely different price scale (avg. \~1 lakh) than cars (avg. \~11 lakhs), so leaving them in would train a "car price" model on the wrong problem entirely. All motorcycle rows were identified and removed, leaving \*\*200 clean car records\*\*.



| | Count |

|---|---|

| Original rows | 301 |

| Motorcycles removed | 100 |

| \*\*Final car dataset\*\* | \*\*200\*\* |



\## Feature Engineering

\- \*\*Car\_Age\*\* — derived from `Year` (dataset collected \~2020), since a car's age is more directly meaningful than a raw year

\- \*\*Brand\*\* — extracted from the car's model name (e.g. `"swift"` → Maruti Suzuki, `"innova"` → Toyota, `"creta"` → Hyundai, `"city"` → Honda) and one-hot encoded, directly testing whether brand reputation affects resale price

\- \*\*Fuel\_Type, Selling\_type, Transmission\*\* — one-hot encoded from text into 0/1 columns

\- `Car\_Name` and raw `Year` dropped after extracting the features above



| Brand | Count |

|---|---|

| Maruti Suzuki | 50 |

| Toyota | 50 |

| Hyundai | 50 |

| Honda | 50 |



\## Tools \& Libraries

\- Python

\- pandas — data cleaning, feature engineering

\- matplotlib — data visualization

\- scikit-learn — model training and evaluation (Linear Regression, Random Forest)



\## Approach

1\. Loaded the data and identified/removed motorcycle listings mixed in with cars

2\. Extracted a `Brand` feature from car model names and created `Car\_Age` from `Year`

3\. Explored relationships between price and each feature visually

4\. One-hot encoded categorical features (Fuel\_Type, Selling\_type, Transmission, Brand)

5\. Split data into training (80%) and testing (20%) sets

6\. Trained and compared two models: Linear Regression and Random Forest Regressor

7\. Evaluated both using Mean Absolute Error (MAE) and R² score, then selected the stronger model



\## Data Exploration



Most cars are priced under 10 lakhs, with a long tail of pricier vehicles:



!\[Distribution of selling prices](price\_distribution.png)



Present price (showroom value) is the strongest single predictor of resale price:



!\[Selling price vs present price](price\_vs\_present\_price.png)



Newer cars generally sell for more, consistent with depreciation over time:



!\[Selling price vs manufacturing year](price\_vs\_year.png)



Kilometers driven shows a weaker, noisier relationship with price:



!\[Selling price vs kilometers driven](price\_vs\_kms.png)



Diesel cars and automatic transmissions command a higher average price:



!\[Average price by fuel type and transmission](price\_by\_category.png)



Toyota and Honda command the highest average resale prices among the four brands in this dataset:



!\[Average price by brand](price\_by\_brand.png)



\## Model Comparison



Two models were trained on identical data and compared honestly:



| Model | MAE (lakhs) | R² Score |

|---|---|---|

| Linear Regression | 1.45 | 0.852 |

| \*\*Random Forest\*\* | \*\*0.72\*\* | \*\*0.958\*\* |



!\[Model comparison: R2 score](model\_comparison.png)



Random Forest substantially outperformed Linear Regression, likely because car pricing involves nonlinear interactions (e.g. mileage mattering more for older cars) that a single straight-line formula can't capture. \*\*Random Forest was selected as the final model.\*\*



\## Results



Using Random Forest, predictions are consistently close to actual prices, most within half a lakh:



!\[Sample predictions vs actual](predictions\_table.png)



The scatter below plots every test car's actual price against its predicted price — points close to the red diagonal line represent accurate predictions:



!\[Predicted vs actual selling price](predicted\_vs\_actual.png)



\*\*Final performance: Mean Absolute Error of 0.72 lakhs (\~$750), R² of 0.958\*\* — the model explains 95.8% of the variation in used car prices.



\## How to Run

1\. Clone this repository

2\. Install dependencies: `pip install pandas scikit-learn matplotlib`

3\. Run `CodeAlpha\_CarPricePrediction.py`

