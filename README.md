# Classical ML Projects

Two scikit-learn exercises: linear classifiers on the Iris dataset, and regression models for daily bike-rental demand.

`Python` `scikit-learn` `pandas` `NumPy` `Matplotlib` `Seaborn`
---

## What's here

**Iris classification.** Perceptron vs. Adaline (`SGDClassifier` with `squared_error` loss) on two binary splits of the Iris dataset: setosa vs. versicolor, which is linearly separable, and versicolor vs. virginica, which is not. Each split is run with 2, 3, and 4 features to see whether extra dimensions rescue a linear boundary.

**Bike rental demand.** Seoul Bike Data — 8,760 hourly rows, 14 columns. Four weather features (temperature, humidity, wind speed, visibility) predict `Rented Bike Count`. Linear Regression is compared against a Random Forest, then tuned with `GridSearchCV` and probed with polynomial features and residual plots.

## Results

Bike demand, on the held-out test split (from the notebook's stored outputs):

| Model | MSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 228,886.24 | 354.62 | 0.40 |
| Random Forest (100 trees) | 203,276.51 | 308.93 | 0.47 |

Validation MSE: 280,628.34 (linear) vs. 235,578.16 (random forest). Degree-2 polynomial features gave 266,221.98 — no better than the plain linear model. `GridSearchCV` settled on `max_depth=10, min_samples_split=10, n_estimators=200` at 206,976.54 CV MSE.

Random Forest feature importances: temperature 0.47, humidity 0.27, wind speed 0.13, visibility 0.13.

Iris accuracies, as recorded in the script's own closing notes — the exported `.py` keeps no cell outputs, so these are not reproducible from the file alone:

| Split | 2 features | 4 features |
|---|---|---|
| Setosa vs. versicolor (separable) | 100% both models | 100% both models |
| Versicolor vs. virginica | Perceptron 90%, Adaline 96% | Perceptron 96%, Adaline 90% |

## The point of the exercise

Both projects are about the limits of a linear model. Iris shows it directly: on separable classes the Perceptron converges to a perfect boundary and extra features change nothing, while on overlapping classes the two learning rules — misclassification-driven vs. squared-error-driven — trade places as features are added. The bike data makes the same point with residuals: Linear Regression systematically underestimates high rental counts, and the Random Forest's residuals sit evenly around zero. R² of 0.47 on four weather features is the ceiling for this feature set, not for the problem.

## Repository map

| Path | Purpose |
|---|---|
| `Iris Classification - Sharayu Rasal.py` | Perceptron / Adaline comparison, exported from a notebook |
| `Bike Demand Prediction - Sharayu Rasal.ipynb` | Regression notebook with stored outputs and plots |

## Limitations

- The Iris file is a notebook export and does not run end to end. Two `train_test_split(...)` calls are broken across lines (lines 278 and 327), so they bind the function instead of calling it, and `X_non_linear_three` is used without ever being defined.
- The bike notebook reads from a Kaggle path (`/kaggle/input/ml-assignment8/SeoulBikeData.csv`); the CSV is not in the repo.
- Only four weather columns are used. Hour of day, season, and holiday flags are in the dataset and untouched, which is a large part of why R² stops at 0.47.
