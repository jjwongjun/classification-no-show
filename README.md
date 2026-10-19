# No-Show Customers Classification

## Project Description
This project implements a machine learning pipeline to classify the No-Show of customers to help a hotel chain to formulate policies to reduce expenses incurred due to No-Shows.
The pipeline includes data cleaning, preprocessing, model training, and evaluation. Multiple classification models are trained and compared against a trivial baseline to find the best performing one based on various metrics.

## Prerequisites and Installation Instructions

### Requirements
- Anaconda or Miniconda
- Python 3.11
- Required packages listed in `requirements.txt`

### Installation
1. Clone the repository:

git clone https://github.com/jjwongjun/no-show-classification.git
cd no-show-classification

2. Create and activate a conda environment:
    ```
    conda create -n hdbenv python=3.11
    conda activate hdbenv
    ```

3. Install the required packages:

pip install -r requirements.txt

## Pipeline Execution
To run the complete pipeline, execute:

python main.py

A different configuration file can be given with `python main.py --config path/to/config.yaml` (default: `./src/config.yaml`).

This will:
1. Load and validate the configuration, then load the data
2. Validate the data schema and clean the data
3. Preprocess features
4. Split data into training, validation, and test sets (stratified, with class distributions logged)
5. Train baseline models (a dummy classifier plus five real models)
6. Perform hyperparameter tuning
7. Evaluate models and select the best one by validation ROC AUC
8. Choose the decision threshold on the validation set and evaluate the best model once on the test set
9. Save the best model, metrics, threshold, config snapshot and plots to `artifacts/`, and write a log to `logs/pipeline.log`

The two output locations are set by `artifacts_dir` and `log_file` in the config. Remove either key to switch it off.

### Checking the helper functions
The feature-engineering helpers (price conversion, lead time, days stayed) contain doctests that act as unit tests. Run them from the project root with:

python -m doctest -v src/data_preparation.py

## Logical Flow of the Pipeline

### 1. Configuration (config.yaml)
The pipeline starts with loading configuration
parameters from `src/config.yaml`, which includes:
- Data file path
- Target column name
- Feature categorization (numerical, nominal, ordinal)
- Random seed (`random_state`), used for every split and model
- Train/validation/test split ratios (`val_test_size` is the held-out share, `val_size` is the share of that which becomes the test set; 0.2 and 0.5 give 80/10/10)
- Imbalance strategy (`class_weight`), tuning metric (`scoring`, ROC AUC) and cross-validation folds (`cv`)
- Hyperparameter grid for model tuning
- Output locations for artifacts and logs, and optional business costs for threshold selection

### 2. Data Preparation (`DataPreparation` class)
The DataPreparation' class handles:
- Checking that all required columns exist and that the target only contains 0 and 1 (clear error message otherwise)
- Removing duplicates (on a copy, so the input DataFrame is not modified)
- Converts price to float in SGD without prefix 
- Removes unusable rows (missing target)
- Standardizing num_adults labelling and the capitalisation of `arrival_month` and `booking_month`
- Computes lead time of booking in months
- Computes days stayed plan (negative checkout days are treated as sign errors; 29 February is accepted)
- Creating a preprocessor for feature transformation

### 3. Model Training (`ModelTraining` class)
The ModelTraining' class manages:
- Splitting data into training, validation, and test
sets
- Training a dummy-classifier baseline and baseline models (Logistic Regression, K-Nearest Neighbours (KNN), Decision Trees, Random Forest, Histogram Gradient Boosting)
- Hyperparameter tuning for Logistic Regression, KNN, and Decision Trees models
- Evaluating models using multiple metrics (Accuracy, Precision, Recall, F1 Score, Specificity, ROC AUC, PR AUC, Brier score)
- Selecting the best model based on ROC-AUC score on the validation set
- Choosing the decision threshold on the validation set and evaluating the best model once on the test set
- Saving the selected model, metrics, threshold, config snapshot and plots

### 4. Main Execution (`main.py`)
The main script orchestrates the entire pipeline by:
- Loading configuration and data
- Initializing data preparation and model training
- Running the training and evaluation process
- Identifying and evaluating the best model on the test set

## Key Findings and Feature Handling

### Exploratory Data Analysis

Before diving into the model training process, an exploratory data analysis (EDA) was conducted to understand the data better and identify any potential issues. Here are some key findings from the EDA:

**EDA Findings and Explanations**: 
- `price` in different currencies (SGD and USD only) and with prefix as string
- `checkout_day` has negative values
- Mild inbalanced distribution of the two categories in `no_show`
- `branch` only 2 values, 'Changi' and 'Orchard', with 'Changi' being more common
- Most common country is China, followed by Indonesia
- Mostly first time customers
- Most common room is King
- Most common platform is website
- Mostly 1 adult
- `num_adults` has labelling error 'one': '1', 'two': '2'
- Mostly 0 and 1 child(ren)
- 4 peaks in `price`, due to 4 room types price difference
- Lower bookings in February, April, May, December
- Labelling capitalization error for `arrival_month`
- Lower arrivals in January, November, December
- `checkout_month` has similar patten to `arrival_month`, indicating short hotel stays and no clear patten with respect to booking_month
- Price increases with room type
- Median price increases with `room` type in general, except for Queen and King
- IQR is smaller for president suite, suggesting more consistent pricing
- IQR larger for queen and king - greater variability in prices
- Presence of outliers for President Suite
- General trend more bookings closer to arrival date
- Days plan to stay mostly 1 to 2 days
- Outliers days stayed above 30 days and price above 500 SGD.
- Outliers are all no_shows and prices of these rooms vary significantly
- No correlation between `days_stayed` and `price` (Pearson r is about 0.03)
- Strongest predictive signal from `country`, `branch`, `first_time`
- Moderate predictive signal from `lead_time_months`, `room`
- Weak to no predictive signal from `platform`, `num_adults`, `price`
- A 20% random sample (about 24k rows) is representative of the full data (no-show rate 36.5% vs 37.0%) and is used for quick exploration; the pipeline is trained on the full data
- `room` and `price` are almost never missing in the same row (1 row has both missing, versus about 4,500 expected if independent), so they are not independent. Missingness of either is **not** informative for the target (no-show rate within 0.4 percentage points, p > 0.35), so no missingness-indicator features are used
- The dates have no year, so standard datetime parsing fails (29 February is rejected and negative checkout days cannot be parsed); month-number arithmetic is used instead
- Interaction features (`country` x `branch`, first-time x long lead time, price x days stayed) were tested and **did not** add cross-validated signal, because the effects are roughly additive, so they are not in the pipeline
- Permutation importance (random forest, validation set) ranks `country` far ahead of everything else, then `arrival_month`, `branch`, `lead_time_months`, `days_stayed` and `first_time`; `platform`, `num_adults` and `num_children` are about zero. `arrival_month` and `days_stayed` matter more than the descriptive tables suggested

### Data Cleaning
- Lead time from booking is extracted and converted to total months because extracted feature is more useful than booking and arrival months alone. In addition, booking and arrival months are stored as an object (string) and conversion to integer is necessary for model 
- Days planned to stay is extracted and converted to total days because because extracted feature is more useful than arrival and checking out days and months  alone. In addition, arrival and checking out days and months are stored as an object (string) and conversion to integer is necessary for model 

### Feature Processing
- **Numerical Features**: `price`, `days_stayed`
- Standardized using `StandardScaler` because the numerical features are on different scales, which can adversely impact distance-based (KNN) and gradient-based (logistic regression) models. Tree-based models are not affected by scaling
- Null values of `price` are imputed with the median (`SimpleImputer`), and null values of `room` with the most frequent value. Missingness is not informative for the target, so no indicator features are added. `room` and `price` are almost never missing in the same row
- **Nominal Features**: `branch`, `country`, `first_time`, `platform`, `arrival_month` - Encoded using `OneHotEncoder` because these features have no intrinsic order
- **Ordinal Features**: `room`, `num_adults`, `num_children` - Encoded using `OrdinalEncoder` with predefined categories because each feature can be considered to have a natural ranking
- **Passthrough Features**: `lead_time_months` - Used as-is after converting to numerical values because already been ordinally encoded

## Model Choices and Evaluation

### Class Imbalance
The target is mildly imbalanced (about 37% no-show, 63% show). **No resampling is used** (no oversampling, undersampling or SMOTE) because the imbalance is mild and resampling risks overfitting or discarding data. Instead:
- Splits are stratified, so train, validation, test and every CV fold keep the same class ratio (the distributions are logged after splitting)
- `class_weight='balanced'` is applied to the models that support it (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting). KNN has no such option. It can be switched off with `class_weight: null` in the config
- A `DummyClassifier` baseline (ROC AUC 0.5) shows the minimum bar a real model must beat
- Models are compared on ROC AUC, PR AUC, precision, recall and F1 rather than accuracy, and the decision threshold is tuned (below)

### Models Implemented and Justifications
0. **Dummy Classifier**: Always predicts from the class prior; the benchmark that every real model must beat
1. **Logistic Regression**: Starting point for binary classification problems, providing a straightforward approach to predicting probabilities of class membership
2. **K-Nearest Neighbours (KNN)**: Non-parametric algorithm that classifies observations based on the majority class of their nearest neighbours in the feature space; can capture complex decision boundaries
3. **Decision Trees**: Non-parametric algorithm, make predictions by following a series of decisions based on the features; highly interpretable
4. **Random Forest**: Ensemble of decision trees; captures non-linear effects and interactions and is robust to outliers
5. **Histogram Gradient Boosting**: Boosted trees that are usually strongest on tabular data and handle missing values natively

### Threshold Selection
A probability threshold of 0.5 is rarely optimal for no-show prediction. The threshold is chosen **on the validation set only** (maximising F1 for the no-show class), then fixed before the one-time test evaluation. If `cost_false_negative` and `cost_false_positive` are set in the config (for example the cost of an empty room versus turning away a guest), the threshold that minimises total cost is chosen instead.

### Hyperparameter Tuning
- Grid search is performed for Logistic Regression, KNN, and Decision Trees models because tuning model parameters can achieve the best possible performance. Grid Search CV's process is exhaustive but it ensures the best combination
- Parameters tuned because the right combination of these settings ensures balance and should perform well on unseen data. These include:
Logistic Regression:
- C (inverse of regularization strength)
- penalty (norm used in the penalization, e.g., 'l2')
K-Nearest Neighbors (KNN):
- n_neighbors (number of neighbors to use)
- weights (uniform or distance)
Decision Tree:
- max_depth (maximum depth of the tree)
- min_samples_split (minimum number of samples required to split an internal node)
- min_samples_leaf (minimum number of samples required to be at a leaf node)
- 5-fold cross-validation is used during tuning because averaged across all 5 folds will give a more reliable estimate of model performance
- Tuning is scored with ROC AUC (set by `scoring` in the config) because accuracy is misleading on an imbalanced target. The best parameters and CV scores are logged and saved to `artifacts/metrics.json`

### Evaluation Metrics
- **Accuracy**: Proportion of correct predictions (both true positives and true negatives) among the total number of cases
- **Precision**: Ratio of correctly predicted positive observations to the total predicted positives
- **Recall**: Ratio of correctly predicted positive observations to all actual positive observations (real no-shows)
- **F1 Score**: Harmonic mean of Precision and Recall
- **Specificity**: Proportion of actual shows that are correctly predicted as shows
- **PR AUC**: Area under the precision-recall curve (average precision); informative when the positive class matters most
- **Brier Score**: Mean squared error of the predicted probabilities, used as a calibration check (lower is better)
- **Receiver Operating Characteristic (ROC) Curve**: Graphical representation that illustrates the performance of a binary classifier across different threshold values
- **Area Under Curve (AUC)**: Area under ROC curve

## Deployment Considerations
- Code is structured into modular well-organized components making code easier to maintain, more scalable, and ready for real world applications
- The configuration is validated at start-up (missing keys and out-of-range values fail early with a clear message) and the data schema is checked before cleaning
- Console and file logging with timestamps; file, config and data errors are reported with clear messages and a non-zero exit code
- The selected model (`best_model.joblib`), its test metrics, decision threshold and a snapshot of the configuration are saved to `artifacts/`, so a result can be reproduced and the model reused

### Known Limitations and Future Work
- There is no separate `tests/` folder; the helper functions are covered by doctests only. Adding `pytest` unit tests and one end-to-end smoke test is the next step
- There is no batch-prediction command yet. The saved model and threshold in `artifacts/` are what such a command would load
- Only Logistic Regression, KNN and Decision Tree are hyperparameter tuned; Random Forest and Gradient Boosting use default settings
- Calibration is only checked through the Brier score; probabilities are not recalibrated
- The data has no year, so lead time is limited to 0-11 months and stays are computed with a leap-year assumption