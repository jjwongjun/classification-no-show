# No-Show Customers Classification

## Project Description
This project implements a machine learning pipeline to classify the No-Show of customers to help a hotel chain to formulate policies to reduce expenses incurred due to No-Shows.
The pipeline includes data cleaning, preprocessing, model training, and evaluation. Multiple regression models are trained and compared to find the best performing one based on various metrics.

## Prerequisites and Installation Instructions

### Requirements
- Anaconda or Miniconda
- Python 3.11
- Required packages listed in `requirements.txt`

### Installation
1. Clone the repository:

git clone https://github.com/jjwongjun/no-show-classification.git
cd classification-no-show

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

This will:
1. Load and clean the data
2. Preprocess features
3. Split data into training, validation, and test sets
4. Train baseline models
5. Perform hyperparameter tuning
6. Evaluate models and select the best one

## Logical Flow of the Pipeline

### 1. Configuration (config.yaml)
The pipeline starts with loading configuration
parameters from `src/config.yaml`, which includes:
- Data file path
- Target column name
- Feature categorization (numerical, nominal, ordinal)
- Train/validation/test split ratios
- Hyperparameter grid for model tuning

### 2. Data Preparation (`DataPreparation` class)
The DataPreparation' class handles:
- Removing duplicates
- Converts price to float in SGD without prefix 
- Removes unusable rows
- Standardizing num_adults and arrival_month labelling
- Computes lead time of booking in months
- Computes days stayed plan 
- Creating a preprocessor for feature transformation

### 3. Model Training (`ModelTraining` class)
The ModelTraining' class manages:
- Splitting data into training, validation, and test
sets
- Training baseline models (Logistic Regression, K-Nearest Neighbours (KNN), Decision Trees)
- Hyperparameter tuning for Logistic Regression, KNN, and Decision Trees models
- Evaluating models using multiple metrics (Accuracy, Precision, Recall, F1 Score, Receiver Operating Characteristic (ROC) Curve, Area Under Curve (AUC))
- Selecting the best model based on ROC-AUC score

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
- Positive trend between days_stayed and price, higher price for more days stayed
- Outliers days stayed above 30 days and price above 500 SGD.
- Outliers are all no_shows and prices of these rooms vary significantly
- No positive correlation berween `days_stayed` and `price`
- Strongest predictive signal from `country`, `branch`, `first_time`
- Moderate predictive signal from `lead_time_months`, `room`
- Weak to no predictive signal from `platform`, `num_adults`, `price`

### Data Cleaning
- Lead time from booking is extracted and converted to total months because extracted feature is more useful than booking and arrival months alone. In addition, booking and arrival months are stored as an object (string) and conversion to integer is necessary for model 
- Days planned to stay is extracted and converted to total days because because extracted feature is more useful than arrival and checking out days and months  alone. In addition, arrival and checking out days and months are stored as an object (string) and conversion to integer is necessary for model 

### Feature Processing
- **Numerical Features**: `floor_area_sqm`,
`remaining_lease_months`, `lease_commence_date`, `year`
- Standardized using `StandardScaler` because the numerical features have approximately normal distributions. In addition, the numerical features are on different scales, which can adversely impact the performance of machine learning algorithms if not addressed. Lastly, machine learning algorithms assume normally distributed data and are sensitive to feature scales, particularly those that rely on distance metrics
- Impute with `SimpleImputer` for null values of `room` and `price`. Furthermore, null values of both features are independent from each other
- **Nominal Features**: `branch`, `country`, `first_time`, `platform`, `arrival_month` - Encoded using `OneHotEncoder` because these features have no intrinsic order
- **Ordinal Features**: `room`, `num_adults`, `num_children` - Encoded using `OrdinalEncoder` with predefined categories because each feature can be considered to have a natural ranking
- **Passthrough Features**: `lead_time_months` - Used as-is after converting to numerical values because already been ordinally encoded

## Model Choices and Evaluation

### Models Implemented and Justifications
1. **Logistic Regression**: Starting point for binary classification problems, providing a straightforward approach to predicting probabilities of class membership
2. **K-Nearest Neighbours (KNN)**: Non-parametric algorithm that classifies observations based on the majority class of their nearest neighbours in the feature space; can capture complex decision boundaries
3. **Decision Trees**: Non-parametric algorithm, make predictions by following a series of decisions based on the features; highly interpretable

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
- 5-fold cross-validation is used during tuning because averaged across all 5 folds will give a more reliable estimate of model performance.

### Evaluation Metrics
- **Accuracy**: Proportion of correct predictions (both true positives and true negatives) among the total number of cases
- **Precision**: Ratio of correctly predicted positive observations to the total predicted positives
- **Recall**: Ratio of correctly predicted positive observations to all observations
- **F1 Score**: Harmonic mean of Precision and Recall
- **Receiver Operating Characteristic (ROC) Curve**: Graphical representation that illustrates the performance of a binary classifier across different threshold values
- **Area Under Curve (AUC)**: Area under ROC curve

## Deployment Considerations
- Code is structured into modular well-organized componenets making code easier to maintain, more scalable, and ready for real world applications