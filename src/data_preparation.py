# Standard library imports
import logging
import re
from typing import Any, Dict, Optional

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class DataPreparation:
    """
    A class used to clean and preprocess HDB resale prices data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        """
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the raw data.

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        logging.info("Starting data cleaning.")
        df.drop_duplicates(inplace=True)
        df['currency'] = df['price'].str.extract(r'^(\D+)')[0].str.strip()
        df['price'] = df['price'].apply(self._convert_to_sgd)
        df = df.dropna(subset=['no_show'])
        df['num_adults'] = df['num_adults'].replace({'one': '1', 'two': '2'}).astype(int)
        df['arrival_month'] = df['arrival_month'].str.title()
        df['lead_time_months'] = df.apply(self._months_before_arrival, axis=1).astype('Int64')
        df['days_stayed'] = df.apply(self._days_stayed, axis=1).astype('Int64')
        df.drop(columns=['booking_id', 'booking_month', 'arrival_day', 'checkout_month', 'checkout_day', 'currency'], inplace=True)
        logging.info("Data cleaning completed.")
        return df

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical, nominal, and ordinal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        numerical_transformer = Pipeline(steps=[
            ("impute", SimpleImputer(strategy='median')),
            ("scaler", StandardScaler())
        ])
        nominal_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        ordinal_transformer = Pipeline(
            steps=[
                (
                    "impute",
                    SimpleImputer(strategy="most_frequent"),
                ),   # <- comma added
                (
                    "ordinal",
                    OrdinalEncoder(
                        categories=[
                            self.config["room_categories"],
                            self.config["num_adults_categories"],
                            self.config["num_children_categories"],
                        ],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"]),
                ("ord", ordinal_transformer, self.config["ordinal_features"]),
                ("pass", "passthrough", self.config["passthrough_features"]),
            ],
            remainder="passthrough",
            n_jobs=-1,
        )
        return preprocessor

    @staticmethod
    def _months_before_arrival(row: pd.Series) -> Optional[int]:
        """
        Calculates the number of months between a customer's booking month and arrival month.

        The function takes a row containing 'booking_month' and 'arrival_month' as full month
        names, converts each to its month number (1-12), and returns the difference. Since the
        data has no year, the difference wraps around the calendar, so a booking made in
        November for a June arrival is 7 months ahead, not -5.

        Args:
            row (pd.Series): A row of the dataframe containing the columns 'booking_month'
                and 'arrival_month', each a full month name such as 'January'.

        Returns:
            Optional[int]: The number of months (0 to 11) between booking and arrival, or None
                if either month is missing or not a recognised month name.

        Example:
            months_before_arrival(row) -> 7
            (where row['booking_month'] = 'November' and row['arrival_month'] = 'June')
        """

        MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December']
        MONTH_NUM = {m: i for i, m in enumerate(MONTHS, start=1)}

        booked = MONTH_NUM.get(row['booking_month'])
        arrival = MONTH_NUM.get(row['arrival_month'])

        if booked is None or arrival is None:
            return pd.NA

        return (arrival - booked) % 12
    
    @staticmethod
    def _convert_to_sgd(price: str, usd_to_sgd=1.35) -> float:
        """
        Convert a price string like 'USD$ 658.23' or 'SGD$ 492.98' to a float in SGD.
        Null values are returned as NaN.

        Args:
            price (str): A string representing the price in USD or SGD, in the format 'CCC$ XXX.XX' where CCC is currency

        Returns:
            float: The price value in SGD without currency prefix, rounded to 2 decimal places.

        Example:
            convert_storey_range('USD$ 658.23') -> 888.62

        """
        if pd.isna(price):
            return float('nan')

        price = str(price).strip()
        amount = float(price.split()[-1].replace(',', ''))   # numeric part after the prefix

        if price.startswith('USD'):
            amount *= usd_to_sgd

        return round(amount, 2)

    @staticmethod
    def _days_stayed(row: pd.Series) -> Optional[int]:
        """
        Calculates the number of days a customer stays at the hotel.

        The function takes a row containing the arrival and checkout month names and day dates,
        converts each date into its day position within a year, and returns the difference.
        Negative day values are treated as sign errors and converted to positive. A leap year
        (366 days) is assumed so that 29 February is accepted, and the difference wraps around
        the calendar, so arriving on 30 December and checking out on 2 January is a stay of 3 days.

        Args:
            row (pd.Series): A row of the dataframe containing the columns 'arrival_month',
                'arrival_day', 'checkout_month' and 'checkout_day'. The months are month
                names such as 'January' (any capitalisation) and the days are numbers.

        Returns:
            Optional[int]: The number of days between arrival and checkout, or None if any
                value is missing, the month is not recognised, or the day is not a valid date
                for that month.

        Example:
            days_stayed(row) -> 2
            (where row['arrival_month'] = 'June', row['arrival_day'] = 25.0,
            row['checkout_month'] = 'June' and row['checkout_day'] = -27.0)
        """
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']
        days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        def day_of_year(month, day):
            if pd.isna(month) or pd.isna(day):
                return None
            month = str(month).strip().title()
            if month not in months:
                return None
            idx = months.index(month)
            day = int(abs(day))
            if not 1 <= day <= days_in_month[idx]:
                return None
            return sum(days_in_month[:idx]) + day

        arrival = day_of_year(row['arrival_month'], row['arrival_day'])
        checkout = day_of_year(row['checkout_month'], row['checkout_day'])

        if arrival is None or checkout is None:
            return None

        return (checkout - arrival) % 366