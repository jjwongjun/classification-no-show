# Standard library imports
import logging
from typing import Any, Dict, Optional

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class DataPreparation:
    """
    A class used to clean and preprocess hotel booking data for no-show prediction.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

    # Columns that must be present in the raw data before cleaning starts
    REQUIRED_COLUMNS = [
        "booking_id", "no_show", "branch", "booking_month", "arrival_month",
        "arrival_day", "checkout_month", "checkout_day", "country", "first_time",
        "room", "price", "platform", "num_adults", "num_children",
    ]

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
        self._validate_schema(df)
        df = df.drop_duplicates().copy()  # work on a copy so the caller's DataFrame is not mutated
        df['currency'] = df['price'].str.extract(r'^(\D+)')[0].str.strip()
        df['price'] = df['price'].apply(self._convert_to_sgd)
        df = df.dropna(subset=['no_show'])
        df['num_adults'] = df['num_adults'].replace({'one': '1', 'two': '2'}).astype(int)
        df['arrival_month'] = df['arrival_month'].str.strip().str.title()
        df['booking_month'] = df['booking_month'].str.strip().str.title()  # same casing issue as arrival_month
        df['lead_time_months'] = df.apply(self._months_before_arrival, axis=1).astype('Int64')
        df['days_stayed'] = df.apply(self._days_stayed, axis=1).astype('Int64')
        df = df.drop(columns=['booking_id', 'booking_month', 'arrival_day', 'checkout_month', 'checkout_day', 'currency'])
        logging.info(f"Data cleaning completed. Cleaned data has {df.shape[0]} rows and {df.shape[1]} columns.")
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """
        Checks that the raw data has the columns and target values the pipeline relies on.

        Args:
        -----
        df (pd.DataFrame): The raw input DataFrame.

        Raises:
        -------
        ValueError: If required columns are missing or the target contains values other than 0 and 1.
        """
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Input data is missing required columns: {missing}")
        invalid_target = set(df["no_show"].dropna().unique()) - {0, 1}
        if invalid_target:
            raise ValueError(f"Target 'no_show' must be 0 or 1, found: {sorted(invalid_target)}")

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
            sparse_threshold=0,  # always return a dense array so every model type can consume it
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
            Optional[int]: The number of months (0 to 11) between booking and arrival, or pd.NA
                if either month is missing or not a recognised month name.

        Example:
            >>> row = pd.Series({'booking_month': 'November', 'arrival_month': 'June'})
            >>> DataPreparation._months_before_arrival(row)
            7
            >>> DataPreparation._months_before_arrival(pd.Series({'booking_month': 'March', 'arrival_month': 'March'}))
            0
            >>> DataPreparation._months_before_arrival(pd.Series({'booking_month': None, 'arrival_month': 'March'}))
            <NA>
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
            >>> DataPreparation._convert_to_sgd('USD$ 100.00')
            135.0
            >>> DataPreparation._convert_to_sgd('SGD$ 492.98')
            492.98
            >>> pd.isna(DataPreparation._convert_to_sgd(None))
            True
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
            >>> row = pd.Series({'arrival_month': 'June', 'arrival_day': 25.0,
            ...                  'checkout_month': 'June', 'checkout_day': -27.0})
            >>> DataPreparation._days_stayed(row)
            2
            >>> row = pd.Series({'arrival_month': 'December', 'arrival_day': 30.0,
            ...                  'checkout_month': 'January', 'checkout_day': 2.0})
            >>> DataPreparation._days_stayed(row)
            3
            >>> row = pd.Series({'arrival_month': 'June', 'arrival_day': 31.0,
            ...                  'checkout_month': 'July', 'checkout_day': 2.0}) # 31 June does not exist
            >>> DataPreparation._days_stayed(row) is None
            True
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