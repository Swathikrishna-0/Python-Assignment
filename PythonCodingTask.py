import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import math
from scipy.optimize import least_squares
import bokeh.plotting as bp


class RegressionModelAnalyzer:
    def __init__(self, training_data_functions, ideal_data_functions):
        # Initialize with training and ideal data
        self.training_data_functions = training_data_functions
        self.ideal_data_functions = ideal_data_functions
        self.functions_choosen = []  # Store chosen functions

    def perform_regression_analysis(self):
        squared_deviations_sum = []
        # Iterate over each column of training data (except x column)
        for i in range(len(self.training_data_functions.columns) - 1):
            # Extract x and y values for regression
            x_train = self.training_data_functions["x"].values
            y_train = self.training_data_functions.iloc[:, i + 1].values

            # Define model function (sinusoidal in this case)
            def model_function(parameters, x):
                return parameters[0] * np.sin(parameters[1] * x + parameters[2])

            # Define function for residual calculation
            def remaining(parameters, x, y):
                return y - model_function(parameters, x)

            initial_params = [1, 1, 1]  # Initial parameters for optimization
            # Perform least squares optimization to fit the model
            res = least_squares(remaining, initial_params, args=(x_train, y_train))
            self.functions_choosen.append(res.x)  # Store optimized parameters

            # Calculate sum of squared deviations for this ideal function
            squared_deviations_sum.append(
                np.sum((y_train - model_function(res.x, x_train)) ** 2)
            )

        # Choose the functions that minimize the sum of squared deviations
        self.functions_choosen_indices = np.argsort(squared_deviations_sum)[:4]

    def assign_test_data(self, test_data):
        mapped_data_table = []
        # Iterate over test data
        for index, row in test_data.iterrows():
            x_test = row["x"]
            min_deviation = float("inf")
            chosen_function = None
            max_deviation_factor = math.sqrt(2)
            # Iterate over chosen function indices
            for func_index in self.functions_choosen_indices:
                parameters = self.functions_choosen[func_index]
                # Define function for the chosen index
                func = lambda x: parameters[0] * np.sin(
                    parameters[1] * x + parameters[2]
                )
                deviation = np.abs(row["y"] - func(x_test))
                # Calculate maximum deviation from ideal data
                max_deviation = np.max(
                    np.abs(
                        func(self.ideal_data_functions["x"])
                        - self.training_data_functions.iloc[:, func_index + 1]
                    )
                )
                # Check if deviation satisfies conditions
                if (
                    deviation <= max_deviation * max_deviation_factor
                    and deviation < min_deviation
                ):
                    chosen_function = func_index
                    min_deviation = deviation
            if chosen_function is not None:
                # Append mapped data to table
                mapped_data_table.append(
                    (x_test, row["y"], min_deviation, chosen_function)
                )
            else:
                print(
                    f"No ideal function chosen for test data at index {index}: x={x_test}, y={row['y']}"
                )
        self.mapped_data_table = pd.DataFrame(
            mapped_data_table,
            columns=[
                "X (test func)",
                "Y (test func)",
                "Delta Y (test func)",
                "No. of ideal func",
            ],
        )

    def save_results(self, engine):
        # Save mapped data and chosen functions with least square error to database
        self.mapped_data_table.to_sql(
            "mapped_data_table", engine, if_exists="replace", index=False
        )
        chosen_functions_df = pd.DataFrame(
            {
                "No. of ideal func": self.functions_choosen_indices,
                "Least Square Error": [
                    np.sum(
                        (
                            self.training_data_functions.iloc[:, i + 1].values
                            - self._reconstruct_function(
                                parameters, self.training_data_functions["x"].values
                            )
                        )
                        ** 2
                    )
                    for i, parameters in enumerate(self.functions_choosen)
                ],
            }
        )
        chosen_functions_df.to_sql(
            "functions_choosen", engine, if_exists="replace", index=False
        )
        print("Mapped Data:")
        print(self.mapped_data_table)

        print("\nChosen Functions:")
        print(chosen_functions_df)

    

    def _reconstruct_function(self, parameters, x):
        # Reconstruct function using parameters
        return parameters[0] * np.sin(parameters[1] * x + parameters[2])


def loading_data_into_database(path_of_files, name_of_table, engine):
    # Load data from CSV to database table
    data = pd.read_csv(path_of_files)
    data.to_sql(name_of_table, engine, if_exists="replace", index=False)


def explore_data(training_data_functions):
    # Visualize training data using Bokeh
    p = bp.figure(title="Training Data Exploration", x_axis_label="X", y_axis_label="Y")
    p.line(
        training_data_functions["x"],
        training_data_functions["y1"],
        line_width=2,
        color="red",
        legend_label="y1",
    )
    p.line(
        training_data_functions["x"],
        training_data_functions["y2"],
        line_width=2,
        color="green",
        legend_label="y2",
    )
    p.line(
        training_data_functions["x"],
        training_data_functions["y3"],
        line_width=2,
        color="blue",
        legend_label="y3",
    )
    p.line(
        training_data_functions["x"],
        training_data_functions["y4"],
        line_width=2,
        color="orange",
        legend_label="y4",
    )
    bp.show(p)


def apply_polynomial_regression(training_data_functions):
    # Apply polynomial regression to training data
    x_train = training_data_functions["x"]
    model1_coeffs = np.polyfit(x_train, training_data_functions["y1"], 2)
    model2_coeffs = np.polyfit(x_train, training_data_functions["y2"], 2)
    model3_coeffs = np.polyfit(x_train, training_data_functions["y3"], 2)
    model4_coeffs = np.polyfit(x_train, training_data_functions["y4"], 2)
    return model1_coeffs, model2_coeffs, model3_coeffs, model4_coeffs


# Load the training data, ideal functions, and test data into pandas dataframes
training_data_functions = pd.read_csv("train.csv")
ideal_data_functions = pd.read_csv("ideal.csv")
test_data = pd.read_csv("test.csv")

# Create a SQLite database and load the training data, ideal functions, and test data into tables
engine = create_engine("sqlite:///PythonCodingTask.db")
loading_data_into_database("train.csv", "training_data_functions", engine)
loading_data_into_database("ideal.csv", "ideal_data_functions", engine)
loading_data_into_database("test.csv", "test_data", engine)

# Call explore_data function before performing regression analysis
explore_data(training_data_functions)

# Instantiate the RegressionModelAnalyzer class
regression_analysis = RegressionModelAnalyzer(
    training_data_functions, ideal_data_functions
)

# Perform the least-square regression analysis to choose the best-fit functions
regression_analysis.perform_regression_analysis()

# Apply polynomial regression if needed
model1_coeffs, model2_coeffs, model3_coeffs, model4_coeffs = (
    apply_polynomial_regression(training_data_functions)
)

# Assign the test data to the chosen ideal functions and save the results
regression_analysis.assign_test_data(test_data)
regression_analysis.save_results(engine)

# Visualize the data
regression_analysis.visualize_data()
