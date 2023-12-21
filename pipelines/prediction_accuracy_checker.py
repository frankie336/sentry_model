import pandas as pd
from ForexMastermind.ML.tools.tools import DataLoader
from ForexMastermind.ML.tools.tools import ForexMastermindConfig


class PredictionAccuracyChecker:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None

    def load_data(self):
        print(f"Loading data from: {self.csv_path}")
        self.df = DataLoader().load_data(data_path=self.csv_path)
        print("Loaded data (first 5 rows):")
        print(self.df.head())

    def remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        print("Data after removing duplicates (first 5 rows):")
        print(self.df.head())

    def preprocess_data(self):
        # Attempt to convert the date column to datetime
        # If there is mixed formatting, pandas will try to infer the correct format
        try:
            self.df['date'] = pd.to_datetime(self.df['date'])
        except ValueError:
            # If there is a ValueError, attempt to infer the date format for each row individually
            self.df['date'] = self.df['date'].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S', errors='coerce')
            # Fill in NaT values that could not be converted due to incorrect formatting
            self.df['date'].fillna(method='ffill', inplace=True)

        self.df.sort_values(by=['Pair', 'date'], inplace=True)
        print("Data after preprocessing (first 5 rows):")
        print(self.df.head())

    def calculate_accuracy(self):
        self.df['Next_Open'] = self.df.groupby('Pair')['open'].shift(-1)
        self.df['Prediction_Correct'] = ((self.df['Prediction'] == 1) & (self.df['Next_Open'] > self.df['close'])) | \
                                        ((self.df['Prediction'] == 0) & (self.df['Next_Open'] <= self.df['close']))
        self.df['Prediction_Correct'] = self.df['Prediction_Correct'].astype(int)
        print("Data after calculating accuracy (first 5 rows):")
        print(self.df[['Pair', 'date', 'Next_Open', 'Prediction', 'Prediction_Correct']].head())

    def drop_unnecessary_columns(self):
        self.df.drop(columns=['Next_Open'], inplace=True)

    def save_results(self, output_path, cleaned_output_path):
        self.df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        self.df.to_csv(cleaned_output_path, index=False)
        print(f"Cleaned data saved to {cleaned_output_path}")

    def calculate_performance_stats(self):
        performance_stats = {}
        for pair in self.df['Pair'].unique():
            df_pair = self.df[self.df['Pair'] == pair]
            total_predictions = len(df_pair)
            correct_predictions = df_pair['Prediction_Correct'].sum()
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

            performance_stats[pair] = {
                'Total Predictions': total_predictions,
                'Correct Predictions': correct_predictions,
                'Accuracy': accuracy
            }

        return performance_stats

    def run(self):
        self.load_data()
        self.remove_duplicates()  # Remove duplicate rows
        self.preprocess_data()
        self.calculate_accuracy()
        self.drop_unnecessary_columns()
        output_path = self.csv_path.replace('.csv', '_with_accuracy.csv')
        cleaned_output_path = self.csv_path.replace('.csv', '_cleaned.csv') # Path for cleaned data
        self.save_results(output_path, cleaned_output_path) # Save both original and cleaned data
        stats = self.calculate_performance_stats()
        for pair, stat in stats.items():
            print(f"Performance stats for {pair}:", stat)


if __name__ == '__main__':
    config = ForexMastermindConfig()
    prediction_log_path = config.get_prediction_log_path()
    print(f"Using prediction log path: {prediction_log_path}")
    checker = PredictionAccuracyChecker(csv_path=prediction_log_path)
    checker.run()