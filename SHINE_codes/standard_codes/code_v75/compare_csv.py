# compare value in v3.csv and v9.csv
import pandas as pd
import numpy as np
def compare_csv(file1, file2):
    # Read the CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    df1_value = df1['speech_prob'].values
    df2_value = df2['speech_prob'].values

    diff_value = df1_value == df2_value
    print("difference:", 1- np.mean(diff_value))


if __name__ == "__main__":
    file1 = 'v75.csv'
    file2 = 'v100.csv'
    compare_csv(file1, file2)
    print(1)