# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd

def load_simulation_data(filename):
    return pd.read_csv(filename)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = load_simulation_data('data/simulations/Streaming_Platform_Simulation_with_Predictive_Metrics.csv')
    print(df.head(10))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
