import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_scatterplot(csv_file):
    # Load the CSV data into a Pandas DataFrame
    data = pd.read_csv(csv_file)

    # Create a scatterplot using seaborn
    sns.scatterplot(x='Age', y='MedianIncome', data=data)

    # Set plot labels and title
    plt.xlabel('Age')
    plt.ylabel('Median Income')
    plt.title('Scatterplot of Median Income vs Age')

    # Show the plot
    plt.show()

# Replace 'your_census_data.csv' with the actual file path to your CSV data
create_scatterplot('your_census_data.csv')
