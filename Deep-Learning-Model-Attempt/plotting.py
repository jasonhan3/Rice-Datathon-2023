import matplotlib.pyplot as plt
import numpy as np
import data_prep
import pandas as pd

test_data = pd.read_csv("./data/2020flipped.csv")
_, actual_results = data_prep.prepare_test_data(test_data)

output = pd.read_csv("./data/output.csv")
our_results = output.sort_values(by='State')['V2'].to_numpy()

m, b = np.polyfit(our_results, actual_results, 1)

_, ax = plt.subplots()
ax.set_title("Model Predictions vs Perfect Fit")
ax.set_xlabel('Our predictions')
ax.set_ylabel('Actual Results')
plt.scatter(actual_results, our_results, alpha=0.7)
plt.plot(actual_results, m*actual_results+b, lw=2.5)
plt.show()

