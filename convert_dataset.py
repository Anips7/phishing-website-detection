from scipy.io import arff
import pandas as pd

data = arff.loadarff('phishing+websites/Training Dataset.arff')
df = pd.DataFrame(data[0])

# Convert byte strings to normal strings
df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Save as CSV
df.to_csv('data/phishing.csv', index=False)

print("Converted to CSV successfully!")