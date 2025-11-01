import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

# --- 🔹 Check and fix inconsistent feature lengths ---
# Get expected length from the first sample
expected_len = len(data[0])
filtered_data = []
filtered_labels = []

for x, y in zip(data, labels):
    if len(x) == expected_len:
        filtered_data.append(x)
        filtered_labels.append(y)
    else:
        print(f"⚠️ Skipped sample with inconsistent length {len(x)} (expected {expected_len})")

data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

print(f"✅ Clean dataset: {len(data)} samples, each with {len(data[0])} features")

# --- 🔹 Split dataset ---
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# --- 🔹 Train model ---
model = RandomForestClassifier()
model.fit(x_train, y_train)

# --- 🔹 Evaluate ---
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f"✅ {score * 100:.2f}% of samples were classified correctly!")

# --- 🔹 Save model ---
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("💾 Model saved as model.p")
