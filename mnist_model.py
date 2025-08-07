from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

# تحميل البيانات
digits = load_digits()
X = digits.data
y = digits.target

# المعالجة
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# حفظ النموذج والمحول
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
