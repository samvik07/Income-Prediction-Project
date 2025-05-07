from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    return model.predict(X_test)


def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
