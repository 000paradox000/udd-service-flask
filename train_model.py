import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import joblib


def main():
    # load dataset
    iris = load_iris()

    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # split data
    x = df.drop(columns=["target"])
    y = df.target

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # train
    model = LogisticRegression(max_iter=200)
    # model = KNeighborsClassifier(n_neighbors=3)
    # model = DecisionTreeClassifier(random_state=42)
    # model = SVC()

    model.fit(x_train, y_train)

    # evaluate
    train_accuracy = model.score(x_train, y_train)
    print(f"train accuracy: {train_accuracy}")

    y_pred = model.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_pred)
    print(f"train accuracy: {train_accuracy}")

    test_accuracy = model.score(x_test, y_test)
    print(f"test accuracy: {test_accuracy}")

    y_pred = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"test accuracy: {test_accuracy}")

    # save the model
    path = "model.joblib"
    joblib.dump(model, path)


if __name__ == "__main__":
    main()
