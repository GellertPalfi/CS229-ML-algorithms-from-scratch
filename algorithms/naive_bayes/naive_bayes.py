import math

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    """Naive Bayes classifier for Gaussian features.

    Args:
        means: DataFrame containing the mean of each feature for each class.
        var: DataFrame containing the variance of each feature for each class.
        prior: DataFrame containing the prior probability of each class.
        classes: List of all possible classes.
    """

    def __init__(
        self,
        means: pl.DataFrame,
        var: pl.DataFrame,
        prior: pl.DataFrame,
        classes: list[str],
    ) -> None:
        self.means = means
        self.var = var
        self.prior = prior
        self.classes = classes
        self.epsilon = 1e-8  # Small constant to prevent log(0)

    def pdf(self, x, mean, sd):
        """Calculate the Gaussian pdf at point x,with given mean, and standard deviation."""
        return math.exp(-((x - mean) ** 2) / (2 * sd**2)) / (
            (2 * math.pi * sd**2) ** 0.5
        )

    def predict(self, instance: dict[str, list[float]]) -> str:
        """Return the class with the highest posterior probability"""
        means_dict = self.means.to_dict(as_series=False)
        vars_dict = self.var.to_dict(as_series=False)
        posteriors = []

        # calculate posterior probability for each class
        for flower_type in self.classes:
            class_index = means_dict["Species"].index(flower_type)
            prior = self.prior.row(by_predicate=(pl.col("Species") == flower_type))[1]
            likelihood = 1

            for feature_name in instance.keys():
                feature_value = instance[feature_name][0]
                mean = means_dict[feature_name][class_index]
                var = vars_dict[feature_name][class_index]
                # Calculate Gaussian PDF
                likelihood *= self.pdf(feature_value, mean, var) + self.epsilon

            # Multiply likelihood by the prior probability
            posteriors.append(np.log(likelihood) + np.log(prior))

        # Return the class with the highest posterior probability
        return self.classes[np.argmax(posteriors)]


if __name__ == "__main__":
    df = pl.read_csv("data/Iris.csv")
    df = df.drop("Id")

    # train test split is commonly done with sklearn
    # but can also be achieved with polars/pandas
    sample = df.sample(fraction=1, shuffle=True, seed=42)
    test_size = int(sample.height * 0.2)
    train, test = (
        sample.tail(-test_size),
        sample.head(test_size),
    )

    y_train = train["Species"]
    X_train = train.drop("Species")

    y_test = test["Species"]
    X_test = test.drop("Species")
    # Calculate mean, variance and prior probability for each class
    means = train.group_by("Species").mean()
    var = train.group_by("Species").agg(pl.all().var())
    prior = train.group_by("Species").count()
    prior = prior.with_columns(
        (pl.col("count") / train.height).alias("proportion")
    ).drop("count")
    classes = np.unique(train["Species"].to_list())  # Storing all possible classes

    naive_bayes = NaiveBayes(means, var, prior, classes)

    predicteded_classes = [
        naive_bayes.predict(X_test[i].to_dict(as_series=False))
        for i in range(X_test.height)
    ]

    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    # comparing to sklearn GaussianNB implementation
    print(accuracy_score(y_test, y_pred))  # 0.9666666666666667
    # ! my implementations seems to be numerically unstable for reasons unknown to me
    # ! and therefore performs somewhat randomly giving these 3 results:
    # ! [0.7 and 1.0]
    # TODO investigate why this is the case
    print(accuracy_score(y_test.to_list(), predicteded_classes))  # 0.9666666666666667
