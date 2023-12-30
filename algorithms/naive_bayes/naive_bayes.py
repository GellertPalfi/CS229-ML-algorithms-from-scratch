import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score

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


train_y = train["Species"]
train_x = train.drop("Species")

test_y = test["Species"]
test_x = test.drop("Species")

# Calculate mean, variance and prior probability for each class
means = train.group_by("Species").mean()
var = train.group_by("Species").agg(pl.all().var())
prior = train.group_by("Species").count()
prior = prior.with_columns((pl.col("count") / train.height).alias("proportion")).drop(
    "count"
)

classes = np.unique(train["Species"].to_list())  # Storing all possible classes


class NaiveBayes:
    def __init__(self, means, var, prior, classes):
        self.means: pl.DataFrame = means
        self.var: pl.DataFrame = var
        self.prior: pl.DataFrame = prior
        self.classes: list[str] = classes

    def pdf(self, x, mean, var):
        """Return pdf of Normal(mean, var) evaluated at x"""
        return np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)

    def predict(self, instance):
        means_dict = self.means.to_dict(as_series=False)
        vars_dict = self.var.to_dict(as_series=False)
        posteriors = []

        # calculate posterior probability for each class
        for flower_type in classes:
            class_index = means_dict["Species"].index(flower_type)
            prior = self.prior.row(by_predicate=(pl.col("Species") == flower_type))[1]
            likelihood = 1
            for feature_name in instance.keys():
                feature_value = instance[feature_name][0]
                mean = means_dict[feature_name][class_index]
                var = vars_dict[feature_name][class_index]
                # Calculate Gaussian PDF
                likelihood *= self.pdf(feature_value, mean, var)

            # Multiply likelihood by the prior probability
            posteriors.append(np.log(likelihood) + np.log(prior))

        # Return the class with the highest posterior probability
        return self.classes[np.argmax(posteriors)]


bayes = NaiveBayes(means, var, prior, classes)

example_instance = test_x[0].to_dict(as_series=False)

predicteded_classes = [
    bayes.predict(test_x[i].to_dict(as_series=False)) for i in range(test_x.height)
]

print(accuracy_score(test_y, predicteded_classes))
