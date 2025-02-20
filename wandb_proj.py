import argparse
import wandb
import random
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Dictionary of models
models = {
    "LogisticRegression": LogisticRegression(solver="liblinear", multi_class="ovr"),
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
    "KNeighborsClassifier": KNeighborsClassifier(),
}

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LogisticRegression", type=str)
    args = parser.parse_args()

    # Start a new wandb run to track this script
    wandb.init(
        project="iris-classification",
        config={
            "model": args.model,
            "solver": "liblinear" if args.model == "LogisticRegression" else None,
            "multi_class": "ovr" if args.model == "LogisticRegression" else None,
            "cv_splits": 10,
            "test_size": 0.20
        }
    )

    # Load dataset
    df = datasets.load_iris()
    X = df.data
    y = df.target

    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    # Instantiate and train the model
    model = models[args.model]
    model.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_probas = model.predict_proba(X_test)

    # K-Fold Cross Validation
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    accuracy = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy").mean()
    f1_macro = cross_val_score(model, X_train, y_train, cv=kfold, scoring="f1_macro").mean()
    neg_log_loss = cross_val_score(model, X_train, y_train, cv=kfold, scoring="neg_log_loss").mean()

    # Log metrics to wandb
    wandb.log({
        "accuracy": accuracy,
        "f1_score": f1_macro,
        "negative_log_likelihood": neg_log_loss
    })

    # Simulate training and log to wandb
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset

        # Log metrics to wandb
        wandb.log({"epoch": epoch, "acc": acc, "loss": loss})

    # Plot also the training points
    fig = sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=df.target_names[y],
        alpha=1.0,
        edgecolor="black",
    )
    plt.title("Scatter Plot of Training Data")
    wandb.log({"scatter_plot": wandb.Image(plt)})
    plt.close()

    # Log sklearn-specific visualizations
    labels = df.target_names
    wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
    wandb.sklearn.plot_learning_curve(model, X_train, y_train)
    wandb.sklearn.plot_roc(y_test, y_probas, labels)
    wandb.sklearn.plot_precision_recall(y_test, y_probas, labels)
    wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)

    # Finish wandb run
    wandb.finish()
