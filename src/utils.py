import matplotlib.pyplot as plt
import torch


def evaluate_regression(regressor, X, y, samples=100, std_multiplier=2):
    preds = [regressor(X) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()
    return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()


def plot_results(
    X_train,
    y_train,
    X_test,
    y_test,
    y_pred,
    std_pred,
    xmin=-1,
    xmax=2,
    ymin=-3,
    ymax=5,
    stdmin=0.0,
    stdmax=10.0,
):
    """
    Given a dataset and predictions on test set, this function draw 2 subplots:
    - left plot compares train set, ground-truth (test set) and predictions
    - right plot represents the predictive variance over input range
    Source: https://github.com/Mattcrmx/RDFIA_reports/blob/main/BayesianDL/tp1/TP1_RDFIA_Bayesian_Linear_Regression.ipynb

    Args:
      X_train: (array) train inputs, sized [N,]
      y_train: (array) train labels, sized [N, ]
      X_test: (array) test inputs, sized [N,]
      y_test: (array) test labels, sized [N, ]
      y_pred: (array) mean prediction, sized [N, ]
      std_pred: (array) std prediction, sized [N, ]
      xmin: (float) min value for x-axis on left and right plot
      xmax: (float) max value for x-axis on left and right plot
      ymin: (float) min value for y-axis on left plot
      ymax: (float) max value for y-axis on left plot
      stdmin: (float) min value for y-axis on right plot
      stdmax: (float) max value for y-axis on right plot

    Returns:
      None
    """
    X_train = X_train.view(-1).detach().numpy()
    X_test = X_test.view(-1).detach().numpy()
    y_train = y_train.view(-1).detach().numpy()
    y_test = y_test.view(-1).detach().numpy()
    y_pred = y_pred.view(-1).detach().numpy()
    std_pred = std_pred.view(-1).detach().numpy()

    fig = plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.xlim(xmin=xmin, xmax=xmax)
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.plot(X_test, y_test, color="green", linewidth=2, label="Ground Truth")
    plt.plot(X_train, y_train, "o", color="blue", label="Training points")
    plt.plot(X_test, y_pred, color="red", label="BLR Poly")
    plt.fill_between(
        X_test,
        y_pred - std_pred,
        y_pred + std_pred,
        color="indianred",
        label="1 std. int.",
    )
    plt.fill_between(
        X_test, y_pred - std_pred * 2, y_pred - std_pred, color="lightcoral"
    )
    plt.fill_between(
        X_test,
        y_pred + std_pred * 1,
        y_pred + std_pred * 2,
        color="lightcoral",
        label="2 std. int.",
    )
    plt.fill_between(
        X_test, y_pred - std_pred * 3, y_pred - std_pred * 2, color="mistyrose"
    )
    plt.fill_between(
        X_test,
        y_pred + std_pred * 2,
        y_pred + std_pred * 3,
        color="mistyrose",
        label="3 std. int.",
    )
    plt.legend()

    plt.subplot(122)
    plt.title("Predictive variance along x-axis")
    plt.xlim(xmin=xmin, xmax=xmax)
    plt.ylim(ymin=stdmin, ymax=stdmax)
    plt.plot(X_test, std_pred ** 2, color="red", label="\u03C3² {}".format("Pred"))

    # Get training domain
    training_domain = []
    current_min = sorted(X_train)[0]
    for i, elem in enumerate(sorted(X_train)):
        if elem - sorted(X_train)[i - 1] > 1:
            training_domain.append([current_min, sorted(X_train)[i - 1]])
            current_min = elem
    training_domain.append([current_min, sorted(X_train)[-1]])

    # Plot domain
    for j, (min_domain, max_domain) in enumerate(training_domain):
        plt.axvspan(
            min_domain,
            max_domain,
            alpha=0.5,
            color="gray",
            label="Training area" if j == 0 else "",
        )
    plt.axvline(X_train.mean(), linestyle="--", label="Training barycentre")

    plt.legend()
    return fig
