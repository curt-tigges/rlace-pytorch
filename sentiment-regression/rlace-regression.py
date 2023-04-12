import numpy as np
import tqdm
import torch
from sklearn.linear_model import SGDClassifier
import time
from torch.optim import SGD, Adam
import random
import sklearn

EVAL_CLF_PARAMS = {
    "loss": "log_loss",
    "tol": 1e-4,
    "iters_no_change": 15,
    "alpha": 1e-4,
    "max_iter": 25000,
}
NUM_CLFS_IN_EVAL = 3  # change to 1 for large dataset / high dimensionality


def init_classifier():
    """Initializes an SGDClassifier, which is only used for evaluation."""

    return SGDClassifier(
        loss=EVAL_CLF_PARAMS["loss"],
        fit_intercept=True,
        max_iter=EVAL_CLF_PARAMS["max_iter"],
        tol=EVAL_CLF_PARAMS["tol"],
        n_iter_no_change=EVAL_CLF_PARAMS["iters_no_change"],
        n_jobs=32,
        alpha=EVAL_CLF_PARAMS["alpha"],
    )


def symmetric(X):
    """Make a matrix symmetric by averaging with its transpose."""
    X.data = 0.5 * (X.data + X.data.T)
    return X


def get_score(X_train, y_train, X_dev, y_dev, P, rank):
    """Get the score of a projection matrix P on the dev set."""
    P_svd = get_projection(P, rank)

    loss_vals = []
    accs = []

    for i in range(NUM_CLFS_IN_EVAL):
        clf = init_classifier()
        clf.fit(X_train @ P_svd, y_train)
        y_pred = clf.predict_proba(X_dev @ P_svd)
        loss = sklearn.metrics.log_loss(y_dev, y_pred)
        loss_vals.append(loss)
        accs.append(clf.score(X_dev @ P_svd, y_dev))

    i = np.argmin(loss_vals)
    return loss_vals[i], accs[i]


def solve_constraint(lambdas, d=1):
    """Solve the constraint sum(min(max(lambdas - theta, 0), 1)) = d for theta."""

    def f(theta):
        return_val = np.sum(np.minimum(np.maximum(lambdas - theta, 0), 1)) - d
        return return_val

    theta_min, theta_max = max(lambdas), min(lambdas) - 1
    assert f(theta_min) * f(theta_max) < 0

    mid = (theta_min + theta_max) / 2
    tol = 1e-4
    iters = 0

    while iters < 25:

        mid = (theta_min + theta_max) / 2

        if f(mid) * f(theta_min) > 0:

            theta_min = mid
        else:
            theta_max = mid
        iters += 1

    lambdas_plus = np.minimum(np.maximum(lambdas - mid, 0), 1)
    # if (theta_min-theta_max)**2 > tol:
    #    print("didn't converge", (theta_min-theta_max)**2)
    return lambdas_plus


def get_majority_acc(y):
    """Get the majority accuracy of a set of labels."""
    from collections import Counter

    c = Counter(y)
    fracts = [v / sum(c.values()) for v in c.values()]
    maj = max(fracts)
    return maj


def get_entropy(y):
    """Get the entropy of a set of labels."""

    from collections import Counter
    import scipy

    c = Counter(y)
    fracts = [v / sum(c.values()) for v in c.values()]
    return scipy.stats.entropy(fracts)


def get_projection(P, rank):
    """Get the projection matrix from the SVD of P.

    Args:
        P: The projection matrix
        rank: The rank of the projection matrix

    Returns:
        The projection matrix after SVD.
    """
    D, U = np.linalg.eigh(P)
    U = U.T
    W = U[-rank:]
    P_final = np.eye(P.shape[0]) - W.T @ W
    return P_final


def prepare_output(P, rank, score):
    """Prepare the output of the algorithm. Prints output before and after SVD.

    Args:
        P: The projection matrix
        rank: The rank of the projection matrix
        score: The score of the projection matrix

    Returns:
        A dictionary with the score, the projection matrix before SVD, and the projection matrix after SVD.
    """
    P_final = get_projection(P, rank)
    return {"score": score, "P_before_svd": np.eye(P.shape[0]) - P, "P": P_final}


def solve_adv_game(
    X_train,
    y_train,
    X_dev,
    y_dev,
    rank=1,
    device="cpu",
    out_iters=75000,
    in_iters_adv=1,
    in_iters_clf=1,
    epsilon=0.0015,
    batch_size=128,
    evalaute_every=1000,
    optimizer_class=SGD,
    optimizer_params_P={"lr": 0.005, "weight_decay": 1e-4},
    optimizer_params_predictor={"lr": 0.005, "weight_decay": 1e-4},
):
    """
    Run the adversarial game.

    Args:
        X_train: The training data
        y_train: The training labels
        X_dev: The dev data
        y_dev: The dev labels
        rank: The rank of the projection matrix
        device: The device to run on
        out_iters: The number of outer iterations
        in_iters_adv: The number of inner iterations for the adversary
        in_iters_clf: The number of inner iterations for the classifier
        epsilon: The epsilon for the adversary
        batch_size: The batch size
        evalaute_every: Evaluate every n iterations
        optimizer_class: The optimizer class
        optimizer_params_P: The optimizer parameters for the projection matrix
        optimizer_params_predictor: The optimizer parameters for the predictor

    Returns:
        The projection matrix
    """

    def get_loss_fn(X, y, predictor, P, bce_loss_fn, optimize_P=False):
        """Get the loss function.

        Args:
            X: The input
            y: The labels
            predictor: The predictor
            P: The projection matrix
            bce_loss_fn: The loss function
            optimize_P: Whether to optimize the projection matrix or not. If True, the loss is negated.

        Returns:
            The loss
        """
        I = torch.eye(X_train.shape[1]).to(device)
        bce = bce_loss_fn(predictor(X @ (I - P)).squeeze(), y)
        if optimize_P:
            bce = -bce
        return bce

    X_torch = X_train.clone().detach().to(device)
    y_torch = y_train.clone().detach().to(device)

    num_labels = len(set(y_train.tolist()))

    # first, create a predictor (which will be perturbed by the adversary)
    # then, set the loss function based on the number of labels
    # here is where we would add support for regression
    if num_labels == 2:
        predictor = torch.nn.Linear(X_train.shape[1], 1).to(device)
        bce_loss_fn = torch.nn.BCEWithLogitsLoss()
        y_torch = y_torch.float()
    else:
        predictor = torch.nn.Linear(X_train.shape[1], num_labels).to(device)
        bce_loss_fn = torch.nn.CrossEntropyLoss()
        y_torch = y_torch.long()

    # initialize the projection matrix
    P = 1e-1 * torch.randn(X_train.shape[1], X_train.shape[1]).to(device)
    P.requires_grad = True

    # initialize the optimizers
    optimizer_predictor = optimizer_class(
        predictor.parameters(), **optimizer_params_predictor
    )
    optimizer_P = optimizer_class([P], **optimizer_params_P)

    # get the majority accuracy and entropy of the training set
    maj = get_majority_acc(y_train)
    label_entropy = get_entropy(y_train)
    pbar = tqdm.tqdm(range(out_iters), total=out_iters, ascii=True)
    count_examples = 0
    best_P, best_score, best_loss = None, 1, -1

    # run the outer loop
    for i in pbar:

        for j in range(in_iters_adv):
            # optimize the projection matrix
            # this matches up with the second inner loop of the algorithm in the paper
            P = symmetric(P)
            optimizer_P.zero_grad()

            idx = np.arange(0, X_torch.shape[0])
            np.random.shuffle(idx)
            X_batch, y_batch = X_torch[idx[:batch_size]], y_torch[idx[:batch_size]]

            loss_P = get_loss_fn(
                X_batch, y_batch, predictor, symmetric(P), bce_loss_fn, optimize_P=True
            )
            loss_P.backward()
            optimizer_P.step()

            # project

            with torch.no_grad():
                # project the projection matrix on the Fantope using lemma B.1
                D, U = torch.linalg.eigh(symmetric(P).detach().cpu())
                D = D.detach().cpu().numpy()
                D_plus_diag = solve_constraint(D, d=rank)
                D = torch.tensor(np.diag(D_plus_diag).real).float().to(device)
                U = U.to(device)
                P.data = U @ D @ U.T

        for j in range(in_iters_clf):
            # optimize the predictor
            # this matches up with the first inner loop of the algorithm in the paper
            optimizer_predictor.zero_grad()
            idx = np.arange(0, X_torch.shape[0])
            np.random.shuffle(idx)
            X_batch, y_batch = X_torch[idx[:batch_size]], y_torch[idx[:batch_size]]

            loss_predictor = get_loss_fn(
                X_batch, y_batch, predictor, symmetric(P), bce_loss_fn, optimize_P=False
            )
            loss_predictor.backward()
            optimizer_predictor.step()
            count_examples += batch_size

        if i % evalaute_every == 0:
            # pbar.set_description("Evaluating current adversary...")
            loss_val, score = get_score(
                X_train, y_train, X_train, y_train, P.detach().cpu().numpy(), rank
            )
            if (
                loss_val > best_loss
            ):  # if np.abs(score - maj) < np.abs(best_score - maj):
                best_P, best_loss = symmetric(P).detach().cpu().numpy().copy(), loss_val
            if np.abs(score - maj) < np.abs(best_score - maj):
                best_score = score

            # update progress bar

            best_so_far = (
                best_score if np.abs(best_score - maj) < np.abs(score - maj) else score
            )

            pbar.set_description(
                "{:.0f}/{:.0f}. Acc post-projection: {:.3f}%; best so-far: {:.3f}%; Maj: {:.3f}%; Gap: {:.3f}%; best loss: {:.4f}; current loss: {:.4f}".format(
                    i,
                    out_iters,
                    score * 100,
                    best_so_far * 100,
                    maj * 100,
                    np.abs(best_so_far - maj) * 100,
                    best_loss,
                    loss_val,
                )
            )
            pbar.refresh()  # to show immediately the update
            time.sleep(0.01)

        if i > 1 and np.abs(best_score - maj) < epsilon:
            # if i > 1 and np.abs(best_loss - label_entropy) < epsilon:
            break
    output = prepare_output(best_P, rank, best_score)
    return output
