import random
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sklearn
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from typing import Union


# =========================================================================================================
# Helper functions
# =========================================================================================================

def _fix_random_seed(seed: int):
    """Fix random seed across numpy and random modules."""
    np.random.seed(seed)
    random.seed(seed)
    sklearn.random.seed(seed)

    # TensorFlow and Keras
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        try:
            import keras
            keras.utils.set_random_seed(seed)
        except ImportError:
            pass
    print(f"Fixed random seed: {seed}")


def _fill_error(i, std, y_test, y_pred, error_list, metric):
    """Fill error list based on selected metric (MAE or MSE)."""
    ranked_confidence_list = np.argsort(std, axis=0).flatten()
    for k in range(len(y_test)):
        conf = ranked_confidence_list[: k + 1]
        if metric == "mae":
            error = mean_absolute_error(y_test[conf], y_pred[conf])
        elif metric == "mse":
            error = mean_squared_error(y_test[conf], y_pred[conf])
        else:
            errMsg = f"Invalid metric given: {metric}"
            raise ValueError(errMsg)
        error_list[i, k] = error


def _add_to_confidence_plot(n_test, y_values_list, label, color='blue', style='-', fill=True, lw=2):
    """Add Y values to the confidence plot with given style and label."""
    confidence_percentiles = np.linspace(1e-14, 100, n_test)
    err_mean = np.flip(np.mean(y_values_list, axis=0))
    plt.plot(confidence_percentiles, err_mean, style, label=label, color=color, linewidth=lw)
    if fill:
        err_std = np.flip(np.std(y_values_list, axis=0))
        lower, upper = err_mean - err_std, err_mean + err_std
        plt.fill_between(confidence_percentiles, lower, upper, alpha=0.1, color=color)


def compute_slope(y_min, y_max, x_min=0, x_max=100):
    """Compute slope of a line given two points."""
    print(f"y_min: {y_min}, y_max: {y_max}, x_min: {x_min}, x_max: {x_max}")
    if x_max - x_min == 0:
        return 0
    return (y_max - y_min) / (x_max - x_min)


def _compute_confidences(
        metric: str,
        output_dir: Path,
        plot_name: str,
        title: str,
        details: str,
        y_test_arr: np.ndarray,
        y_pred_arr: np.ndarray,
        std_total_arr: np.ndarray,
        std_al_arr: np.ndarray,
        std_ep_arr: np.ndarray,
        plot_std_total_only: bool,
        epistemic_model: bool,
):
    """Compute and plot confidence intervals based on error metric (MAE or MSE)."""
    n_trials, n_test = y_test_arr.shape
    print(f"n_trials: {n_trials}, n_test: {n_test}")
    error_confidence_arr = np.zeros((n_trials, n_test))
    oracle_confidence_arr = np.zeros((n_trials, n_test))

    plt.figure(figsize=(8,5))

    LABELSIZE = 20
    TICKLABELSIZE = 14  # for both axes
    TITLESIZE = 20

    if not plot_std_total_only and len(std_al_arr) > 0:
        for i in range(n_trials):
            _fill_error(i, std_ep_arr[i], y_test_arr[i], y_pred_arr[i], error_confidence_arr, metric)
        _add_to_confidence_plot(n_test, error_confidence_arr, 'epistemic', 'blue', fill=True)

        for i in range(n_trials):
            _fill_error(i, std_al_arr[i], y_test_arr[i], y_pred_arr[i], error_confidence_arr, metric)
        _add_to_confidence_plot(n_test, error_confidence_arr, 'aleatoric', 'orange', fill=True)

    for i in range(n_trials):
        _fill_error(i, std_total_arr[i], y_test_arr[i], y_pred_arr[i], error_confidence_arr, metric)
    if epistemic_model: 
        _add_to_confidence_plot(n_test, error_confidence_arr, 'epistemic', 'blue', fill=True)
    else: 
        _add_to_confidence_plot(n_test, error_confidence_arr, 'total', 'green', fill=True)
    
    # Compute slope for annotation
    y_total_max = np.max(error_confidence_arr)
    y_total_min = np.min(error_confidence_arr)
    slope = compute_slope(y_total_min, y_total_max)
    slop_percent = slope * 100
    slope_string = f"slope: {slop_percent:.4f}%"

    for i in range(n_trials):
        _fill_error(i, np.abs(y_test_arr[i] - y_pred_arr[i]), y_test_arr[i], y_pred_arr[i], oracle_confidence_arr, metric)
    _add_to_confidence_plot(n_test, oracle_confidence_arr, 'oracle', 'black', '--', fill=False)
    y_oracle_max = np.max(oracle_confidence_arr)

    # title 
    details = details + f" | {slope_string}"
    plt.suptitle(details+"\n", fontsize=TITLESIZE/2)
    plt.title(title, fontsize=TITLESIZE)  

    # labels 
    if metric == "mae":
        ylabel = "mean absolute error"
    elif metric == "mse":
        ylabel = "mean squared error"
    else:
        errMsg = f"Invalid metric given: {metric}"
        raise ValueError(errMsg)
    plt.ylabel(ylabel, fontsize=LABELSIZE)
    plt.xlabel(f'% discarded samples', fontsize=LABELSIZE)

    # Boarders 
    plt.xlim([0, 100])  
    plt.ylim(bottom=-0.1 * y_oracle_max, top=1.4 * y_oracle_max)
    plt.xticks(fontsize=TICKLABELSIZE)
    plt.yticks(fontsize=TICKLABELSIZE)


    # finishing touch
    plt.legend(loc='upper left', fontsize=TICKLABELSIZE)
    plt.grid(alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(output_dir / f'{plot_name}.png', dpi=600)
    plt.show()
    plt.close()


# =========================================================================================================
# Validation functions
# =========================================================================================================

def _validate_metric(metric):
    assert isinstance(metric, str), f"metric argument must be a string!"
    metric = metric.lower()
    assert metric in ("mae", "mse"), f"Invalid metric given: '{metric}'"
    return metric


def _validate_dir(directory):
    assert isinstance(directory, str) or isinstance(directory, Path), f"directory be a string or a pathlib.Path!"
    directory = Path(directory).absolute()
    assert directory.exists(), f"directory does not exist: '{directory}'"
    return directory


# =========================================================================================================
# Main function
# =========================================================================================================

def plot_confidence(
        regressor: BaseEstimator,
        df: pd.DataFrame,
        features: list,
        label: str,
        title: str,
        n_trials: int = 10,
        test_size: float = 0.5,
        #output_dir: Path | str = ".",
        output_dir: Union[Path, str] = ".",
        plot_name_prefix: str = "confidence_curve",
        seed: int = None,
        plot_std_total_only: bool = False,
        metric: str = "mae",
        epistemic_model: bool = False,
):
    """
    This script implements a function to evaluate a sklearn model's confidence through a confidence curve plot. 
    It will produce a plot for visual evaluation.
    The machine learning model is expected to predict 2 or 4 values.
    2 values: [y_pred, std_total]
    2 values: [y_pred, std_total, std_aleatoric, std_epistemic]

    :param regressor: an untrained sklearn model
    :param df: a dataframe holding features and labels data
    :param features: column headers for features
    :param label: column header for the label (one single label only)
    :param title: title for plot. 
    :param n_trials: number of repetitions
    :param test_size: proportion of the dataset to include in the test split.
    :param output_dir: directory to save plot to. Default = "."
    :param plot_name_prefix: prefix for plot filenames
    :param seed: random seed. 
    :param plot_std_total_only: if True, plot only the total standard deviation and not aleatoric & epistemic
    :param metric: error metric for prediction error. Default = "mae" (mean absolute error)
    :param epistemic_model: set True, if the model only outputs epistemic uncertainty (will adapt label and color)
    :return:
    """

    metric = _validate_metric(metric)
    output_dir = _validate_dir(output_dir)

    y_test_list, y_pred_list, std_list, std_al_list, std_ep_list = [], [], [], [], []

    for trial in range(n_trials):
        _fix_random_seed(seed + trial if seed is not None else None)
        regressor_test = clone(regressor)
        print(f"n_samples in df: {len(df.index)}")
        X_train, X_test, y_train, y_test = train_test_split(df[features], df[label], test_size=0.5, random_state=seed)
        regressor_test.fit(X_train, y_train.values)
        result = regressor_test.predict(X_test)

        if len(result) == 2:
            y_pred_test, std_total = result
            std_al, std_ep = np.array([]), np.array([])
        elif len(result) == 4:
            y_pred_test, std_total, std_al, std_ep = result
        else:
            raise ValueError(f"Unexpected prediction length {len(result)} (expected 2 or 4).")

        # Append values
        y_pred_list.append(y_pred_test.flatten())
        y_test_list.append(y_test.values.flatten())
        std_list.append(std_total.flatten())
        if len(std_al) > 0:
            std_al_list.append(std_al.flatten())
            std_ep_list.append(std_ep.flatten())

    # Convert to arrays
    y_test_arr = np.array(y_test_list)
    y_pred_arr = np.array(y_pred_list)
    std_total_arr = np.array(std_list)
    std_al_arr = np.array(std_al_list) if std_al_list else np.empty((0,))
    std_ep_arr = np.array(std_ep_list) if std_ep_list else np.empty((0,))

    # Plot the results
    plot_name = f"{plot_name_prefix}_{metric}_{'total' if plot_std_total_only else 'all'}"
    details = f"{metric.upper()}" + f" - {n_trials} trials on {len(df.index)} samples"

    _compute_confidences(
        metric=metric,
        output_dir=output_dir,
        plot_name=plot_name,
        title=title,
        details=details,
        y_test_arr=y_test_arr,
        y_pred_arr=y_pred_arr,
        std_total_arr=std_total_arr,
        std_al_arr=std_al_arr,
        std_ep_arr=std_ep_arr,
        plot_std_total_only=plot_std_total_only,
        epistemic_model=epistemic_model,
    )



# =========================================================================================================
# Multiple confidence plots
# =========================================================================================================

def plot_multiple_confidences(
        regressor: BaseEstimator,
        df: pd.DataFrame,
        features: list,
        label: str,
        title: str,
        n_trials: int = 10,
        test_size: float = 0.5,
        combinations: list[tuple[str, bool]] = None,
        output_dir: Union[Path, str] = ".",
        plot_name_prefix: str = "confidence_curve",
        seed: int = None,
        epistemic_model: bool = False,
):
    """
    Create up to 4 confidence plots from the same fitted trial data.

    :param regressor: an untrained sklearn model
    :param df: DataFrame holding features and labels
    :param features: feature column names
    :param label: label column name (single target only)
    :param n_trials: number of repetitions for fitting (default=5)
    :param test_size: proportion of the dataset to include in the test split (default=0.5)
    :param combinations: list of (metric, plot_std_total_only) tuples, e.g.:
                         [("mae", True), ("mae", False), ("mse", True), ("mse", False)]
    :param output_dir: directory to save plots
    :param plot_name_prefix: prefix for plot filenames
    :param seed: random seed for reproducibility
    """
    if combinations is None:
        combinations = [
            ("mae", True),
            ("mae", False),
            ("mse", True),
            ("mse", False)
        ]

    # Validate and prepare
    output_dir = _validate_dir(output_dir)
    combinations = [( _validate_metric(m), bool(tot_only) ) for m, tot_only in combinations]

    y_test_list, y_pred_list, std_list, std_al_list, std_ep_list = [], [], [], [], []

    # Fitting loop (done once for all plots)
    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}")
        _fix_random_seed(seed + trial if seed is not None else None)
        regressor_test = clone(regressor)
        print(f"n_samples in df: {len(df.index)}")
        X_train, X_test, y_train, y_test = train_test_split(df[features], df[label], test_size=test_size, random_state=seed+trial)
        regressor_test.fit(X_train, y_train.values)

        # Predict and collect results
        if isinstance(regressor_test, GaussianProcessRegressor):
            # Special handling for GaussianProcessRegressor
            print("Using GaussianProcessRegressor for prediction.")
            result = regressor_test.predict(X_test, return_std=True)
        else:
            # General case for other regressors
            print("Using general regressor for prediction.")
            result = regressor_test.predict(X_test)

        if len(result) == 2:
            y_pred_test, std_total = result
            std_al, std_ep = np.array([]), np.array([])
        elif len(result) == 4:
            y_pred_test, std_total, std_al, std_ep = result
        else:
            raise ValueError(f"Unexpected prediction length {len(result)} (expected 2 or 4).")

        y_pred_list.append(y_pred_test.flatten())
        y_test_list.append(y_test.values.flatten())
        std_list.append(std_total.flatten())
        if len(std_al) > 0:
            std_al_list.append(std_al.flatten())
            std_ep_list.append(std_ep.flatten())

    # Convert to arrays
    y_test_arr = np.array(y_test_list)
    y_pred_arr = np.array(y_pred_list)
    std_total_arr = np.array(std_list)
    std_al_arr = np.array(std_al_list) if std_al_list else np.empty((0,))
    std_ep_arr = np.array(std_ep_list) if std_ep_list else np.empty((0,))

    # Generate requested plots
    for metric, plot_std_total_only in combinations:
        plot_name = f"{plot_name_prefix}_{metric}_{'total' if plot_std_total_only else 'all'}"
        details = f"{metric.upper()}" + f" - {n_trials} trials on {len(df.index)} samples"
        _compute_confidences(
            metric=metric,
            output_dir=output_dir,
            plot_name=plot_name,
            title=title,
            details=details,
            y_test_arr=y_test_arr,
            y_pred_arr=y_pred_arr,
            std_total_arr=std_total_arr,
            std_al_arr=std_al_arr,
            std_ep_arr=std_ep_arr,
            plot_std_total_only=plot_std_total_only,
            epistemic_model=epistemic_model,
        )


# =========================================================================================================
# Multiple confidence plots with 5-fold strategy
# =========================================================================================================

def plot_multiple_confidences_5fold(
        regressor: BaseEstimator,
        df: pd.DataFrame,
        features: list,
        label: str,
        n_trials: int = 5,
        n_splits: int = 5,
        combinations: list[tuple[str, bool]] = None,
        #output_dir: Path | str = ".",
        output_dir: Union[Path, str] = ".",
        plot_name_prefix: str = "confidence_curve",
        title_prefix: str = "Confidence plot",
        seed: int = None,
):
    """
    Create up to 4 confidence plots from the same fitted trial data with a 5-fold strategy.

    :param regressor: an untrained sklearn model
    :param df: DataFrame holding features and labels
    :param features: feature column names
    :param label: label column name (single target only)
    :param n_trials: number of repetitions for fitting (default=5)
    :param n_splits: number of folds in K-Fold (default=5)
    :param combinations: list of (metric, plot_std_total_only) tuples, e.g.:
                         [("mae", True), ("mae", False), ("mse", True), ("mse", False)]
    :param output_dir: directory to save plots
    :param plot_name_prefix: prefix for plot filenames
    :param title_prefix: prefix for plot titles
    :param seed: random seed for reproducibility
    """
    if combinations is None:
        combinations = [
            ("mae", True),
            ("mae", False),
            ("mse", True),
            ("mse", False)
        ]

    # Validate and prepare
    output_dir = _validate_dir(output_dir)
    combinations = [( _validate_metric(m), bool(tot_only) ) for m, tot_only in combinations]

    # Prepare
    X = df[features].values
    y = df[label].values
    n_samples = len(df.index)

    # Preallocate arrays with shape (n_trials, n_samples)
    y_test_arr = np.zeros((n_trials, n_samples))
    y_pred_arr = np.zeros((n_trials, n_samples))
    std_total_arr = np.zeros((n_trials, n_samples))
    std_al_arr = np.zeros((n_trials, n_samples))
    std_ep_arr = np.zeros((n_trials, n_samples))

    # Fitting loop (done once for all plots)
    for trial in range(n_trials):
        _fix_random_seed(seed + trial if seed is not None else None)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + trial if seed else None)

        # Arrays for this trial
        y_test_full = np.zeros(n_samples)
        y_pred_full = np.zeros(n_samples)
        std_total_full = np.zeros(n_samples)
        std_al_full = np.zeros(n_samples)
        std_ep_full = np.zeros(n_samples)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            print(f"Trial {trial + 1}/{n_trials}, Fold {fold_idx + 1}/{n_splits}")

            # Split data for this fold
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone and fit the regressor for this fold
            regressor_fold = clone(regressor)
            regressor_fold.fit(X_train, y_train)

            # Predict and collect results
            if isinstance(regressor_fold, GaussianProcessRegressor):
                # Special handling for GaussianProcessRegressor
                print("Using GaussianProcessRegressor for prediction.")
                result = regressor_fold.predict(X_test, return_std=True)
            else:
                # General case for other regressors
                print("Using general regressor for prediction.")
                result = regressor_fold.predict(X_test)

            if len(result) == 2:
                y_pred_fold, std_total = result
                std_al, std_ep = np.array([]), np.array([])
            elif len(result) == 4:
                y_pred_fold, std_total, std_al, std_ep = result
            else:
                raise ValueError(f"Unexpected prediction length {len(result)} (expected 2 or 4).")

            # Fill predictions in correct positions
            y_test_full[test_idx] = y_test.flatten()
            y_pred_full[test_idx] = y_pred_fold.flatten()
            std_total_full[test_idx] = std_total.flatten()
            if len(std_al) > 0:
                std_al_full[test_idx] = std_al.flatten()
                std_ep_full[test_idx] = std_ep.flatten()

        # Save full-length arrays for this trial
        y_test_arr[trial] = y_test_full
        y_pred_arr[trial] = y_pred_full
        std_total_arr[trial] = std_total_full
        std_al_arr[trial] = std_al_full
        std_ep_arr[trial] = std_ep_full


    # Generate requested plots
    for metric, plot_std_total_only in combinations:

        plot_name = f"{plot_name_prefix}_{metric}_{'total' if plot_std_total_only else 'all'}"
        # title with prefix, metric and fold information
        title = f"{title_prefix} ({metric.upper()})" + f"\n{n_trials} trials with {n_splits} folds on {n_samples} samples"

        _compute_confidences(
            metric=metric,
            output_dir=output_dir,
            plot_name=plot_name,
            title=title,
            y_test_arr=y_test_arr,
            y_pred_arr=y_pred_arr,
            std_total_arr=std_total_arr,
            std_al_arr=std_al_arr,
            std_ep_arr=std_ep_arr,
            plot_std_total_only=plot_std_total_only,
        )
