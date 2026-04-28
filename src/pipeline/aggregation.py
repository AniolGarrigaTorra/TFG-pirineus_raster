import numpy as np


def aggregate_stack(stack: np.ndarray, metric: str) -> np.ndarray:
    """
    Aggregate a monthly stack.

    Input shape:
      (n_months, height, width)

    NaN values are ignored.
    """
    metric = metric.lower().strip()

    if stack.ndim != 3:
        raise ValueError(
            f"Expected stack with shape (months, height, width), got {stack.shape}"
        )

    if metric == "mean":
        return np.nanmean(stack, axis=0)

    if metric == "sum":
        return np.nansum(stack, axis=0)

    if metric == "std":
        return np.nanstd(stack, axis=0)

    if metric == "min":
        return np.nanmin(stack, axis=0)

    if metric == "max":
        return np.nanmax(stack, axis=0)

    raise ValueError(
        f"Unsupported temporal aggregation metric: {metric}. "
        "Supported: mean, sum, std, min, max"
    )


def months_from_range(month_range: list[int]) -> list[int]:
    """
    Convert [start_month, end_month] into an inclusive month list.

    Example:
      [5, 9] -> [5, 6, 7, 8, 9]
    """
    if len(month_range) != 2:
        raise ValueError(f"Month range must have two values: {month_range}")

    start_month, end_month = int(month_range[0]), int(month_range[1])

    if not 1 <= start_month <= 12:
        raise ValueError(f"Invalid start month: {start_month}")

    if not 1 <= end_month <= 12:
        raise ValueError(f"Invalid end month: {end_month}")

    if start_month > end_month:
        raise ValueError(
            f"Month ranges crossing year boundary are not supported yet: {month_range}"
        )

    return list(range(start_month, end_month + 1))