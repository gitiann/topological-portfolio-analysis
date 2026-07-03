"""Sub-windowing and Takens delay embedding.

``sub_windows`` is implemented (outer windowing bookkeeping). ``takens_embedding``
is the first piece of mathematics you write: you derived its shape already
(length-126 series, d=3, tau=1 -> 124 x 3).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray


def sub_windows(
    series: NDArray[np.float64],
    sub_len: int,
    shift: int,
) -> Iterator[NDArray[np.float64]]:
    """Yield overlapping sub-series of a single asset's return series.  [IMPLEMENTED]

    With a series of length ``T``, this yields ``(T - sub_len) // shift + 1``
    sub-series, each of length ``sub_len``, starting at 0, ``shift``, ``2*shift``, ...

    Paper values: ``sub_len = 126`` (T~), ``shift = 21`` (h). With a training
    window of T = 252 that gives 7 sub-windows. **Verify these two numbers
    against your copy of the paper before relying on them.**

    Args:
        series: 1-D return series for one asset.
        sub_len: Sub-window length in days.
        shift: Stride between successive sub-window starts.

    Yields:
        1-D arrays of length ``sub_len``.

    Raises:
        ValueError: If ``sub_len`` or ``shift`` is out of range.
    """
    n = series.shape[0]
    if not 0 < sub_len <= n:
        raise ValueError(f"sub_len must be in 1..{n}, got {sub_len}")
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")

    for start in range(0, n - sub_len + 1, shift):
        yield series[start : start + sub_len]


def takens_embedding(
    series: NDArray[np.float64],
    d: int = 3,
    tau: int = 1,
) -> NDArray[np.float64]:
    """Time-delay (Takens) embedding of a 1-D series into R^d.  [TODO]

    Map the scalar series ``s`` to a cloud of delay vectors. Row ``j`` is

        ( s[j], s[j + tau], s[j + 2*tau], ..., s[j + (d-1)*tau] ).

    The number of rows is ``len(series) - (d - 1) * tau`` and each row lives in
    R^d, so the output has shape ``(len(series) - (d-1)*tau, d)``.

    Paper values: ``d = 3``, ``tau = 1`` (fixed by convention, not tuned; the
    authors flag parameter selection as future work).

    Guard: raise ``ValueError`` if the series is too short to form even one
    delay vector, i.e. if ``len(series) - (d-1)*tau < 1``.

    Args:
        series: 1-D series (one asset's returns over one sub-window).
        d: Embedding dimension.
        tau: Time delay.

    Returns:
        Array of shape ``(len(series) - (d-1)*tau, d)``.

    Raises:
        ValueError: If the series is too short, or d/tau are non-positive.
    """
    raise NotImplementedError("implement takens_embedding")
