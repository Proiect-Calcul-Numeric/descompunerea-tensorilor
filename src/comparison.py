import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from matematica import (
    HOSVD,
    QR_Gram_Schmidt,
    QR_Householder,
    QR_iteration,
    SVD,
    SVD_redus,
    Tridiag_Householder,
    margine_teoretica,
    matricize,
    mode_n_product,
    norma_extra_diag,
    norma_frobenius,
    permutare,
    permutare_inv,
    reconstruct,
    toate_extra_diag_sub_prag,
    tucker_error,
)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "comparison_results"
MATRIX_SIZES = [3, 5, 7, 9]
TENSOR_SHAPES = [(3, 4, 5), (4, 5, 6)]
REPETITIONS = 3


def timed_call(function, *args, repetitions=REPETITIONS, **kwargs):
    best_time = float("inf")
    last_result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        last_result = function(*args, **kwargs)
        elapsed = time.perf_counter() - start
        best_time = min(best_time, elapsed)
    return last_result, best_time


def frobenius_relative_error(actual, expected):
    denominator = np.linalg.norm(expected.ravel())
    if denominator < 1e-14:
        return np.linalg.norm((actual - expected).ravel())
    return np.linalg.norm((actual - expected).ravel()) / denominator


def np_tridiagonalize(A):
    values, vectors = np.linalg.eigh(A)
    return vectors, np.diag(values)


def np_qr_iteration(A):
    values, vectors = np.linalg.eigh(A)
    return np.diag(values), vectors


def np_svd_full(A):
    U, sigmas, Vt = np.linalg.svd(A, full_matrices=True)
    S = np.zeros_like(A, dtype=float)
    np.fill_diagonal(S, sigmas)
    return U, S, Vt


def np_svd_reduced(A, r):
    U, sigmas, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :r], np.diag(sigmas[:r]), Vt[:r, :]


def np_margine_teoretica(tensor, ranguri):
    suma_patrate = 0.0
    for mode in range(tensor.ndim):
        unfolded = np.transpose(tensor, permutare(tensor.ndim, mode)).reshape(
            tensor.shape[mode], -1
        )
        sigmas = np.linalg.svd(unfolded, compute_uv=False)
        suma_patrate += np.sum(sigmas[ranguri[mode] :] ** 2)
    return np.sqrt(suma_patrate)


def add_record(records, group, name, size, error, manual_time, numpy_time):
    records.append(
        {
            "group": group,
            "name": name,
            "size": str(size),
            "error": float(error),
            "manual_time": float(manual_time),
            "numpy_time": float(numpy_time),
        }
    )


def compare_matrix_functions(rng):
    records = []

    for n in MATRIX_SIZES:
        A = rng.normal(size=(n, n))
        symmetric = (A + A.T) / 2.0

        manual, manual_time = timed_call(norma_frobenius, A)
        reference, numpy_time = timed_call(np.linalg.norm, A, "fro")
        add_record(
            records,
            "matrix",
            "norma_frobenius",
            n,
            abs(manual - reference),
            manual_time,
            numpy_time,
        )

        manual, manual_time = timed_call(norma_extra_diag, A)
        reference, numpy_time = timed_call(
            lambda X: np.linalg.norm(X - np.diag(np.diagonal(X)), "fro"), A
        )
        add_record(
            records,
            "matrix",
            "norma_extra_diag",
            n,
            abs(manual - reference),
            manual_time,
            numpy_time,
        )

        threshold = 0.75
        manual, manual_time = timed_call(toate_extra_diag_sub_prag, A, threshold)
        reference, numpy_time = timed_call(
            lambda X, p: np.all(np.abs(X - np.diag(np.diagonal(X))) <= p),
            A,
            threshold,
        )
        add_record(
            records,
            "matrix",
            "toate_extra_diag_sub_prag",
            n,
            0.0 if manual == reference else 1.0,
            manual_time,
            numpy_time,
        )

        for name, function in [
            ("QR_Gram_Schmidt", QR_Gram_Schmidt),
            ("QR_Householder", QR_Householder),
        ]:
            (Q, R), manual_time = timed_call(function, A)
            (Q_np, R_np), numpy_time = timed_call(np.linalg.qr, A)
            manual_error = frobenius_relative_error(Q @ R, A)
            numpy_error = frobenius_relative_error(Q_np @ R_np, A)
            add_record(
                records,
                "matrix",
                name,
                n,
                abs(manual_error - numpy_error),
                manual_time,
                numpy_time,
            )

        (Q_tri, T_tri), manual_time = timed_call(Tridiag_Householder, symmetric)
        (_, T_np), numpy_time = timed_call(np_tridiagonalize, symmetric)
        manual_values = np.sort(np.linalg.eigvalsh(T_tri))
        numpy_values = np.sort(np.diag(T_np))
        add_record(
            records,
            "matrix",
            "Tridiag_Householder",
            n,
            np.linalg.norm(manual_values - numpy_values),
            manual_time,
            numpy_time,
        )

        Q_start, _ = Tridiag_Householder(symmetric)
        (T_final, _), manual_time = timed_call(
            QR_iteration, symmetric, Q_start, max_iter=1500
        )
        (T_np, _), numpy_time = timed_call(np_qr_iteration, symmetric)
        add_record(
            records,
            "matrix",
            "QR_iteration",
            n,
            np.linalg.norm(np.sort(np.diag(T_final)) - np.sort(np.diag(T_np))),
            manual_time,
            numpy_time,
        )

        (U, S, Vt), manual_time = timed_call(SVD, A)
        (U_np, S_np, Vt_np), numpy_time = timed_call(np_svd_full, A)
        manual_reconstruction = frobenius_relative_error(U @ S @ Vt, A)
        numpy_reconstruction = frobenius_relative_error(U_np @ S_np @ Vt_np, A)
        sigma_error = np.linalg.norm(np.diag(S) - np.diag(S_np))
        add_record(
            records,
            "matrix",
            "SVD",
            n,
            manual_reconstruction + numpy_reconstruction + sigma_error,
            manual_time,
            numpy_time,
        )

        r = max(1, n // 2)
        (U_r, S_r, Vt_r), manual_time = timed_call(SVD_redus, A, r)
        (U_np_r, S_np_r, Vt_np_r), numpy_time = timed_call(np_svd_reduced, A, r)
        manual_sigmas = np.diag(S_r)
        numpy_sigmas = np.diag(S_np_r)
        add_record(
            records,
            "matrix",
            "SVD_redus",
            f"{n}, r={r}",
            np.linalg.norm(manual_sigmas - numpy_sigmas),
            manual_time,
            numpy_time,
        )

    return records


def compare_tensor_functions(rng):
    records = []

    for shape in TENSOR_SHAPES:
        tensor = rng.normal(size=shape)
        ranks = tuple(max(1, dim - 1) for dim in shape)

        for mode in range(len(shape)):
            manual, manual_time = timed_call(permutare, len(shape), mode)
            reference, numpy_time = timed_call(
                lambda N, m: [m] + [axis for axis in range(N) if axis != m],
                len(shape),
                mode,
            )
            add_record(
                records,
                "tensor",
                "permutare",
                f"{shape}, mode={mode}",
                0.0 if manual == reference else 1.0,
                manual_time,
                numpy_time,
            )

            manual, manual_time = timed_call(permutare_inv, len(shape), mode)
            reference, numpy_time = timed_call(
                lambda N, m: np.argsort(permutare(N, m)).tolist(), len(shape), mode
            )
            add_record(
                records,
                "tensor",
                "permutare_inv",
                f"{shape}, mode={mode}",
                0.0 if manual == reference else 1.0,
                manual_time,
                numpy_time,
            )

            manual, manual_time = timed_call(matricize, tensor, mode)
            reference, numpy_time = timed_call(
                lambda T, m: np.transpose(T, permutare(T.ndim, m)).reshape(
                    T.shape[m], -1
                ),
                tensor,
                mode,
            )
            add_record(
                records,
                "tensor",
                "matricize",
                f"{shape}, mode={mode}",
                frobenius_relative_error(manual, reference),
                manual_time,
                numpy_time,
            )

            matrix = rng.normal(size=(shape[mode] + 1, shape[mode]))
            manual, manual_time = timed_call(mode_n_product, tensor, matrix, mode)
            reference, numpy_time = timed_call(
                lambda T, M, m: np.moveaxis(
                    np.tensordot(M, T, axes=([1], [m])), 0, m
                ),
                tensor,
                matrix,
                mode,
            )
            add_record(
                records,
                "tensor",
                "mode_n_product",
                f"{shape}, mode={mode}",
                frobenius_relative_error(manual, reference),
                manual_time,
                numpy_time,
            )

        (G, Us), hosvd_time = timed_call(HOSVD, tensor, ranks)
        reconstruction, reconstruction_time = timed_call(reconstruct, G, Us)
        manual_error, tucker_error_time = timed_call(tucker_error, tensor, G, Us)
        reference_error, numpy_error_time = timed_call(
            lambda original, approx: np.linalg.norm((original - approx).ravel()),
            tensor,
            reconstruction,
        )
        add_record(
            records,
            "tensor",
            "HOSVD + reconstruct",
            f"{shape}, ranks={ranks}",
            frobenius_relative_error(reconstruction, tensor),
            hosvd_time + reconstruction_time,
            numpy_error_time,
        )
        add_record(
            records,
            "tensor",
            "tucker_error",
            f"{shape}, ranks={ranks}",
            abs(manual_error - reference_error),
            tucker_error_time,
            numpy_error_time,
        )

        manual, manual_time = timed_call(margine_teoretica, tensor, ranks)
        reference, numpy_time = timed_call(np_margine_teoretica, tensor, ranks)
        add_record(
            records,
            "tensor",
            "margine_teoretica",
            f"{shape}, ranks={ranks}",
            abs(manual - reference),
            manual_time,
            numpy_time,
        )

    return records


def aggregate_records(records):
    grouped = {}
    for record in records:
        key = record["name"]
        grouped.setdefault(key, {"errors": [], "manual": [], "numpy": []})
        grouped[key]["errors"].append(record["error"])
        grouped[key]["manual"].append(record["manual_time"])
        grouped[key]["numpy"].append(record["numpy_time"])

    summary = []
    for name, values in grouped.items():
        summary.append(
            {
                "name": name,
                "error": np.mean(values["errors"]),
                "manual_time": np.mean(values["manual"]),
                "numpy_time": np.mean(values["numpy"]),
            }
        )
    return sorted(summary, key=lambda row: row["name"])


def plot_accuracy(summary, title, output_path):
    labels = [row["name"] for row in summary]
    errors = [max(row["error"], 1e-16) for row in summary]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, errors, color="#2f6f9f")
    ax.set_yscale("log")
    ax.set_ylabel("Mean absolute/relative error (log scale)")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_runtime(summary, title, output_path):
    labels = [row["name"] for row in summary]
    x = np.arange(len(labels))
    width = 0.38
    manual_times = [max(row["manual_time"], 1e-9) for row in summary]
    numpy_times = [max(row["numpy_time"], 1e-9) for row in summary]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, manual_times, width, label="Manual", color="#b24c38")
    ax.bar(x + width / 2, numpy_times, width, label="NumPy", color="#3f7f4f")
    ax.set_yscale("log")
    ax.set_ylabel("Mean runtime in seconds (log scale)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_summary(title, summary):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'Function':<28} {'Mean error':>14} {'Manual time':>14} {'NumPy time':>14}")
    for row in summary:
        print(
            f"{row['name']:<28} "
            f"{row['error']:>14.6e} "
            f"{row['manual_time']:>14.6e} "
            f"{row['numpy_time']:>14.6e}"
        )


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    matrix_records = compare_matrix_functions(rng)
    tensor_records = compare_tensor_functions(rng)

    matrix_summary = aggregate_records(matrix_records)
    tensor_summary = aggregate_records(tensor_records)

    plot_accuracy(
        matrix_summary,
        "Matrix functions: NumPy vs manual accuracy",
        RESULTS_DIR / "matrix_accuracy.png",
    )
    plot_runtime(
        matrix_summary,
        "Matrix functions: NumPy vs manual runtime",
        RESULTS_DIR / "matrix_runtime.png",
    )
    plot_accuracy(
        tensor_summary,
        "Tensor functions: NumPy primitive comparison accuracy",
        RESULTS_DIR / "tensor_accuracy.png",
    )
    plot_runtime(
        tensor_summary,
        "Tensor functions: NumPy primitive comparison runtime",
        RESULTS_DIR / "tensor_runtime.png",
    )

    print_summary("Matrix comparisons", matrix_summary)
    print_summary("Tensor comparisons", tensor_summary)
    print(f"\nPlots saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
