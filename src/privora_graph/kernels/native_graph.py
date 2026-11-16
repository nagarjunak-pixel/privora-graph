"""SHA-256 cached C++20 compiler & ctypes bridge for CSR Personalized PageRank & Cosine Search."""

from __future__ import annotations

import ctypes
import hashlib
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

F64Array = npt.NDArray[np.float64]
F32Array = npt.NDArray[np.float32]
I32Array = npt.NDArray[np.int32]


class PrivoraKernelBridge:
    """Compiles `csrc/privora_kernels.cpp` (`-O3 -std=c++20`) with exact NumPy fallback."""

    def __init__(self, csrc_path: Path | None = None) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self._src = csrc_path or (repo_root / "csrc" / "privora_kernels.cpp")
        self._lib: ctypes.CDLL | None = None
        self._backend = "numpy-fallback"
        self._try_compile_and_load()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_native_cpp(self) -> bool:
        return self._lib is not None

    def _try_compile_and_load(self) -> None:
        if not self._src.exists():
            return
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            return

        cache_dir = self._src.parent.parent / ".privora_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        src_hash = hashlib.sha256(self._src.read_bytes()).hexdigest()[:12]
        ext = "dylib" if platform.system() == "Darwin" else "so"
        out_lib = cache_dir / f"libprivora_kernels_{src_hash}.{ext}"

        if not out_lib.exists():
            cmd = [
                compiler,
                "-O3",
                "-std=c++20",
                "-shared",
                "-fPIC",
                str(self._src),
                "-o",
                str(out_lib),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                return

        lib = ctypes.CDLL(str(out_lib))
        i32_ptr = ctypes.POINTER(ctypes.c_int32)
        f64_ptr = ctypes.POINTER(ctypes.c_double)
        f32_ptr = ctypes.POINTER(ctypes.c_float)

        lib.privora_csr_ppr_f64.argtypes = [
            ctypes.c_int32,
            i32_ptr,
            i32_ptr,
            f64_ptr,
            f64_ptr,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int32,
            f64_ptr,
        ]
        lib.privora_csr_ppr_f64.restype = ctypes.c_int32

        lib.privora_batch_cosine_f32.argtypes = [
            f32_ptr,
            f32_ptr,
            ctypes.c_int32,
            ctypes.c_int32,
            f32_ptr,
        ]
        lib.privora_batch_cosine_f32.restype = None

        self._lib = lib
        self._backend = f"c++20-native ({Path(compiler).name} -O3)"

    def csr_personalized_pagerank(
        self,
        num_nodes: int,
        row_ptr: I32Array,
        col_idx: I32Array,
        edge_weights: F64Array,
        seed_dist: F64Array,
        *,
        teleport_alpha: float = 0.20,
        tol: float = 1e-9,
        max_iters: int = 100,
    ) -> tuple[F64Array, int]:
        rp = np.ascontiguousarray(row_ptr, dtype=np.int32)
        ci = np.ascontiguousarray(col_idx, dtype=np.int32)
        ew = np.ascontiguousarray(edge_weights, dtype=np.float64)
        sd = np.ascontiguousarray(seed_dist, dtype=np.float64)
        sd = sd / max(float(np.sum(sd)), 1e-15)

        out = np.empty(num_nodes, dtype=np.float64)
        if self._lib is not None:
            iters = int(
                self._lib.privora_csr_ppr_f64(
                    num_nodes,
                    rp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    ci.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    ew.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    sd.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    ctypes.c_double(teleport_alpha),
                    ctypes.c_double(tol),
                    max_iters,
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                )
            )
            return out, iters

        # Exact NumPy fallback
        out[:] = sd
        row_sums = np.zeros(num_nodes, dtype=np.float64)
        for u in range(num_nodes):
            row_sums[u] = float(np.sum(ew[rp[u] : rp[u + 1]]))

        iters_used = 0
        for it in range(max_iters):
            iters_used = it + 1
            dangling = float(np.sum(out[row_sums <= 1e-15]))
            nxt = (teleport_alpha + (1.0 - teleport_alpha) * dangling) * sd
            walk = 1.0 - teleport_alpha
            for u in range(num_nodes):
                if row_sums[u] > 1e-15:
                    c = walk * (out[u] / row_sums[u])
                    for e in range(int(rp[u]), int(rp[u + 1])):
                        nxt[ci[e]] += c * ew[e]
            diff = float(np.sum(np.abs(nxt - out)))
            out[:] = nxt
            if diff <= tol:
                break
        return out, iters_used

    def batch_cosine_similarity(self, query: F32Array, corpus: F32Array) -> F32Array:
        q = np.ascontiguousarray(query, dtype=np.float32)
        c = np.ascontiguousarray(corpus, dtype=np.float32)
        n_vecs, dim = int(c.shape[0]), int(c.shape[1])
        if int(q.shape[0]) != dim:
            raise ValueError(f"Query dim {q.shape[0]} != corpus dim {dim}")

        scores = np.empty(n_vecs, dtype=np.float32)
        if self._lib is not None:
            self._lib.privora_batch_cosine_f32(
                q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                n_vecs,
                dim,
                scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            return scores

        q64 = q.astype(np.float64)
        c64 = c.astype(np.float64)
        q_n = max(float(np.linalg.norm(q64)), 1e-12)
        c_n = np.maximum(np.linalg.norm(c64, axis=1), 1e-12)
        return ((c64 @ q64) / (c_n * q_n)).astype(np.float32)


_GLOBAL_BRIDGE: PrivoraKernelBridge | None = None


def get_privora_bridge() -> PrivoraKernelBridge:
    global _GLOBAL_BRIDGE
    if _GLOBAL_BRIDGE is None:
        _GLOBAL_BRIDGE = PrivoraKernelBridge()
    return _GLOBAL_BRIDGE
