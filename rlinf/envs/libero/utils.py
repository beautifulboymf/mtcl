# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
from typing import Union

import numpy as np


def get_libero_type() -> str:
    """
    Returns the type of LIBERO, which can be "standard", "pro", or "plus".
    """
    return os.environ.get("LIBERO_TYPE", "standard").lower()


libero_type = get_libero_type()

if libero_type == "pro":
    try:
        import liberopro.liberopro.benchmark as benchmark
        from liberopro.liberopro.benchmark import Benchmark
    except ImportError:
        print(
            "[Utils] Warning: LIBERO_TYPE=pro but 'liberopro' not found. Falling back to 'libero'."
        )
        import libero.libero.benchmark as benchmark
        from libero.libero.benchmark import Benchmark

elif libero_type == "plus":
    try:
        import liberoplus.liberoplus.benchmark as benchmark
        from liberoplus.liberoplus.benchmark import Benchmark
    except ImportError:
        print(
            "[Utils] Warning: LIBERO_TYPE=plus but 'liberoplus' not found. Falling back to 'libero'."
        )
        import libero.libero.benchmark as benchmark
        from libero.libero.benchmark import Benchmark

else:
    try:
        import libero.libero.benchmark as benchmark
        from libero.libero.benchmark import Benchmark
    except ImportError:
        try:
            import liberopro.liberopro.benchmark as benchmark
            from liberopro.liberopro.benchmark import Benchmark
        except ImportError:
            try:
                import liberoplus.liberoplus.benchmark as benchmark
                from liberoplus.liberoplus.benchmark import Benchmark
            except ImportError:
                raise ImportError(
                    "No valid LIBERO package (libero, liberopro, or liberoplus) found."
                )


def get_libero_image(obs: dict[str, np.ndarray]) -> np.ndarray:
    """
    Extracts image from observations and preprocesses it.

    Args:
        obs: Observation dictionary from LIBERO environment

    Returns:
        Preprocessed image as numpy array
    """
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(
    obs: dict[str, np.ndarray], resize_size: Union[int, tuple[int, int]] = 224
) -> np.ndarray:
    """
    Extracts wrist camera image from observations and preprocesses it.

    Args:
        obs: Observation dictionary from LIBERO environment
        resize_size: Target size for resizing

    Returns:
        Preprocessed wrist camera image as numpy array
    """
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_benchmark_overridden(benchmark_name) -> Benchmark:
    """
    Return the Benchmark class for a given name.
    For "libero_130": return a dynamically aggregated class from all suites.
    For others: delegate to the original LIBERO get_benchmark.

    Args:
        benchmark_name: Name of the benchmark to get

    Returns:
        Benchmark class
    """
    name = str(benchmark_name).lower()
    if name != "libero_130":
        return benchmark.get_benchmark(benchmark_name)

    libero_cls = benchmark.BENCHMARK_MAPPING.get("libero_130", None)
    if libero_cls is not None:
        return libero_cls

    # Build aggregated task map once, preserving order and de-duplicating by task name
    aggregated_task_map: dict[str, benchmark.Task] = {}
    suites = getattr(benchmark, "libero_suites", [])
    for suite_name in suites:
        suite_map = benchmark.task_maps.get(suite_name, {})
        for task_name, task in suite_map.items():
            if task_name not in aggregated_task_map:
                aggregated_task_map[task_name] = task

    class LIBERO_ALL(Benchmark):
        def __init__(self, task_order_index=0):
            super().__init__(task_order_index=task_order_index)
            self.name = "libero_130"
            self._make_benchmark()

        def _make_benchmark(self):
            tasks = list(aggregated_task_map.values())
            self.tasks = tasks
            self.n_tasks = len(self.tasks)

    # Register for discoverability/help
    benchmark.BENCHMARK_MAPPING["libero_130"] = LIBERO_ALL
    return LIBERO_ALL


def get_libero130_task_id_to_suite() -> dict:
    """Map each ``libero_130`` aggregated task_id -> its origin suite name.

    Mirrors the aggregation in ``get_benchmark_overridden("libero_130")`` EXACTLY (same
    suite order from ``benchmark.libero_suites``, same de-dup by task name), so the
    task_id index returned here equals the task_id ``LiberoEnv`` assigns when
    ``task_suite_name == "libero_130"``. Used to translate suite names into task ids and
    to weight suites for sequential continual learning (see rlinf.algorithms.embodied_seqcl).
    """
    id_to_suite: dict = {}
    seen: set = set()
    idx = 0
    for suite_name in getattr(benchmark, "libero_suites", []):
        suite_map = benchmark.task_maps.get(suite_name, {})
        for task_name in suite_map:
            if task_name not in seen:
                seen.add(task_name)
                id_to_suite[idx] = suite_name
                idx += 1
    return id_to_suite


def get_libero130_suite_to_task_ids() -> dict:
    """Inverse of :func:`get_libero130_task_id_to_suite`: suite name -> sorted task_ids."""
    out: dict = {}
    for tid, suite in get_libero130_task_id_to_suite().items():
        out.setdefault(suite, []).append(tid)
    for suite in out:
        out[suite].sort()
    return out


def expand_active_suites_to_task_ids(active_suites) -> list:
    """Translate suite names (e.g. ``["libero_object", "libero_spatial"]``) into a sorted,
    de-duplicated ``task_id_filter`` over the ``libero_130`` benchmark. Unknown suites raise."""
    suite_to_ids = get_libero130_suite_to_task_ids()
    ids: list = []
    for suite in active_suites:
        if suite not in suite_to_ids:
            raise ValueError(
                f"active suite '{suite}' not found in libero_130 "
                f"(known suites: {sorted(suite_to_ids)})"
            )
        ids.extend(suite_to_ids[suite])
    return sorted(set(ids))
