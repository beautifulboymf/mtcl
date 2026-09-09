# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import glob
import importlib
import os
import sys
from typing import Optional, Union

# EGL is DEFAULT-DENIED on this machine: two whole-host crashes (2026-08-27/29) via
# NVIDIA Bug 4905391 (host driver 535.179 < fixed 535.216.01; admin will not upgrade).
# 2026-09-04 the user re-approved EGL as an EXPLICIT PER-JOB OPT-IN, keyed by
# RLINF_ALLOW_EGL=1 (set only by dev/gpu_render_env.sh after its own gates), and only
# for fully-idle self-occupied cards -- the pattern that ran hundreds of hours clean.
# This fuse MUST run before ANY import below: `rlinf.envs.libero.venv` imports
# `libero.libero.envs` -> `robosuite`, whose binding_utils.py reads MUJOCO_GL at
# import time and FREEZES its GLContext class choice -- fixing the env var after that
# import is too late. Keep this block above every rlinf/libero import.
if (
    os.environ.get("RLINF_ALLOW_EGL") == "1"
    and os.environ.get("MUJOCO_GL", "").lower() == "egl"
):
    pass  # keyed opt-in: gpu_render_env.sh armed EGL deliberately for this job
else:
    if os.environ.get("MUJOCO_GL", "").lower() == "egl":
        sys.stderr.write(
            "[libero_env] MUJOCO_GL=egl OVERRIDDEN to osmesa: no RLINF_ALLOW_EGL key "
            "(EGL is opt-in only on this machine; host crashes 2026-08-27/29).\n"
        )
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import gym
import numpy as np
import torch
from omegaconf.omegaconf import OmegaConf

from rlinf.envs.libero.utils import (
    expand_active_suites_to_task_ids,
    get_benchmark_overridden,
    get_libero130_task_id_to_suite,
    get_libero_image,
    get_libero_type,
    get_libero_wrist_image,
    quat2axisangle,
)
from rlinf.envs.libero.venv import ReconfigureSubprocEnv
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor

libero_type = get_libero_type()

if libero_type in ["pro", "plus"]:
    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]
    LIBERO_PKG_NAME = f"libero{libero_type}"
    LIBERO_MAIN_MODULE_PATH = f"{LIBERO_PKG_NAME}.{LIBERO_PKG_NAME}"
    try:
        real_libero_pkg = importlib.import_module(LIBERO_PKG_NAME)
        real_libero_core = importlib.import_module(LIBERO_MAIN_MODULE_PATH)

        try:
            real_libero_benchmark = importlib.import_module(
                f"{LIBERO_MAIN_MODULE_PATH}.benchmark"
            )
        except ImportError:
            real_libero_benchmark = importlib.import_module(
                f"{LIBERO_PKG_NAME}.benchmark"
            )

        try:
            real_libero_envs = importlib.import_module(
                f"{LIBERO_MAIN_MODULE_PATH}.envs"
            )
        except ImportError:
            real_libero_envs = importlib.import_module(f"{LIBERO_PKG_NAME}.envs")

        sys.modules["libero"] = real_libero_pkg
        sys.modules["libero.libero"] = real_libero_core
        sys.modules["libero.libero.benchmark"] = real_libero_benchmark
        sys.modules["libero.libero.envs"] = real_libero_envs
    except ImportError as e:
        print(
            f"[Main Process Routing Error] Failed to import '{LIBERO_MAIN_MODULE_PATH}'. Error: {e}"
        )

if libero_type == "pro":
    from liberopro.liberopro.benchmark import Benchmark
elif libero_type == "plus":
    from liberoplus.liberoplus.benchmark import Benchmark
else:
    from libero.libero.benchmark import Benchmark


class LiberoEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.specific_reset_id = cfg.get("specific_reset_id", None)
        self.task_id_filter = cfg.get("task_id_filter", None)
        if self.task_id_filter is not None:
            self.task_id_filter = list(self.task_id_filter)

        # Sequential continual learning (rlinf.algorithms.embodied_seqcl): let a run name
        # its active suites (e.g. ["libero_object", "libero_spatial"]) instead of raw task
        # ids. Only valid for the aggregated libero_130 benchmark; expands to the union of
        # those suites' task ids when task_id_filter was not given explicitly.
        self.active_suites = cfg.get("active_suites", None)
        if self.active_suites is not None:
            self.active_suites = list(self.active_suites)
        if self.task_id_filter is None and self.active_suites:
            if str(cfg.task_suite_name).lower() != "libero_130":
                raise ValueError(
                    "active_suites requires task_suite_name == 'libero_130' "
                    f"(got '{cfg.task_suite_name}')"
                )
            self.task_id_filter = expand_active_suites_to_task_ids(self.active_suites)

        # Optional per-suite oversampling for rehearsal weighting: the new suite gets more
        # on-policy data than each old suite ({suite_name: weight}, e.g. new=1.0/old=0.3).
        # Applied to TRAINING sampling only (never eval). None -> uniform (unchanged).
        self.suite_sample_weights = cfg.get("suite_sample_weights", None)
        if self.suite_sample_weights is not None:
            # ${oc.decode:...} yields a NATIVE python dict (not a DictConfig), so
            # to_container would reject it; only convert when it really is a config.
            if OmegaConf.is_config(self.suite_sample_weights):
                self.suite_sample_weights = OmegaConf.to_container(
                    self.suite_sample_weights, resolve=True
                )
            self.suite_sample_weights = dict(self.suite_sample_weights)

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0

        self.task_suite: Benchmark = get_benchmark_overridden(cfg.task_suite_name)()

        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        self.update_reset_state_ids()
        self._init_task_and_trial_ids()
        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward
        self.use_step_penalty = getattr(cfg, "use_step_penalty", False)

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.video_cfg = cfg.video_cfg
        self.current_raw_obs = None

    def _init_env(self):
        env_fns = self.get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def get_env_fns(self):
        env_fn_params = self.get_env_fn_params()
        env_fns = []

        current_type_val = get_libero_type()

        for env_fn_param in env_fn_params:

            def env_fn(param=env_fn_param, _type_val=current_type_val):
                os.environ["LIBERO_TYPE"] = _type_val
                seed = param.pop("seed")

                if _type_val in ["pro", "plus"]:
                    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]

                    pkg_name = f"libero{_type_val}"
                    core_name = f"{pkg_name}.{pkg_name}"

                    try:
                        real_pkg = importlib.import_module(pkg_name)
                        real_core = importlib.import_module(core_name)
                        real_bench = importlib.import_module(f"{core_name}.benchmark")
                        real_envs = importlib.import_module(f"{core_name}.envs")

                        sys.modules["libero"] = real_pkg
                        sys.modules["libero.libero"] = real_core
                        sys.modules["libero.libero.benchmark"] = real_bench
                        sys.modules["libero.libero.envs"] = real_envs

                        loaded_path = os.path.dirname(real_core.__file__)
                        os.environ["LIBERO_ASSET_ROOT"] = os.path.join(
                            loaded_path, "assets"
                        )
                        os.environ["LIBERO_BDDL_PATH"] = os.path.join(
                            loaded_path, "bddl_files"
                        )
                        os.environ["LIBERO_INIT_STATES_PATH"] = os.path.join(
                            loaded_path, "init_files"
                        )

                        WorkerEnv = real_envs.OffScreenRenderEnv

                    except ImportError as e:
                        print(f"[Worker Env Error] {e}")
                        raise e
                else:
                    from libero.libero.envs import OffScreenRenderEnv as WorkerEnv

                env = WorkerEnv(**param)
                env.seed(seed)
                return env

            env_fns.append(env_fn)
        return env_fns

    def get_env_fn_params(self, env_idx=None):
        env_fn_params = []
        base_env_args = OmegaConf.to_container(self.cfg.init_params, resolve=True)

        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        raw_suffix = os.environ.get(
            "LIBERO_SUFFIX",
            os.environ.get(
                "LIBERO_PERTURBATION",
                self.cfg.get("perturbation_suffix", None)
                if hasattr(self.cfg, "get")
                else None,
            ),
        )
        if variant == "pro":
            import liberopro.liberopro as l_pro

            bddl_root = l_pro.get_libero_path("bddl_files")
        elif variant == "plus":
            import liberoplus.liberoplus as l_plus

            bddl_root = l_plus.get_libero_path("bddl_files")
        else:
            from libero.libero import get_libero_path

            bddl_root = get_libero_path("bddl_files")

        suite_name = self.cfg.task_suite_name.lower()
        suite_keyword = suite_name.replace("libero_", "").strip()

        task_descriptions = []
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        for env_id in range(self.num_envs):
            if env_id not in env_idx:
                task_descriptions.append(
                    self.task_descriptions[env_id]
                    if hasattr(self, "task_descriptions")
                    else ""
                )
                continue

            task = self.task_suite.get_task(self.task_ids[env_id])
            folder_name = task.problem_folder
            file_name = task.bddl_file
            original_path = os.path.join(bddl_root, folder_name, file_name)

            final_path = original_path

            if variant == "pro":
                pro_suffix = raw_suffix.replace(".bddl", "") if raw_suffix else None

                valid_perts = ["_lan", "_object", "_swap", "_task"]
                if pro_suffix == "all":
                    filter_perts = valid_perts
                elif pro_suffix is not None:
                    # Map bare name (e.g. "task") to directory suffix (e.g. "_task")
                    normalized = (
                        f"_{pro_suffix}"
                        if not pro_suffix.startswith("_")
                        else pro_suffix
                    )
                    filter_perts = [normalized] if normalized in valid_perts else []
                else:
                    filter_perts = []

                if filter_perts:
                    all_sub_dirs = [
                        d
                        for d in os.listdir(bddl_root)
                        if os.path.isdir(os.path.join(bddl_root, d))
                        and suite_keyword in d
                        and any(d.endswith(pert) for pert in filter_perts)
                    ]

                    core_task_name = file_name.replace(".bddl", "")
                    all_candidates = []

                    for sub_dir in all_sub_dirs:
                        target_dir_path = os.path.join(bddl_root, sub_dir)
                        matches = [
                            os.path.join(target_dir_path, f)
                            for f in os.listdir(target_dir_path)
                            if core_task_name in f and f.endswith(".bddl")
                        ]
                        all_candidates.extend(matches)

                    if all_candidates:
                        all_candidates.sort()
                        if getattr(self.cfg, "is_eval", False):
                            idx_offset = (
                                list(env_idx).index(env_id) if env_id in env_idx else 0
                            )
                            final_path = all_candidates[
                                (self.seed + idx_offset) % len(all_candidates)
                            ]
                        else:
                            final_path = self._generator.choice(all_candidates)

            elif variant == "plus":
                plus_suffix = raw_suffix.replace(".bddl", "") if raw_suffix else None
                if plus_suffix == "all":
                    clean_name = file_name.replace(".bddl", "")
                    for marker in [
                        "_view",
                        "_initstate",
                        "_noise",
                        "_sample",
                        "_light",
                        "_table",
                        "_add_1",
                        "_lan",
                        "_language",
                        "_copy",
                        "_level",
                        "_tb",
                    ]:
                        if marker in clean_name:
                            clean_name = clean_name.split(marker)[0]
                            break

                    suite_pattern = folder_name.replace("_", "").lower()
                    all_dirs = [
                        d
                        for d in os.listdir(bddl_root)
                        if os.path.isdir(os.path.join(bddl_root, d))
                    ]
                    search_dirs = [
                        os.path.join(bddl_root, d)
                        for d in all_dirs
                        if suite_pattern in d.lower().replace("_", "")
                    ]

                    if not search_dirs:
                        search_dirs = [os.path.join(bddl_root, folder_name)]

                    all_candidates = []
                    for target_dir in search_dirs:
                        matches = [
                            f
                            for f in glob.glob(os.path.join(target_dir, "*.bddl"))
                            if clean_name in os.path.basename(f)
                        ]
                        all_candidates.extend(matches)

                    if all_candidates:
                        all_candidates.sort()
                        if getattr(self.cfg, "is_eval", False):
                            idx_offset = (
                                list(env_idx).index(env_id) if env_id in env_idx else 0
                            )
                            final_path = all_candidates[
                                (self.seed + idx_offset) % len(all_candidates)
                            ]
                        else:
                            final_path = self._generator.choice(all_candidates)

            env_fn_params.append(
                {
                    **base_env_args,
                    "bddl_file_name": final_path,
                    "seed": self.seed,
                }
            )
            task_descriptions.append(task.language)

        self.task_descriptions = task_descriptions
        return env_fn_params

    def _compute_total_num_group_envs(self):
        self.total_num_group_envs = 0
        self.trial_id_bins = []
        for task_id in range(self.task_suite.get_num_tasks()):
            task_num_trials = len(self.task_suite.get_task_init_states(task_id))
            self.trial_id_bins.append(task_num_trials)
            self.total_num_group_envs += task_num_trials
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

        if self.task_id_filter is not None:
            num_tasks = len(self.trial_id_bins)
            validated_tids = []
            for tid in self.task_id_filter:
                if not isinstance(tid, (int, np.integer)):
                    raise ValueError(
                        f"task_id_filter must contain ints, got "
                        f"{type(tid).__name__}: {tid}"
                    )
                tid_int = int(tid)
                if tid_int < 0 or tid_int >= num_tasks:
                    raise ValueError(
                        f"task_id {tid_int} in task_id_filter is out of range "
                        f"[0, {num_tasks - 1}]"
                    )
                validated_tids.append(tid_int)
            validated_tids = sorted(set(validated_tids))

            self._valid_reset_state_ids = []
            for tid in validated_tids:
                start = self.cumsum_trial_id_bins[tid - 1] if tid > 0 else 0
                end = self.cumsum_trial_id_bins[tid]
                self._valid_reset_state_ids.extend(range(start, end))
            self._valid_reset_state_ids = np.array(self._valid_reset_state_ids)
        else:
            self._valid_reset_state_ids = None

        # Rehearsal weighting: oversample the current suite's reset states relative to old
        # suites so the new task gets more on-policy data (== more teacher weight in the
        # averaged OPD loss). TRAINING only; eval keeps uniform coverage. Built once here,
        # consumed by _get_random_reset_state_ids. None -> uniform (unchanged behavior).
        self._train_biased_reset_state_ids = None
        if (
            not self.cfg.is_eval
            and getattr(self, "suite_sample_weights", None)
            and self._valid_reset_state_ids is not None
        ):
            self._train_biased_reset_state_ids = self._build_suite_weighted_pool()

    def _build_suite_weighted_pool(self):
        """Tile ``_valid_reset_state_ids`` so each suite's share of the training sampling
        pool is proportional to ``suite_sample_weights`` (integer repeats, smallest
        positive weight -> 1x). Returns None if no active suite has a positive weight."""
        id_to_suite = get_libero130_task_id_to_suite()
        task_ids, _ = self._get_task_and_trial_ids_from_reset_state_ids(
            self._valid_reset_state_ids
        )
        weights = [
            float(self.suite_sample_weights.get(id_to_suite.get(int(t)), 0.0))
            for t in task_ids
        ]
        positive = [w for w in weights if w > 0.0]
        if not positive:
            return None
        min_w = min(positive)
        pool = []
        for reset_state_id, w in zip(self._valid_reset_state_ids, weights):
            reps = int(round(w / min_w)) if w > 0.0 else 0
            if reps > 0:
                pool.extend([reset_state_id] * reps)
        return np.array(pool) if pool else None

    def update_reset_state_ids(self):
        if self.cfg.is_eval or self.cfg.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _init_task_and_trial_ids(self):
        self.task_ids, self.trial_ids = (
            self._get_task_and_trial_ids_from_reset_state_ids(self.reset_state_ids)
        )

    def _get_random_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (num_reset_states,), dtype=int
            )
        elif self._valid_reset_state_ids is not None:
            # use the suite-weighted training pool when present (rehearsal oversampling),
            # else uniform over all valid reset states.
            pool = self._train_biased_reset_state_ids
            if pool is None:
                pool = self._valid_reset_state_ids
            indices = self._generator.integers(
                low=0, high=len(pool), size=(num_reset_states,)
            )
            reset_state_ids = pool[indices]
        else:
            reset_state_ids = self._generator.integers(
                low=0, high=self.total_num_group_envs, size=(num_reset_states,)
            )
        return reset_state_ids

    def get_reset_state_ids_all(self):
        if self._valid_reset_state_ids is not None:
            reset_state_ids = self._valid_reset_state_ids.copy()
        else:
            reset_state_ids = np.arange(self.total_num_group_envs)

        if not self.cfg.is_eval:
            self._generator_ordered.shuffle(reset_state_ids)
        elif self.cfg.get("eval_shuffle_for_task_coverage", False):
            # Opt-in: default eval order is np.arange(total), so reshape row 0 holds
            # consecutive IDs [0..num_group-1] which all land in task 0
            # (cumsum_trial_id_bins puts task 0 at [0, trials_per_task_0)). For
            # bank-collection runs where the bank must span all tasks within a single
            # batch, shuffle deterministically with cfg.seed for reproducibility.
            np.random.default_rng(seed=self.cfg.seed).shuffle(reset_state_ids)

        # Ensure we have enough IDs for all processes by tiling if needed
        if len(reset_state_ids) < self.total_num_processes:
            repeats = (self.total_num_processes // len(reset_state_ids)) + 1
            reset_state_ids = np.tile(reset_state_ids, repeats)

        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self.total_num_processes
        )
        reset_state_ids = reset_state_ids[:valid_size]
        reset_state_ids = reset_state_ids.reshape(self.total_num_processes, -1)
        return reset_state_ids

    def _get_ordered_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (self.num_group,), dtype=int
            )
        else:
            if self.start_idx + num_reset_states > len(self.reset_state_ids_all[0]):
                self.reset_state_ids_all = self.get_reset_state_ids_all()
                self.start_idx = 0
            reset_state_ids = self.reset_state_ids_all[self.seed_offset][
                self.start_idx : self.start_idx + num_reset_states
            ]
            self.start_idx = self.start_idx + num_reset_states
        return reset_state_ids

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        task_ids = []
        trial_ids = []
        # get task id and trial id from reset state ids
        for reset_state_id in reset_state_ids:
            start_pivot = 0
            for task_id, end_pivot in enumerate(self.cumsum_trial_id_bins):
                if reset_state_id < end_pivot and reset_state_id >= start_pivot:
                    task_ids.append(task_id)
                    trial_ids.append(reset_state_id - start_pivot)
                    break
                start_pivot = end_pivot

        return np.array(task_ids), np.array(trial_ids)

    def _get_reset_states(self, env_idx):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        init_state = [
            self.task_suite.get_task_init_states(self.task_ids[env_id])[
                self.trial_ids[env_id]
            ]
            for env_id in env_idx
        ]
        return init_state

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.success_episode_len = np.zeros(self.num_envs, dtype=np.int32)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self.success_episode_len[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self.success_episode_len[:] = 0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        # Only accumulate returns while not yet succeeded
        self.returns += step_reward * (~self.success_once)
        # Record episode_len at first success
        new_success_mask = terminations & ~self.success_once
        if new_success_mask.any():
            self.success_episode_len[new_success_mask] = self.elapsed_steps[
                new_success_mask
            ]

        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()

        # Use success episode_len for reward if already succeeded, else current elapsed
        episode_len_for_reward = np.where(
            self.success_once, self.success_episode_len, self.elapsed_steps
        )
        episode_info["reward"] = episode_info["return"] / np.maximum(
            episode_len_for_reward, 1
        )

        # PER-SUITE success, for multi-teacher CL. The combined success_once averages the
        # suites together, which hides exactly what we need to see: whether the suites move
        # ANTI-CORRELATED step to step (= the teachers really are fighting) or all drift down
        # together (= a shared capacity/drift problem, not a conflict).
        # Emitted as numerator/denominator PAIRS rather than a ratio: a rank may hold zero envs
        # of a given suite, and a per-rank ratio would be undefined there and poison the
        # cross-rank average. num/den each aggregate linearly, so suite SR = mean(num)/mean(den)
        # is correct no matter how many envs of that suite a rank happens to hold.
        # NB: only the id->suite dict is cached. self.task_ids is re-drawn every reset
        # (update_reset_state_ids -> _init_task_and_trial_ids), so caching the per-env suite
        # array would go stale and mis-attribute successes to the wrong suite.
        if getattr(self, "_id_to_suite", None) is None:
            self._id_to_suite = get_libero130_task_id_to_suite()
            # FIXED key set, computed from the id->suite map (identical on every rank), NOT from
            # the suites this rank happens to hold. all_reduce_dict packs the metric dict into ONE
            # tensor whose size is the KEY COUNT, so if rank 0 emits {spatial,object} and rank 1
            # emits {goal,libero_10} the two ranks all-reduce different-sized tensors and the
            # collective blocks forever -- which is what hung the 4-GPU 4-teacher run on 2026-08-19
            # (85 min of zero log output with all four GPUs pinned at 100%: NCCL spin-waits, so a
            # deadlock looks exactly like full utilisation). Single-rank and single-suite runs never
            # exposed it because every rank emitted the same keys by luck.
            self._all_suites = sorted(set(self._id_to_suite.values())) + ["unknown"]
        suite_of_env = np.array(
            [str(self._id_to_suite.get(int(t), "unknown")) for t in self.task_ids]
        )
        for _s in self._all_suites:
            _m = (suite_of_env == _s).astype(np.float32)   # all-zero when this rank holds none
            episode_info[f"succ_num_{_s}"] = self.success_once.astype(np.float32) * _m
            episode_info[f"succ_den_{_s}"] = _m

        infos["episode"] = to_tensor(episode_info)
        return infos

    def _extract_image_and_state(self, obs):
        return {
            "full_image": get_libero_image(obs),
            "wrist_image": get_libero_wrist_image(obs),
            "state": np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ),
        }

    # ---- OFF-DEMO state capture (opt-in via RLINF_DUMP_OBS_DIR) ---------------------------------
    # The states a policy actually REACHES are exactly the ones the demonstrations do not contain,
    # and our measurements say that is where the lost capability lives (fitting the demos BETTER than
    # the original model still leaves success 22 points down). Setting RLINF_DUMP_OBS_DIR during any
    # rollout writes those visited states to disk, so an old checkpoint can be queried there OFFLINE
    # afterwards -- this is what makes the "ask the teacher off the demo manifold" anchor possible
    # WITHOUT on-policy training. Written in shards (not one file per state) to avoid a small-file
    # explosion on the shared disk.
    #   RLINF_DUMP_OBS_EVERY  keep 1 step in N   (default 4)
    #   RLINF_DUMP_OBS_MAX    stop after N states (default 20000)
    def _maybe_dump_obs(self, ias_list):
        d = os.environ.get("RLINF_DUMP_OBS_DIR", "")
        if not d:
            return
        self._dump_step = getattr(self, "_dump_step", -1) + 1
        if self._dump_step % int(os.environ.get("RLINF_DUMP_OBS_EVERY", "4")):
            return
        self._dump_n = getattr(self, "_dump_n", 0)
        cap = int(os.environ.get("RLINF_DUMP_OBS_MAX", "20000"))
        if self._dump_n >= cap:
            return
        buf = getattr(self, "_dump_buf", None)
        if buf is None:
            buf = self._dump_buf = []
            os.makedirs(d, exist_ok=True)
        if not hasattr(self, "_ep_id"):
            self._ep_id = np.zeros(self.num_envs, dtype=np.int64)
            self._ep_outcomes = []
        sig = getattr(self, "_noise_sigma", None)
        descs = getattr(self, "task_descriptions", None)
        for i, ias in enumerate(ias_list):
            if self._dump_n >= cap:
                break
            buf.append(
                (
                    np.asarray(ias["full_image"], dtype=np.uint8),
                    np.asarray(ias["wrist_image"], dtype=np.uint8),
                    np.asarray(ias["state"], dtype=np.float32),
                    (descs[i] if descs is not None and i < len(descs) else ""),
                    self._dump_step,
                    i,                                        # env index
                    int(self._ep_id[i]),                      # episode index within that env
                    float(sig[i]) if sig is not None else 0.0,  # the noise level that produced it
                )
            )
            self._dump_n += 1
        # flush on a full shard, and also the moment the cap is reached (there is no close() hook on
        # this env, so a partial tail buffer would otherwise never be written)
        if len(buf) >= 500 or self._dump_n >= cap:
            self._flush_dump(d)

    def _flush_dump(self, d):
        buf = self._dump_buf
        if not buf:
            return
        shard = getattr(self, "_dump_shard", 0)
        self._dump_shard = shard + 1
        # zlib on ~200MB of uint8 frames per shard costs real CPU inside the env worker, and a 48-env
        # osmesa rollout is already using ~50 of 56 cores. Over a multi-pass collection that is the
        # difference between load ~30 and load ~52, so compression is OPT-IN (disk is not the
        # constraint here: 1.7TB free, ~400KB/state uncompressed).
        _save = np.savez_compressed if os.environ.get("RLINF_DUMP_OBS_COMPRESS", "0") == "1" else np.savez
        _save(
            os.path.join(d, f"states_{os.getpid()}_{shard:04d}.npz"),
            full_image=np.stack([b[0] for b in buf]),
            wrist_image=np.stack([b[1] for b in buf]),
            state=np.stack([b[2] for b in buf]),
            task=np.array([b[3] for b in buf], dtype=object),
            step=np.array([b[4] for b in buf], dtype=np.int32),
            env_idx=np.array([b[5] for b in buf], dtype=np.int32),
            ep_id=np.array([b[6] for b in buf], dtype=np.int32),
            sigma=np.array([b[7] for b in buf], dtype=np.float32),
        )
        buf.clear()

    # DART-style noise injection (Laskey et al. 2017): perturb the SUPERVISOR's actions during
    # collection so the rollout leaves the demonstration manifold while staying near the region the
    # supervisor can still handle -- "corrective examples at the boundary of the supervisor's policy,
    # without visiting highly sub-optimal states". Several sigmas can be measured in ONE rollout by
    # assigning them round-robin across envs (RLINF_ACT_NOISE="0.1,0.2,0.3"), which saves two extra
    # CPU-saturating rollout passes. The gripper dimension is excluded by default: it is effectively
    # binary, so noise there flips open/closed mid-grasp -- an instant, unrecoverable failure rather
    # than the small recoverable deviation we are trying to create.
    def _maybe_inject_noise(self, actions):
        spec = os.environ.get("RLINF_ACT_NOISE", "")
        if not spec or actions is None:
            return actions
        if not hasattr(self, "_noise_sigma"):
            sig = np.array([float(x) for x in spec.split(",") if x.strip()], dtype=np.float32)
            self._noise_sigma = sig[np.arange(self.num_envs) % len(sig)]
            self._noise_ndim = int(os.environ.get("RLINF_ACT_NOISE_DIMS", "6"))
            print(
                f"[dart] action noise sigmas={sorted(set(sig.tolist()))} on first "
                f"{self._noise_ndim} dims (gripper excluded)",
                flush=True,
            )
        a = np.asarray(actions, dtype=np.float32).copy()
        k = min(self._noise_ndim, a.shape[-1])
        sig = self._noise_sigma.reshape((-1,) + (1,) * (a.ndim - 1))[: a.shape[0]]
        a[..., :k] += np.random.normal(0.0, 1.0, size=a[..., :k].shape).astype(np.float32) * sig
        return a

    # A state is only usable as a distillation input if the episode it came from SUCCEEDED: in a
    # failed episode the teacher itself was wrong there, so its action is a wrong target. This
    # ledger is what the offline filter joins against.
    # It is written HERE, not in _flush_dump: episodes only end at the horizon (~512 steps), long
    # after the state cap has stopped the dumping/flushing, so writing it on flush would leave the
    # ledger permanently empty and make success-filtering impossible.
    def _maybe_record_episode_end(self, dones):
        d = os.environ.get("RLINF_DUMP_OBS_DIR", "")
        if not d:
            return
        idx = np.nonzero(np.asarray(dones).reshape(-1))[0]
        if len(idx) == 0:
            return
        if not hasattr(self, "_ep_id"):
            self._ep_id = np.zeros(self.num_envs, dtype=np.int64)
            self._ep_outcomes = []
        sig = getattr(self, "_noise_sigma", None)
        for i in idx:
            self._ep_outcomes.append(
                (int(i), int(self._ep_id[i]), bool(self.success_once[i]), float(sig[i]) if sig is not None else 0.0)
            )
            self._ep_id[i] += 1
        out = self._ep_outcomes
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(
            os.path.join(d, f"episodes_{os.getpid()}.npz"),
            env_idx=np.array([o[0] for o in out], dtype=np.int32),
            ep_id=np.array([o[1] for o in out], dtype=np.int32),
            success=np.array([o[2] for o in out], dtype=bool),
            sigma=np.array([o[3] for o in out], dtype=np.float32),
        )

    def _wrap_obs(self, obs_list):
        images_and_states_list = []
        for obs in obs_list:
            images_and_states = self._extract_image_and_state(obs)
            images_and_states_list.append(images_and_states)
        self._maybe_dump_obs(images_and_states_list)

        images_and_states = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list)
        )

        full_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["full_image"]]
        )
        wrist_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["wrist_image"]]
        )

        states = images_and_states["state"]

        obs = {
            "main_images": full_image_tensor,
            "wrist_images": wrist_image_tensor,
            "states": states,
            "task_descriptions": self.task_descriptions,
        }
        return obs

    def _reconfigure(self, reset_state_ids, env_idx):
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            task_changed = self.task_ids[env_id] != task_ids[j]
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
            if task_changed or not getattr(self.cfg, "is_eval", False):
                reconfig_env_idx.append(env_id)
        if reconfig_env_idx:
            env_fn_params = self.get_env_fn_params(reconfig_env_idx)
            self.env.reconfigure_env_fns(env_fn_params, reconfig_env_idx)
        self.env.seed(self.seed * len(env_idx))
        self.env.reset(id=env_idx)
        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        if variant != "plus":
            init_state = self._get_reset_states(env_idx=env_idx)
            self.env.set_init_state(init_state=init_state, id=env_idx)

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if self.is_start:
            reset_state_ids = (
                self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self._is_start = False

        if reset_state_ids is None:
            num_reset_states = len(env_idx)
            reset_state_ids = self._get_random_reset_state_ids(num_reset_states)

        self._reconfigure(reset_state_ids, env_idx)
        for _ in range(15):
            zero_actions = np.zeros((len(env_idx), 7))
            if self.cfg.reset_gripper_open:
                zero_actions[:, -1] = -1
            raw_obs, _reward, terminations, info_lists = self.env.step(
                zero_actions, env_idx
            )
        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs
        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]

        obs = self._wrap_obs(self.current_raw_obs)
        self._reset_metrics(env_idx)
        infos = {}
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        """Step the environment with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = self._maybe_inject_noise(actions)   # no-op unless RLINF_ACT_NOISE is set

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions)
        self.current_raw_obs = raw_obs
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        obs = self._wrap_obs(raw_obs)

        step_reward = self._calc_step_reward(terminations)

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        self._maybe_record_episode_end(dones)   # no-op unless RLINF_DUMP_OBS_DIR is set
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        if self.cfg.is_eval:
            self.update_reset_state_ids()
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=self.reset_state_ids[env_idx]
            if self.use_fixed_reset_state_ids
            else None,
        )
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations):
        step_penalty = -1 if self.use_step_penalty else 0
        termination_bonus = self.cfg.reward_coef * terminations
        reward = step_penalty + termination_bonus

        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return reward_diff
        else:
            return reward
