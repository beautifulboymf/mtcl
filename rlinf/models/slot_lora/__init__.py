# Copyright 2026 The RLinf Authors.
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

"""slot-LoRI: per-suite LoRA slots on mutually orthogonal input subspaces."""

from rlinf.models.slot_lora.inject import (
    SlotInjection,
    collect_slot_diag,
    enable_slot_diag,
    inject_slot_lora,
)
from rlinf.models.slot_lora.modules import (
    SlotGate,
    SlotLoRALinear,
    SlotOut,
    SlotProj,
)
from rlinf.models.slot_lora.orth import orth_error, orthogonalize
from rlinf.models.slot_lora.routing import match_suite_ids

__all__ = [
    "orthogonalize",
    "orth_error",
    "SlotProj",
    "SlotGate",
    "SlotOut",
    "SlotLoRALinear",
    "match_suite_ids",
    "inject_slot_lora",
    "SlotInjection",
    "enable_slot_diag",
    "collect_slot_diag",
]
