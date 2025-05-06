"""Evaluation metadata helper functions that control which evaluations are run."""

import json
import time
from enum import Enum

State = Enum(
    "State",
    [
        ("NOT_EVALUATED", 0),
        ("SUBMITTED", 1),
        ("RUNNING", 2),
        ("FINISHED", 3),
        ("FAILED", 4),
    ],
)

EVAL_METADATA_PATH = "/users/amarfurt/eval_metadata.json"


class EvalMetadata:
    def __init__(self, path=EVAL_METADATA_PATH):
        self.path = path
        with open(self.path) as f:
            self.metadata = json.load(f)

    def get_model_metadata(self, model_name):
        return self.metadata[model_name]

    def get_state(self, model_name, iteration):
        iterations_metadata = self.metadata[model_name]["iterations"]
        iteration = str(iteration)  # JSON keys are strings
        if iteration in iterations_metadata:
            state_name = iterations_metadata[iteration]["state"]
            return State[state_name]
        return None

    def update_iteration_metadata(self, model_name, iteration, new_state, **kwargs):
        iteration = str(iteration)  # JSON keys are strings
        new_state_name = new_state.name if type(new_state) is State else new_state
        if iteration not in self.metadata[model_name]["iterations"]:
            self.metadata[model_name]["iterations"][iteration] = {}
        self.metadata[model_name]["iterations"][iteration].update(kwargs)
        self.metadata[model_name]["iterations"][iteration]["state"] = new_state_name
        self.metadata[model_name]["iterations"][iteration]["timestamp"] = time.time()
        with open(self.path, "w") as f:
            json.dump(self.metadata, f, indent=4)
