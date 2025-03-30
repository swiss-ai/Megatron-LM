"""Manually changes an evaluation's state in the metadata (e.g. to rerun an evaluation)."""

from evaluations_metadata import EvalMetadata, State


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Manually set an evaluation's state.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument(
        "--state", required=True, choices=[s.name.lower() for s in State]
    )
    return parser.parse_args()


def main(args):
    new_state = None
    for state in State:
        if state.name.lower() == args.state:
            new_state = state
            break
    eval_metadata = EvalMetadata()
    eval_metadata.update_iteration_metadata(args.model_name, args.iteration, new_state)


if __name__ == "__main__":
    main(parse_args())
