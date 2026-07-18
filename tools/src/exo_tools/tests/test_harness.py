import argparse
from typing import Literal

from exo_tools.harness import Comm, add_common_instance_args, placement_filter


def test_placement_filter() -> None:
    cases: list[tuple[str, Literal["ring", "jaccl", "nccl", "both"], bool]] = [
        ("MlxRing", "ring", True),
        ("MlxJaccl", "jaccl", True),
        ("MlxNccl", "nccl", True),
        ("MlxNccl", "both", False),
        ("MlxNccl", "ring", False),
        ("MlxRing", "nccl", False),
    ]
    for instance_meta, wanted, expected in cases:
        assert placement_filter(instance_meta, wanted) is expected


def test_common_arguments_accept_nccl() -> None:
    parser = argparse.ArgumentParser()
    add_common_instance_args(parser)

    parser.parse_args(["--model", "test/model", "--instance-meta", "nccl"])

    assert Comm.NCCL.value == "MlxNccl"
