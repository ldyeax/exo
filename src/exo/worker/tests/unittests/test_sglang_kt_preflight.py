import pytest
from pydantic import ValidationError

from exo.shared.models.model_cards import ModelId
from exo.worker.sglang_kt.launch_spec import (
    SglangKtProcessLaunchSpec,
    build_glm_5_2_fp8_process_launch_specs,
)
from exo.worker.sglang_kt.preflight import (
    SglangKtHostPreflightObservation,
    SglangKtModelSnapshotReceiptObservation,
    SglangKtPreflightFailed,
    SglangKtPreflightPassed,
    SglangKtPythonVersionObservation,
    SglangKtRuntimeObservation,
    SglangKtRuntimeValidationReceiptObservation,
    evaluate_sglang_kt_preflight,
)
from exo.worker.tests.unittests.test_sglang_kt_launch_spec import (
    PYTHON_EXECUTABLE,
    make_plan,
)

CONFIG_SHA256 = "a" * 64
FULL_INDEXER_LAYER_STARTS = (0, 1, 2, *range(6, 78, 4))


def make_specs() -> tuple[SglangKtProcessLaunchSpec, ...]:
    return build_glm_5_2_fp8_process_launch_specs(make_plan(), PYTHON_EXECUTABLE)


def make_runtime(spec: SglangKtProcessLaunchSpec) -> SglangKtRuntimeObservation:
    return SglangKtRuntimeObservation(
        executable=spec.executable,
        python_implementation="CPython",
        python_version=SglangKtPythonVersionObservation(
            major=3,
            minor=13,
            patch=14,
        ),
        sglang_revision=spec.expected_sglang_revision,
        ktransformers_revision=spec.expected_ktransformers_revision,
        transformers_distribution_version=spec.required_transformers_version,
        transformers_module_version=spec.required_transformers_version,
    )


def make_runtime_validation_receipt(
    spec: SglangKtProcessLaunchSpec,
) -> SglangKtRuntimeValidationReceiptObservation:
    return SglangKtRuntimeValidationReceiptObservation(
        gpu_uuid=spec.gpu_uuid,
        gpu_compute_capability=(8, 6),
        cpu_cores=spec.cpu_cores,
        memory_nodes=spec.memory_nodes,
        executed_cpu_backend="AMX",
        model_id=spec.model_id,
        model_revision=spec.expected_model_revision,
        model_config_sha256=CONFIG_SHA256,
        sglang_revision=spec.expected_sglang_revision,
        ktransformers_revision=spec.expected_ktransformers_revision,
        transformers_distribution_version=spec.required_transformers_version,
        transformers_module_version=spec.required_transformers_version,
        torch_version="2.10.0+cu130",
        cuda_version="13.0",
        sgl_kernel_build_id="sgl-kernel-test-build",
        deep_gemm_build_id="deep-gemm-test-build",
        kv_cache_dtype="fp8_e4m3",
        capabilities=(
            "kt_tp_group_local_broadcast_v1",
            "glm52_nsa_sm86_short_forward_v1",
            "kt_physical_numa_mapping_v1",
            "kt_process_cpu_affinity_v1",
            "kt_fp8_amx_executed_v1",
        ),
    )


def make_host_observation(
    host_specs: tuple[SglangKtProcessLaunchSpec, ...],
) -> SglangKtHostPreflightObservation:
    first_spec = host_specs[0]
    model_paths = tuple(dict.fromkeys(spec.model_path for spec in host_specs))
    weight_paths = tuple(
        dict.fromkeys(spec.ktransformers_weight_path for spec in host_specs)
    )
    snapshot_paths = tuple(dict.fromkeys((*model_paths, *weight_paths)))
    rank_zero_spec = next(
        (spec for spec in host_specs if spec.pipeline_rank == 0),
        None,
    )
    return SglangKtHostPreflightObservation(
        node_id=first_spec.node_id,
        runtime=make_runtime(first_spec),
        runtime_validation_receipts=tuple(
            make_runtime_validation_receipt(spec) for spec in host_specs
        ),
        readable_directories=tuple(dict.fromkeys((*model_paths, *weight_paths))),
        model_snapshot_receipts=tuple(
            SglangKtModelSnapshotReceiptObservation(
                model_path=snapshot_path,
                model_id=first_spec.model_id,
                revision=first_spec.expected_model_revision,
                weight_format="safetensors",
                ktransformers_method=first_spec.ktransformers_method,
                config_sha256=CONFIG_SHA256,
                full_indexer_layer_starts=FULL_INDEXER_LAYER_STARTS,
                receipt_verified=True,
                snapshot_complete=True,
            )
            for snapshot_path in snapshot_paths
        ),
        gpu_uuids=tuple(spec.gpu_uuid for spec in host_specs),
        cpu_cores=tuple(core for spec in host_specs for core in spec.cpu_cores),
        memory_nodes=tuple(
            dict.fromkeys(node for spec in host_specs for node in spec.memory_nodes)
        ),
        hca_devices=tuple(
            dict.fromkeys(device for spec in host_specs for device in spec.hca_devices)
        ),
        available_bind_endpoints=(
            *(spec.service_endpoint for spec in host_specs),
            *(
                (rank_zero_spec.distributed_coordinator,)
                if rank_zero_spec is not None
                else ()
            ),
        ),
    )


def make_observations(
    specs: tuple[SglangKtProcessLaunchSpec, ...],
) -> tuple[SglangKtHostPreflightObservation, ...]:
    return (
        make_host_observation((specs[0], specs[1])),
        make_host_observation((specs[2],)),
    )


def checks_for_rank(result: SglangKtPreflightFailed, rank: int) -> tuple[str, ...]:
    return tuple(
        failure.check for failure in result.failures if failure.pipeline_rank == rank
    )


def test_valid_observations_release_the_complete_process_group() -> None:
    specs = make_specs()
    observations = make_observations(specs)

    result = evaluate_sglang_kt_preflight(specs, observations)

    assert isinstance(result, SglangKtPreflightPassed)
    assert result.process_specs == specs
    assert (
        SglangKtPreflightPassed.model_validate_json(result.model_dump_json()) == result
    )


def test_version_only_runtime_facts_cannot_release_the_process_group() -> None:
    specs = make_specs()
    observations = tuple(
        observation.model_copy(update={"runtime_validation_receipts": ()})
        for observation in make_observations(specs)
    )

    result = evaluate_sglang_kt_preflight(specs, observations)

    assert isinstance(result, SglangKtPreflightFailed)
    assert all(
        checks_for_rank(result, rank) == ("runtime_validation_receipt",)
        for rank in range(3)
    )


@pytest.mark.parametrize(
    "receipt_update",
    (
        {"gpu_compute_capability": (9, 0)},
        {"cpu_cores": (999,)},
        {"memory_nodes": (999,)},
        {"model_config_sha256": "b" * 64},
        {"sglang_revision": "c" * 40},
        {"transformers_distribution_version": "5.6.0.post2"},
        {"capabilities": ("kt_tp_group_local_broadcast_v1",)},
    ),
)
def test_stale_or_incomplete_runtime_validation_receipt_fails_closed(
    receipt_update: dict[str, object],
) -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    first_receipt = dwagon.runtime_validation_receipts[0].model_copy(
        update=receipt_update
    )
    dwagon = dwagon.model_copy(
        update={
            "runtime_validation_receipts": (
                first_receipt,
                dwagon.runtime_validation_receipts[1],
            )
        }
    )

    result = evaluate_sglang_kt_preflight(specs, (dwagon, fwuff))

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == ("runtime_validation_receipt",)
    assert checks_for_rank(result, 1) == ()
    assert checks_for_rank(result, 2) == ()


def test_snapshot_receipt_boundaries_must_authorize_every_pipeline_start() -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    incomplete_boundaries = tuple(
        start for start in FULL_INDEXER_LAYER_STARTS if start != 58
    )
    dwagon_receipt = dwagon.model_snapshot_receipts[0].model_copy(
        update={"full_indexer_layer_starts": incomplete_boundaries}
    )
    fwuff_receipt = fwuff.model_snapshot_receipts[0].model_copy(
        update={"full_indexer_layer_starts": incomplete_boundaries}
    )

    result = evaluate_sglang_kt_preflight(
        specs,
        (
            dwagon.model_copy(update={"model_snapshot_receipts": (dwagon_receipt,)}),
            fwuff.model_copy(update={"model_snapshot_receipts": (fwuff_receipt,)}),
        ),
    )

    assert isinstance(result, SglangKtPreflightFailed)
    for rank in range(3):
        assert checks_for_rank(result, rank) == (
            "model_revision_receipt",
            "ktransformers_weight_revision_receipt",
        )


def test_runtime_mismatches_are_aggregated_for_each_affected_stage() -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    bad_runtime = SglangKtRuntimeObservation(
        executable="/opt/wrong/bin/python",
        python_implementation="PyPy",
        python_version=SglangKtPythonVersionObservation(
            major=3,
            minor=10,
            patch=16,
        ),
        sglang_revision="5" * 40,
        ktransformers_revision="6" * 40,
        transformers_distribution_version="5.6.0.post2",
        transformers_module_version="5.6.0.post2",
    )
    dwagon = dwagon.model_copy(update={"runtime": bad_runtime})

    result = evaluate_sglang_kt_preflight(specs, (dwagon, fwuff))

    assert isinstance(result, SglangKtPreflightFailed)
    expected_checks = (
        "python_executable",
        "python_implementation",
        "python_version",
        "sglang_revision",
        "ktransformers_revision",
        "transformers_distribution_version",
        "transformers_module_version",
    )
    assert checks_for_rank(result, 0) == expected_checks
    assert checks_for_rank(result, 1) == expected_checks
    assert checks_for_rank(result, 2) == ()
    assert all(
        failure.message.startswith(
            f"pipeline rank {failure.pipeline_rank} on {failure.node_id}:"
        )
        for failure in result.failures
    )


def test_unobserved_runtime_facts_fail_closed() -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    dwagon = dwagon.model_copy(update={"runtime": SglangKtRuntimeObservation()})

    result = evaluate_sglang_kt_preflight(specs, (dwagon, fwuff))

    assert isinstance(result, SglangKtPreflightFailed)
    expected_checks = (
        "python_executable",
        "python_implementation",
        "python_version",
        "sglang_revision",
        "ktransformers_revision",
        "transformers_distribution_version",
        "transformers_module_version",
    )
    assert checks_for_rank(result, 0) == expected_checks
    assert all(
        failure.observed == ("<unobserved>",)
        for failure in result.failures
        if failure.pipeline_rank == 0
    )


def test_missing_stage_resources_and_ports_fail_as_one_group() -> None:
    specs = make_specs()
    _dwagon, fwuff = make_observations(specs)
    empty_dwagon = SglangKtHostPreflightObservation(
        node_id=specs[0].node_id,
        runtime=make_runtime(specs[0]),
    )

    result = evaluate_sglang_kt_preflight(specs, (empty_dwagon, fwuff))

    assert isinstance(result, SglangKtPreflightFailed)
    common_checks = (
        "model_path",
        "ktransformers_weight_path",
        "model_revision_receipt",
        "ktransformers_weight_revision_receipt",
        "runtime_validation_receipt",
        "gpu_uuid",
        "cpu_cores",
        "memory_nodes",
        "hca_devices",
        "service_endpoint",
    )
    assert checks_for_rank(result, 0) == (
        *common_checks,
        "distributed_coordinator",
    )
    assert checks_for_rank(result, 1) == common_checks
    assert checks_for_rank(result, 2) == ()


def test_model_receipt_must_match_path_model_revision_and_completeness() -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    bad_receipt = SglangKtModelSnapshotReceiptObservation(
        model_path=specs[0].model_path,
        model_id=ModelId("zai-org/GLM-5-FP8"),
        revision="9" * 40,
        weight_format="safetensors",
        ktransformers_method="FP8",
        config_sha256=CONFIG_SHA256,
        full_indexer_layer_starts=FULL_INDEXER_LAYER_STARTS,
        receipt_verified=False,
        snapshot_complete=False,
    )
    dwagon = dwagon.model_copy(update={"model_snapshot_receipts": (bad_receipt,)})

    result = evaluate_sglang_kt_preflight(specs, (dwagon, fwuff))

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == (
        "model_revision_receipt",
        "ktransformers_weight_revision_receipt",
    )
    assert checks_for_rank(result, 1) == (
        "model_revision_receipt",
        "ktransformers_weight_revision_receipt",
    )
    assert result.failures[0].expected == (
        "zai-org/GLM-5.2-FP8",
        specs[0].expected_model_revision,
        "safetensors",
        "FP8",
        "config_sha256=<verified>",
        "pipeline starts on verified full indexers",
        "receipt_verified=True",
        "snapshot_complete=True",
    )
    assert result.failures[0].observed == (
        "zai-org/GLM-5-FP8",
        "9" * 40,
        "safetensors",
        "FP8",
        CONFIG_SHA256,
        "full_indexer_layer_starts="
        + ",".join(str(start) for start in FULL_INDEXER_LAYER_STARTS),
        "receipt_verified=False",
        "snapshot_complete=False",
    )


def test_ktransformers_weight_receipt_must_be_exact_and_compatible() -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    bad_weight_receipt = SglangKtModelSnapshotReceiptObservation(
        model_path=specs[0].ktransformers_weight_path,
        model_id=specs[0].model_id,
        revision=specs[0].expected_model_revision,
        weight_format="safetensors",
        ktransformers_method="BF16",
        config_sha256=CONFIG_SHA256,
        full_indexer_layer_starts=FULL_INDEXER_LAYER_STARTS,
        receipt_verified=True,
        snapshot_complete=True,
    )
    dwagon = dwagon.model_copy(
        update={
            "model_snapshot_receipts": (bad_weight_receipt,),
        }
    )

    result = evaluate_sglang_kt_preflight(specs, (dwagon, fwuff))

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == (
        "model_revision_receipt",
        "ktransformers_weight_revision_receipt",
    )
    assert checks_for_rank(result, 1) == (
        "model_revision_receipt",
        "ktransformers_weight_revision_receipt",
    )


def test_each_planned_endpoint_requires_an_injected_availability_fact() -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    dwagon = dwagon.model_copy(
        update={
            "available_bind_endpoints": (specs[0].service_endpoint,),
        }
    )

    result = evaluate_sglang_kt_preflight(specs, (dwagon, fwuff))

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == ("distributed_coordinator",)
    assert checks_for_rank(result, 1) == ("service_endpoint",)


@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_ambiguous_host_observations_fail_closed(duplicate: bool) -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    observations = (dwagon, dwagon, fwuff) if duplicate else (dwagon,)

    result = evaluate_sglang_kt_preflight(specs, observations)

    assert isinstance(result, SglangKtPreflightFailed)
    affected_ranks = (0, 1) if duplicate else (2,)
    for rank in affected_ranks:
        assert checks_for_rank(result, rank) == ("host_observation",)
    assert not isinstance(result, SglangKtPreflightPassed)


def test_host_observations_reject_ambiguous_duplicate_facts() -> None:
    specs = make_specs()

    with pytest.raises(ValidationError, match="gpu_uuids observations must be unique"):
        SglangKtHostPreflightObservation(
            node_id=specs[0].node_id,
            runtime=make_runtime(specs[0]),
            gpu_uuids=(specs[0].gpu_uuid, specs[0].gpu_uuid),
        )


def test_empty_or_duplicate_process_groups_are_rejected_before_release() -> None:
    specs = make_specs()

    with pytest.raises(ValueError, match="requires process specs"):
        evaluate_sglang_kt_preflight((), ())
    with pytest.raises(ValueError, match="ranks must be unique"):
        evaluate_sglang_kt_preflight((specs[0], specs[0]), ())
    with pytest.raises(ValueError, match="ranks must be contiguous"):
        evaluate_sglang_kt_preflight((specs[0], specs[2]), ())


def test_process_groups_must_share_one_canonical_launch_plan() -> None:
    specs = make_specs()
    other_plan = make_plan().model_copy(update={"model_revision": "7" * 40})
    other_specs = build_glm_5_2_fp8_process_launch_specs(other_plan, PYTHON_EXECUTABLE)

    with pytest.raises(ValueError, match="share one launch plan"):
        evaluate_sglang_kt_preflight(
            (specs[0], other_specs[1], specs[2]),
            (),
        )
