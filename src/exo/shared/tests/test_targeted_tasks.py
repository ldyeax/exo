from exo.shared.types.tasks import ConnectToGroup, LoadModel, StartWarmup
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId


def test_targeted_lifecycle_tasks_roundtrip() -> None:
    instance_id = InstanceId("instance-a")
    runner_id = RunnerId("runner-a")
    connect = ConnectToGroup(instance_id=instance_id, runner_id=runner_id)
    load = LoadModel(instance_id=instance_id, runner_id=runner_id)
    warmup = StartWarmup(instance_id=instance_id, runner_id=runner_id)

    assert ConnectToGroup.model_validate_json(connect.model_dump_json()) == connect
    assert LoadModel.model_validate_json(load.model_dump_json()) == load
    assert StartWarmup.model_validate_json(warmup.model_dump_json()) == warmup


def test_legacy_lifecycle_tasks_default_to_representative_runner() -> None:
    instance_id = InstanceId("instance-a")
    legacy_connect = ConnectToGroup.model_validate_json(
        '{"ConnectToGroup":{"task_id":"legacy-connect","task_status":"Pending",'
        '"instance_id":"instance-a"}}'
    )

    assert legacy_connect.runner_id is None
    assert LoadModel(instance_id=instance_id).runner_id is None
    assert StartWarmup(instance_id=instance_id).runner_id is None
