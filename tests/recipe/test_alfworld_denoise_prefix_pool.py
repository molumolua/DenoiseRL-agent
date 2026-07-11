import json

from recipe.alfworld_denoise.trajectory_prefix import PrefixPool


def test_sample_for_task_never_falls_back_to_another_task(tmp_path):
    pool_path = tmp_path / "prefixes.jsonl"
    records = [
        {
            "gamefile": "/data/pick_and_place/task-a/game.tw-pddl",
            "success": False,
            "actions": ["open cabinet 1"],
        },
        {
            "gamefile": "/data/pick_clean_then_place/task-b/game.tw-pddl",
            "success": False,
            "actions": ["take soap 1 from sinkbasin 1"],
        },
    ]
    pool_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    pool = PrefixPool(str(pool_path), seed=0)

    matches = pool.sample_for_task("/another-machine/task-a/game.tw-pddl", 8)

    assert len(matches) == 8
    assert {prefix.task_key for prefix in matches} == {"/data/pick_and_place/task-a/game.tw-pddl"}
    assert pool.sample_for_task("/data/look_at_obj/task-c/game.tw-pddl", 1) == []
