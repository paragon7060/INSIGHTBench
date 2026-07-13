from __future__ import annotations

import pytest

from cfg.helper import resolve_cabinet_collect_handle_id


def test_cabinet_collect_uses_handle_from_partmanip_asset_id_for_1ext() -> None:
    json_info = {"handle": ["handle_0", "handle_3"]}

    hand_id = resolve_cabinet_collect_handle_id(
        "StorageFurniture-47565-link_2-handle_3-joint_2-handlejoint_3",
        "1ext",
        json_info,
        "link_handle_relationship.json",
    )

    assert hand_id == "handle_3"


def test_cabinet_collect_supports_legacy_index_scene_keys() -> None:
    json_info = {"handle": ["handle_0", "handle_3"]}

    hand_id = resolve_cabinet_collect_handle_id(
        "StorageFurniture-47565",
        "1b",
        json_info,
        "link_handle_relationship.json",
    )

    assert hand_id == "handle_3"


def test_cabinet_collect_rejects_ambiguous_1ext_without_handle_id() -> None:
    json_info = {"handle": ["handle_0", "handle_3"]}

    with pytest.raises(ValueError, match="Cannot resolve cabinet handle"):
        resolve_cabinet_collect_handle_id(
            "StorageFurniture-47565",
            "1ext",
            json_info,
            "link_handle_relationship.json",
        )
