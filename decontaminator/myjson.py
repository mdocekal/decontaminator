# -*- coding: UTF-8 -*-
import json
from typing import Any

import orjson


def json_dump(obj, file):
    """
    Serializes object to JSON file.

    :param obj: Object to serialize.
    :param file: File to write JSON string.
    """

    json.dump(obj, file, separators=(',', ':'), ensure_ascii=False)


def json_dumps(obj):
    """
    Serializes object to JSON string.

    :param obj: Object to serialize.
    :return: JSON string.
    """

    return json.dumps(obj, separators=(',', ':'), ensure_ascii=False)


def json_load(file) -> Any:
    """
    Deserializes object from JSON file.

    :param file: File with JSON string.
    :return: Deserialized object.
    """

    return json.load(file)


def json_loads(json_str: str) -> Any:
    """
    Deserializes object from JSON string.

    :param json_str: JSON string.
    :return: Deserialized object.
    """

    return orjson.loads(json_str)
