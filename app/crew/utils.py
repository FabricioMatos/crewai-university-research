import json
from typing import Type, TypeVar

from pydantic import BaseModel


T = TypeVar('T', bound=BaseModel)


def str_to_model(data: str, model: Type[T]) -> T:
    data_dict = json.loads(data)
    return model.parse_obj(data_dict)
