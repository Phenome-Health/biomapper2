import copy
from typing import Dict, Any

from .core.annotator import annotate
from .core.normalizer import Normalizer
from .core.linker import link
from .core.resolver import resolve
from .utils import setup_logging

setup_logging()


def map_to_kg(entity: Dict[str, Any], stop_on_failure: bool = False) -> Dict[str, Any]:
    entity = copy.deepcopy(entity)  # Use a copy to avoid editing input item
    normalizer = Normalizer()  # Instantiate the ID normalizer (should only be done once, up front)

    # Perform all mapping steps
    annotate(entity)
    normalizer.normalize(entity, stop_on_failure=stop_on_failure)
    link(entity)
    resolve(entity)

    return entity
