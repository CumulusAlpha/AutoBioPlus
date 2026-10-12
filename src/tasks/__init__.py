from mjlab.utils.lab_api.tasks.importer import import_packages

from src.compat.mjlab_observation_refresh import apply_patch as _patch_mjlab_observation_refresh
from src.compat.mjlab_segmentation import apply_patch as _patch_mjlab_segmentation

_patch_mjlab_segmentation()
_patch_mjlab_observation_refresh()

_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)
