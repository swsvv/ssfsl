from transform.simclr_transform import SimCLRTransform
from transform.byol_transform import BYOLTransform
from transform.moco_transform import MoCoTransform
from transform.swav_transform import SwAVTransform
from transform.simsiam_transform import SimSiamTransform
from transform.barlowtwins_transform import BarlowTwinsTransform
from transform.vicreg_transform import VICRegTransform
from transform.wmse_transform import WMSETransform
from transform.dinov2_transform import DINOv2Transform
from transform.episodetest_transform import EpisodeTestTransform

TRANSFORM_REGISTRY: dict[str, type] = {
    "SimCLR": SimCLRTransform,
    "BYOL": BYOLTransform,
    "MoCo": MoCoTransform,
    "SwAV": SwAVTransform,
    "SimSiam": SimSiamTransform,
    "BarlowTwins": BarlowTwinsTransform,
    "VICReg": VICRegTransform,
    "WMSE": WMSETransform,
    "DINOv2": DINOv2Transform,
    "EpisodeTest": EpisodeTestTransform,
    "Baseline": EpisodeTestTransform,
}
