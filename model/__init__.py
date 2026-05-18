from model.simclr import SimCLR
from model.byol import BYOL
from model.moco import MoCo
from model.swav import SwAV
from model.simsiam import SimSiam
from model.barlowtwins import BarlowTwins
from model.vicreg import VICReg
from model.wmse import WMSE
from model.dinov2 import DINOv2
from model.nnclr import NNCLR
from model.baseline import Baseline
from model.episodetest import EpisodeTest

MODEL_REGISTRY: dict[str, type] = {
    "SimCLR": SimCLR,
    "BYOL": BYOL,
    "MoCo": MoCo,
    "SwAV": SwAV,
    "SimSiam": SimSiam,
    "BarlowTwins": BarlowTwins,
    "VICReg": VICReg,
    "WMSE": WMSE,
    "DINOv2": DINOv2,
    "NNCLR": NNCLR,
    "Baseline": Baseline,
    "EpisodeTest": EpisodeTest,
}
