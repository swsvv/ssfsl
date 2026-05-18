import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from interface import UnSup
from util.fro_norm import get_fro_norm


class DINOv2(UnSup):
    def __init__(self, config: Config, encoder: nn.Module) -> None:
        super().__init__(config, encoder)
        self.student_encoder = encoder
        self.encoder = copy.deepcopy(encoder)

        self.proj = self._build_proj(**config.proj)
        self.teacher_proj = copy.deepcopy(self.proj)

        self.teacher_center = nn.Parameter(torch.zeros(1, self.proj_out_dim))
        self.teacher_center_temp = config.model.teacher_center_temp
        self.student_temp = config.model.student_temp

        self.momentum = config.model.momentum

        self.register_buffer("teacher_center_ema", torch.zeros(1, self.proj_out_dim))

    @torch.no_grad()
    def _update_teacher(self, teacher, student, m):
        for teacher_param, student_param in zip(
            teacher.parameters(), student.parameters()
        ):
            teacher_param.data.mul_(m).add_(student_param.data, alpha=1 - m)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        student_output = self.proj(self.student_encoder(x[0]))
        teacher_output = self.teacher_proj(self.encoder(x[1]))

        student_output = F.normalize(student_output, dim=-1, p=2)
        teacher_output = F.normalize(teacher_output, dim=-1, p=2)

        loss = self.dinov2_loss(student_output, teacher_output)

        self._update_teacher(self.encoder, self.student_encoder, self.momentum)
        self._update_teacher(self.teacher_proj, self.proj, self.momentum)
        self.update_center(teacher_output)

        corr_diff = get_fro_norm(teacher_output, student_output)

        return loss, corr_diff

    def dinov2_loss(self, student_output, teacher_output):
        student_scores = student_output / self.student_temp
        teacher_scores = (
            teacher_output - self.teacher_center
        ) / self.teacher_center_temp

        student_sm = F.log_softmax(student_scores, dim=-1)
        teacher_sm = F.softmax(teacher_scores, dim=-1)

        loss = -(teacher_sm * student_sm).sum(dim=-1).mean()
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.teacher_center_ema = self.teacher_center_ema * 0.9 + batch_center * 0.1
        self.teacher_center = nn.Parameter(self.teacher_center_ema.clone().detach())
