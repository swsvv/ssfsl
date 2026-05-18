from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from config import Config


class EpisodeTest(nn.Module):
    def __init__(self, config: Config, encoder: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.classifier = config.classifier

        self.n_way = config.n_way
        self.num_spt = config.num_spt
        self.num_qry = config.num_qry

    def forward(self, x: Tensor, y: Tensor) -> float:
        feat_dim = x.shape[1:]
        x = x.reshape(self.n_way, self.num_spt + self.num_qry, *feat_dim)

        spt_x = x[:, : self.num_spt].reshape(self.n_way * self.num_spt, *feat_dim)
        qry_x = x[:, self.num_spt :].reshape(self.n_way * self.num_qry, *feat_dim)

        self.encoder.eval()
        with torch.no_grad():
            spt_z = F.normalize(self.encoder(spt_x))
            qry_z = F.normalize(self.encoder(qry_x))

        spt_z = spt_z.detach().cpu().numpy()
        qry_z = qry_z.detach().cpu().numpy()

        y = y.reshape(self.n_way, self.num_spt + self.num_qry)

        spt_y = y[:, : self.num_spt].reshape(-1)
        qry_y = y[:, self.num_spt :].reshape(-1)

        spt_y = spt_y.detach().cpu().numpy()
        qry_y = qry_y.detach().cpu().numpy()

        if self.classifier == "LogisticRegression":
            clf = LogisticRegression(
                penalty="l2",
                random_state=0,
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                multi_class="multinomial",
            )
        elif self.classifier == "SVM":
            clf = LinearSVC(C=1.0)
        else:
            raise ValueError("Unknown classifier!")

        clf.fit(spt_z, spt_y)
        pred = clf.predict(qry_z)

        acc = metrics.accuracy_score(qry_y, pred)

        return acc
