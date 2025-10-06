# detection_network.py
import torch
import torch.nn as nn


import torch
import torch.nn as nn


class DetectionNet(nn.Module):
    def __init__(self, in_ch=1, num_anchors=3, num_classes=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # --- Feature extractor (backbone) ---
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 53→26
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 26→13
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )

        # --- Global pooling + shared representation ---
        self.pool = nn.AdaptiveAvgPool2d((7, 7))   # (N,64,7,7)
        self.flat = nn.Flatten(1)                  # (N,3136)
        self.shared = nn.Sequential(
            nn.Linear(64 * 7 * 7, 96),
            nn.ReLU(),
        )

        # --- Three separate heads ---
        self.obj_head = nn.Linear(96, num_anchors * 1)               # (N,3)
        self.box_head = nn.Linear(96, num_anchors * 3)               # (N,9)
        self.cls_head = nn.Linear(96, num_anchors * num_classes)     # (N,9)

    def forward(self, x):
        N = x.size(0)
        x = self.backbone(x)              # (N,64,13,13)
        x = self.pool(x)                  # (N,64,7,7)
        x = self.flat(x)                  # (N,3136)
        h = self.shared(x)                # (N,96)

        obj = self.obj_head(h).view(N, self.num_anchors, 1)
        box = self.box_head(h).view(N, self.num_anchors, 3)
        cls = self.cls_head(h).view(N, self.num_anchors, self.num_classes)

        # (N, num_anchors, 1+3+num_classes)
        return torch.cat([obj, box, cls], dim=-1)


class SimpleDetLoss(nn.Module):
    """
    Perte composite simple pour nos cibles (N, A, 5):
      [:, :, 0] presence {0/1}
      [:, :, 1:4] x, y, size  in [0,1]
      [:, :, 4]   class_idx   in {0..C-1}
    """

    def __init__(self, w_obj=1.0, w_box=20.0, w_cls=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.SmoothL1Loss()
        self.ce = nn.CrossEntropyLoss()
        self.w_obj, self.w_box, self.w_cls = w_obj, w_box, w_cls

    def forward(self, pred, target):
        # Perdictions
        obj_logit = pred[..., 0]
        box_raw = pred[..., 1:4]
        cls_logit = pred[..., 4:]

        # Targets
        presence = target[..., 0]
        box_tgt = target[..., 1:4]
        cls_idx = target[..., 4].long()

        # 1) Objectness
        loss_obj = self.bce(obj_logit, presence)

        # Only for examples with objects
        pos = presence > 0.5
        if pos.any():
            # 2) Box (x,y,size)
            box_pred = torch.sigmoid(box_raw)
            loss_box = self.l1(box_pred[pos], box_tgt[pos])

            # 3) Class
            loss_cls = self.ce(cls_logit[pos], cls_idx[pos])
        else:
            loss_box = box_raw.sum() * 0.0
            loss_cls = cls_logit.sum() * 0.0

        print(
            f"loss_obj: {loss_obj.item():.4f}, loss_box: {loss_box.item():.4f}, loss_cls: {loss_cls.item():.4f}")

        return self.w_obj*loss_obj + self.w_box*loss_box + self.w_cls*loss_cls


def build_detection_model(in_channels=1, num_anchors=3, num_classes=3):
    return DetectionNet(in_ch=in_channels, num_anchors=num_anchors, num_classes=num_classes)


def build_detection_criterion():
    return SimpleDetLoss()
