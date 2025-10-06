import torch.nn as nn
import torch


###------UTILISATION YOLO POUR DETECTION - A COMPLETER

### ---- ENTREE BATCH IMAGES N X 1 X 53 X 53
### ---  TARGET (N, 3, 5)
### --- MAX PARAMS : 400 000

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
            # garde le spatial (13x13)
            nn.Conv2d(32, 128, 3, padding=1), nn.ReLU(),
        )

        # --- Global pooling + shared representation ---
        self.pool = nn.AdaptiveAvgPool2d((7, 7))   # (N,128,7,7)
        self.flat = nn.Flatten(1)                  # (N,6272)
        self.shared = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
        )

        # --- Three separate heads ---
        self.obj_head = nn.Linear(256, num_anchors * 1)               # (N,3)
        self.box_head = nn.Linear(256, num_anchors * 3)               # (N,9)
        self.cls_head = nn.Linear(256, num_anchors * num_classes)     # (N,9)

    def forward(self, x):
        N = x.size(0)
        x = self.backbone(x)              # (N,128,13,13)
        x = self.pool(x)                  # (N,128,7,7)
        x = self.flat(x)                  # (N,6272)
        h = self.shared(x)                # (N,256)

        obj = self.obj_head(h).view(N, self.num_anchors, 1)
        box = self.box_head(h).view(N, self.num_anchors, 3)
        cls = self.cls_head(h).view(N, self.num_anchors, self.num_classes)

        # Pas de sigmoid ni softmax ici → pertes utilisent logits bruts
        # (N, num_anchors, 1+3+num_classes)
        return torch.cat([obj, box, cls], dim=-1)

class DetectionNet2(nn.Module):
    def __init__(self, in_ch=1, num_anchors=3, num_classes=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # --- Feature extractor (backbone) ---
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 53→26
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 26→13
            # garde le spatial (13x13)
            nn.Conv2d(16, 64, 3, padding=1), nn.ReLU(),
        )

        # --- Global pooling + shared representation ---
        self.pool = nn.AdaptiveAvgPool2d((7, 7))   # (N,128,7,7)
        self.flat = nn.Flatten(1)                  # (N,6272)
        self.shared = nn.Sequential(
            nn.Linear(64 * 7 * 7, 122),
            nn.ReLU(),
        )

        # --- Three separate heads ---
        self.obj_head = nn.Linear(122, num_anchors * 1)               # (N,3)
        self.box_head = nn.Linear(122, num_anchors * 3)               # (N,9)
        self.cls_head = nn.Linear(122, num_anchors * num_classes)     # (N,9)

    def forward(self, x):
        N = x.size(0)
        x = self.backbone(x)              # (N,128,13,13)
        x = self.pool(x)                  # (N,128,7,7)
        x = self.flat(x)                  # (N,6272)
        h = self.shared(x)                # (N,256)

        obj = self.obj_head(h).view(N, self.num_anchors, 1)
        box = self.box_head(h).view(N, self.num_anchors, 3)
        cls = self.cls_head(h).view(N, self.num_anchors, self.num_classes)

        return torch.cat([obj, box, cls], dim=-1)

class YOLO(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(YOLO, self).__init__()
        self.n_classes = n_classes
        self.num_anchors = 3

        # --- Feature extractor (backbone) ---
        self.conv_1 = nn.Conv2d(input_channels, 16, 4, stride=1, padding=1)
        self.relu_1 = nn.LeakyReLU()
        self.max1 = nn.MaxPool2d(2, stride=2)
        self.conv_2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.relu_2 = nn.LeakyReLU()
        self.max2 = nn.MaxPool2d(2, stride=2)
        self.conv_3 = nn.Conv2d(32, 16, 1, stride=1, padding=2)
        self.relu_3 = nn.LeakyReLU()
        self.conv_4 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.relu_4 = nn.LeakyReLU()
        self.conv_5 = nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.relu_5 = nn.LeakyReLU()
        self.conv_6 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu_6 = nn.LeakyReLU()
        self.max3 = nn.MaxPool2d(3, stride=2)
        self.conv_7 = nn.Conv2d(64, 32, 1, stride=1, padding=1)
        self.relu_7 = nn.LeakyReLU()
        self.conv_8 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu_8 = nn.LeakyReLU()
        self.conv_9 = nn.Conv2d(64, 32, 1, stride=1, padding=1)
        self.relu_9 = nn.LeakyReLU()
        self.conv_10 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu_10 = nn.LeakyReLU()
        self.conv_11 = nn.Conv2d(64, 32, 1, stride=1, padding=1)
        self.relu_11 = nn.LeakyReLU()
        self.conv_12 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.relu_12 = nn.LeakyReLU()
        self.conv_13 = nn.Conv2d(32, 16, 1, stride=1, padding=1)
        self.relu_13 = nn.LeakyReLU()
        self.conv_14 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.relu_14 = nn.LeakyReLU()
        self.conv_15 = nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.relu_15 = nn.LeakyReLU()
        self.conv_16 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu_16 = nn.LeakyReLU()
        self.max4 = nn.MaxPool2d(3, stride=2)
        self.conv_17 = nn.Conv2d(64, 32, 1, stride=1, padding=1)
        self.relu_17 = nn.LeakyReLU()
        self.conv_18 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu_18 = nn.LeakyReLU()
        self.conv_19 = nn.Conv2d(64, 32, 1, stride=1, padding=1)
        self.relu_19 = nn.LeakyReLU()
        self.conv_20 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu_20 = nn.LeakyReLU()
        self.conv_21 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.relu_21 = nn.LeakyReLU()
        self.conv_22 = nn.Conv2d(64, 48, 3, stride=1, padding=1)
        self.relu_22 = nn.LeakyReLU()
        self.conv_23 = nn.Conv2d(48, 48, 3, stride=1, padding=1)
        self.relu_23 = nn.LeakyReLU()
        self.conv_24 = nn.Conv2d(48, 128, 5, stride=1, padding=1)
        self.relu_24 = nn.LeakyReLU()
        self.conv_25 = nn.Conv2d(128, 128, 1, stride=1, padding=1)
        self.relu_25 = nn.LeakyReLU()

        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.flat = nn.Flatten(1)
        self.shared = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
        )

        self.obj_head = nn.Linear(256, self.num_anchors * 1)
        self.box_head = nn.Linear(256, self.num_anchors * 3)
        self.cls_head = nn.Linear(256, self.num_anchors * self.n_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.max1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.max2(x)
        x = self.conv_3(x)
        x = self.relu_3(x)
        x = self.conv_4(x)
        x = self.relu_4(x)
        x = self.conv_5(x)
        x = self.relu_5(x)
        x = self.conv_6(x)
        x = self.relu_6(x)
        x = self.max3(x)
        x = self.conv_7(x)
        x = self.relu_7(x)
        x = self.conv_8(x)
        x = self.relu_8(x)
        x = self.conv_9(x)
        x = self.relu_9(x)
        x = self.conv_10(x)
        x = self.relu_10(x)
        x = self.conv_11(x)
        x = self.relu_11(x)
        x = self.conv_12(x)
        x = self.relu_12(x)
        x = self.conv_13(x)
        x = self.relu_13(x)
        x = self.conv_14(x)
        x = self.relu_14(x)
        x = self.conv_15(x)
        x = self.relu_15(x)
        x = self.conv_16(x)
        x = self.relu_16(x)
        x = self.max4(x)
        x = self.conv_17(x)
        x = self.relu_17(x)
        x = self.conv_18(x)
        x = self.relu_18(x)
        x = self.conv_19(x)
        x = self.relu_19(x)
        x = self.conv_20(x)
        x = self.relu_20(x)
        x = self.conv_21(x)
        x = self.relu_21(x)
        x = self.conv_22(x)
        x = self.relu_22(x)
        x = self.conv_23(x)
        x = self.relu_23(x)
        x = self.conv_24(x)
        x = self.relu_24(x)
        x = self.conv_25(x)
        x = self.relu_25(x)

        x = self.pool(x)           # (N,128,7,7)
        x = self.flat(x)           # (N,6272)
        h = self.shared(x)         # (N,256)

        N = x.size(0)
        obj = self.obj_head(h).view(N, self.num_anchors, 1)                   # (N, 3, 1)
        box = self.box_head(h).view(N, self.num_anchors, 3)                    # (N, 3, 3)
        cls = self.cls_head(h).view(N, self.num_anchors, self.n_classes)       # (N, 3, n_classes)

        return torch.cat([obj, box, cls], dim=-1)

class TinyYOLO(nn.Module):  #Nx3x7
    def __init__(self, in_channels, n_classes, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.n_classes = n_classes
        self.output_dim = 7  # your 7 output features per anchor

        # Simplified backbone
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((7,7))
        )

        self.final_layer = nn.Linear(32 * 7 * 7, self.num_anchors * self.output_dim)

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        out = self.final_layer(feat)
        out = out.view(-1, self.num_anchors, self.output_dim)
        return out

class TinyYOLO2(nn.Module):  #Nx3x7
    def __init__(self, in_channels, n_classes, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.n_classes = n_classes
        self.output_dim = 7  # your 7 output features per anchor

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((7,7))
        )

        self.final_layer = nn.Linear(64 * 7 * 7, self.num_anchors * self.output_dim)

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        out = self.final_layer(feat)
        out = out.view(-1, self.num_anchors, self.output_dim)
        return out
    
class TinyYOLO3(nn.Module):
    def __init__(self, in_channels, n_classes, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.n_classes = n_classes
        self.output_dim = 7  # 7 output features per anchor

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # Shared fully connected layers for better capacity
        self.shared = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Final output layer with shape: N x num_anchors x 7
        self.final_layer = nn.Linear(64, self.num_anchors * self.output_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.shared(x)
        x = self.final_layer(x)
        x = x.view(-1, self.num_anchors, self.output_dim)
        return x


class LocalizationLoss(nn.Module): 
    def __init__(self):
        super(LocalizationLoss, self).__init__()

        alpha_obj = 1
        alpha_box = 50
        alpha_class = 1
        alpha_tot = alpha_obj + alpha_class + alpha_box

        self.alpha_obj = alpha_obj / (alpha_tot)
        self.alpha_box = alpha_box / (alpha_tot)
        self.alpha_class = alpha_class / (alpha_tot)

        ### Ajout en dehors d'inference
        self.L_obj = nn.BCEWithLogitsLoss()    ### Calcul de perte sur tous les instances pour detection d'objet ou pas
        self.Lbox = nn.SmoothL1Loss()          ### Calcul de perte instance avec objets position de boite
        self.Lclass = nn.CrossEntropyLoss()    ### Calcul de perte instance avec objets categorie de boite

    def forward(self, output, target):

        Output_Conf = output[:, :, 0]
        Output_boxes = output[:, :, 1:4]
        Output_class = output[:, :, 4:]

        target_conf = target[:, :, 0]
        target_boxes = target[:, :, 1:4]
        target_class = target[:, :, 4].long()      ## OU argmax(dim=2)??

        # Objet présent perte BCE
        loss_obj = self.L_obj(Output_Conf, target_conf.float())         ### Calcul de perte sur tous les objets pour detection d'objet ou pas

        ##definition d'un masque pour calcul de perte classe et box seulement pour instance avec object
        seuil = target_conf > 0.5

        # Positions boites perte MSE
        ### Calcul de perte instance avec objets position de boite
        if seuil.any():
            Output_boxes_norm = torch.sigmoid(Output_boxes)
            loss_boxes = self.Lbox(Output_boxes_norm[seuil], target_boxes[seuil])
        else:
            loss_boxes = torch.tensor(0.0, device=output.device)

        # Perte de Classe (CrossEntropy)
        ### Calcul de perte instance avec objets categorie de boite
        if seuil.any():
            loss_class = self.Lclass(Output_class[seuil], target_class[seuil])
        else:
            loss_class = torch.tensor(0.0, device=output.device)       

        print(f"Perte OBJ: {loss_obj.item()}, Perte Box: {loss_boxes.item()}, Perte Classe {loss_class.item()}")
        # Total loss
        loss = self.alpha_obj * loss_obj + self.alpha_box * loss_boxes + self.alpha_class * loss_class

        return loss


#model = DetectionNet(1,3,3)
#total_params = sum(p.numel() for p in model.parameters())
#print(f"Total parameters: {total_params}")

""" 
        :param boxes: Le tenseur PyTorch cible pour la tâche de détection:
            Dimensions: (N, 3, 5)
                Si un 1 est présent à (i, j, 0), le vecteur (i, j, 0:5) représente un objet.
                Si un 0 est présent à (i, j, 0), le vecteur (i, j, 0:5) ne représente aucun objet.
                Si le vecteur représente un objet (i, j, :):
                    (i, j, 1) est la position x centrale normalisée de l'objet j de l'image i.
                    (i, j, 2) est la position y centrale normalisée de l'objet j de l'image i.
                    (i, j, 3) est la largeur normalisée et la hauteur normalisée de l'objet j de l'image i.
                    (i, j, 4) est l'indice de la classe de l'objet j de l'image i.
"""