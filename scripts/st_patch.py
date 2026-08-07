"""Патч SadTalker: убираем RetinaFace-детектор (нужен вес detection_Resnet50_Final.pth,
недоступный из песочницы). Лицо на нашем аватаре известно и фронтальное —
возвращаем захардкоженные 68 landmarks в масштабе кадра."""
import os
import sys
import numpy as np
import torch

# только для CPU-машины и только один раз за процесс (защита от рекурсии)
if os.environ.get("RR_TL") != "1" and not torch.cuda.is_available():
    os.environ["RR_TL"] = "1"
    _tl0 = torch.load

    def _tlcpu(*a, **k):
        k.setdefault("map_location", "cpu")
        return _tl0(*a, **k)

    torch.load = _tlcpu

sys.path.insert(0, ".")
import src.utils.croper as _cr

LM68 = np.array([
    # jaw 0-16
    [215, 520], [222, 570], [232, 618], [246, 662], [268, 698], [296, 726],
    [330, 746], [358, 756], [384, 758], [410, 756], [438, 746], [472, 726],
    [500, 698], [522, 662], [536, 618], [546, 570], [553, 520],
    # brow right 17-21
    [283, 462], [298, 446], [318, 440], [338, 442], [355, 450],
    # brow left 22-26
    [413, 450], [430, 442], [450, 440], [470, 446], [485, 462],
    # nose 27-30
    [384, 470], [384, 502], [384, 532], [384, 552],
    # nostrils 31-35
    [346, 558], [364, 566], [384, 570], [404, 566], [422, 558],
    # eye right 36-41
    [288, 424], [301, 410], [318, 407], [334, 414], [331, 428], [315, 434],
    # eye left 42-47
    [456, 424], [443, 410], [426, 407], [410, 414], [413, 428], [429, 434],
    # mouth outer 48-59
    [296, 588], [330, 572], [360, 564], [384, 565], [408, 564], [438, 572],
    [452, 588], [438, 612], [408, 622], [384, 623], [360, 622], [330, 612],
    # mouth inner 60-67
    [322, 590], [352, 586], [384, 586], [414, 590], [414, 600], [384, 604],
    [352, 600], [322, 596],
], np.float32)


def _fake_get_landmark(self, img_np):
    H, W = img_np.shape[:2]
    lm = LM68.copy()
    lm[:, 0] *= W / 768.0
    lm[:, 1] *= H / 1376.0
    return lm


_cr.Preprocesser.get_landmark = _fake_get_landmark


# KeypointExtractor без RetinaFace (det_net не создаём — весов нет и не надо)
from src.face3d import extract_kp_videos_safe as _ek
from facexlib.alignment import init_alignment_model


def _init_no_det(self, device="cuda"):
    self.detector = init_alignment_model("awing_fan", device="cpu",
                                         model_rootpath="gfpgan/weights")
    self.det_net = None


_ek.KeypointExtractor.__init__ = _init_no_det


# extract_keypoint без det_net: фиксированный bbox лица (аватар известен)
import os as _os
from facexlib.alignment import landmark_98_to_68 as _lm98to68


def _one(self, image):
    img = np.array(image)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = int(W*0.18), int(H*0.10), int(W*0.82), int(H*0.62)
    crop = img[y0:y1, x0:x1]
    kp = _lm98to68(self.detector.get_landmarks(crop))
    kp[:, 0] += x0
    kp[:, 1] += y0
    return kp


def _fake_extract(self, images, name=None, info=True):
    if isinstance(images, list):
        kps = [self._one_im(im)[None] for im in images]
        kps = np.concatenate(kps, 0)
        if name:
            np.savetxt(_os.path.splitext(name)[0] + ".txt", kps.reshape(-1))
        return kps
    return self._one_im(images)


_ek.KeypointExtractor._one_im = _one
_ek.KeypointExtractor.extract_keypoint = _fake_extract


# POS иногда возвращает t/s не скалярами — приводим к скалярам
import src.face3d.util.preprocess as _pp
_orig_POS = _pp.POS


def _POS(xp, x):
    t, s = _orig_POS(xp, x)
    return np.asarray(t, dtype=np.float64).ravel(), np.float64(np.asarray(s).ravel()[0])


_pp.POS = _POS
print("st_patch: детектор обойдён (68 точек заданы), POS-скаляры исправлены")
