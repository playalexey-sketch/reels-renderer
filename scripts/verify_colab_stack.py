#!/usr/bin/env python3
"""Проверка стека Colab-ноутбука ДО выдачи пользователю:
те же шимы, что в ячейках, + инференс SadTalker на коротком аудио."""
import os, sys, runpy, types
import numpy as np

# --- шимы numpy (как в FIX-ячейке ноутбука) ---
try:
    np.VisibleDeprecationWarning = np.exceptions.VisibleDeprecationWarning
except Exception:
    pass
for _a, _v in [('float', float), ('int', int), ('bool', bool), ('complex', complex), ('object', object)]:
    setattr(np, _a, _v)
import sys as _sys
import torchvision.transforms.functional as _F
if 'torchvision.transforms.functional_tensor' not in _sys.modules:
    _mt = types.ModuleType('torchvision.transforms.functional_tensor')
    _mt.rgb_to_grayscale = _F.rgb_to_grayscale
    _sys.modules['torchvision.transforms.functional_tensor'] = _mt
# torch.load НЕ патчим: на torch 2.1.2 weights_only по умолчанию False
sys.path.insert(0, os.path.expanduser("~/third_party/SadTalker"))
# в песочнице вес детектора недоступен — обход (в Colab он скачается сам, патч не нужен)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st_patch  # noqa

ST = os.path.expanduser("~/third_party/SadTalker")
audio = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT := os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "render", "voice_test.wav")
res = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ST, "results_test")

sys.argv = ['inference.py', '--driven_audio', audio,
            '--source_image', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "render", "avatar_closed.jpg"),
            '--checkpoint_dir', 'checkpoints', '--result_dir', res,
            '--preprocess', 'full', '--enhancer', 'none',
            '--cpu', '--batch_size', '1']
runpy.run_path(os.path.join(ST, 'inference.py'), run_name='__main__')
print("ПРОВЕРКА ЗАВЕРШЕНА, результат:", res)
