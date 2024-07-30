# ConetentAdv_3D
This is a repo of "3D adversarial sample".

所用checkpoint: stablezero123


## Bug位置
zero123-main/zero123/my_inversion.py  其中的ddim_loop位置, 在ddim inversion中产生了偏差。
