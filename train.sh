python scripts/train_smpl_cam.py --cfg configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml \
    --launcher pytorch --nThreads 16 --rank 0 --exp-id surfer-checked --dist-url tcp://127.0.1.1:7000
