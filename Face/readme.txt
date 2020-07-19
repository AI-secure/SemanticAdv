Test Environment:
pytorch 0.4.1

Default Test:
python verification_attack.py

For face-verfication model resnet101sofxmax,
it has threshold 1.244, 1.048, 0.597 for FPR 1e-3, 3e-4, 1e-4 respectively.

Optional Test:
python verification_attack.py --threshold 1.048
python verification_attack.py --threshold 0.597
python verification_attack.py --max_iteration 500 --lr 0.01 --tv_lambda 0.1 --threshold 0.597

Results:
All results will be stored into folder './results/'
It contains i_k_original_img.png(original image), i_k_target_img.png(target image) and i_k_True/False_adv_G(X,cj).png(adversarial image).
i: original image id, k: target image id, j: attribute index

Each attack record is written into './results/records.txt'

