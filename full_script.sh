# # # # Add these lines to your full_script.sh after the train_ae command
# #!/bin/bash
# # # # Encode the training split
# set -e

python -m src.ae.train_ae --config config/ae_config.yaml

python -m src.ae.encode \
    --model data/models/ae/ae_best.pth \
    --manifest data/splits/train_split.csv \
    --out_file data/splits/train/encoder_feats.npy \
    --processed_dir data/processed \
    --config config/ae_config.yaml

python -m src.ae.encode \
    --model data/models/ae/ae_best.pth \
    --manifest data/splits/test_split.csv \
    --out_file data/splits/test/encoder_feats.npy \
    --processed_dir data/processed \
    --config config/ae_config.yaml

python -m src.ae.encode \
    --model data/models/ae/ae_best.pth \
    --manifest data/splits/val_split.csv \
    --out_file data/splits/val/encoder_feats.npy \
    --processed_dir data/processed \
    --config config/ae_config.yaml


# python -m src.emotion_discriminator.train_ed --config config/ed_config.yaml
# python -m src.gan.train_gan --config config/gan_config.yaml 

python -m src.gan.test_gan --emotion happy --samples 2
python -m src.gan.test_gan --emotion sad --samples 2
python -m src.gan.test_gan --emotion angry --samples 2
python -m src.gan.test_gan --emotion calm --samples 2
# python -m src.gan.analyze_midi generated_tests/test_happy_1.mid generated_tests/test_sad_1.mid generated_tests/test_angry_1.mid generated_tests/test_calm_1.mid


# python -m src.gan.diagnose




# python -m src.emotion_discriminator.train_ed --config config/ed_config.yaml
# python -m src.gan.train_gan --config config/gan_config.yaml --ed_config config/ed_config.yaml


