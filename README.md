Python 3.12.4로 개발

모델을 이용하여 분석을 실행하는 코드
python -m apply_trainer

모델을 생성하는 코드
python -m my_trainer

예전 모델을 신규 모델로 변경하는 코드  `.ckpt` 파일을 `.bak` 로 업그레이드 시켜주는 코드
python -m pytorch_lightning.utilities.upgrade_checkpoint [.ckpt 파일 절대경로]
