Python 3.12.4로 개발
250403: uv를 이용한 가상환경 구성 추가 - .python-version, pyproject.toml, uv.lock 파일 있으면 사용 가능
        uv관련 실행코드를 사용할려면 `pip3 install uv`로 설치할 것

# 아래 모든 코드들은 터미널에서 사용
## 모델을 이용하여 분석을 실행하는 코드
```powershell
python -m apply_trainer
```
```powershell
uv run apply_trainer.py
```

## 모델을 생성하는 코드
```powershell
python -m my_trainer
```
```powershell
uv run my_trainer.py
```

## 예전 모델을 신규 모델로 변경하는 코드  `.ckpt` 파일을 `.bak` 로 업그레이드 시켜주는 코드
```powershell
python -m pytorch_lightning.utilities.upgrade_checkpoint [.ckpt 파일 절대경로]
```

## 텐서보드 실행 코드
```powershell
tensorboard --logdir=grinding_v0_test
```