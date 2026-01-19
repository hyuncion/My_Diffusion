# My_Diffusion

PyTorch로 만든 **기본 DDPM(Diffusion Model)** 예제 코드입니다.

- `torchvision` 없이 동작 (PIL 기반 로컬 이미지 폴더 데이터셋)
- 학습(`train.py`), 추론/샘플링(`sample.py`), 성능 측정(평가 `eval_loss.py`, 속도 `benchmark.py`) 포함

## 설치

```bash
cd My_Diffusion
pip install -r requirements.txt
```

## (Option) Toy Dataset 생성

외부 다운로드 없이 빠르게 파이프라인을 확인할 수 있도록 간단한 도형 이미지 데이터셋을 생성합니다.

```bash
python make_toy_dataset.py --out_dir data/toy --num_images 20000 --image_size 32
```

## 학습

로컬 이미지 폴더(`--data_dir`)를 사용합니다(하위 폴더 포함, jpg/png/webp/bmp 지원).

```bash
python train.py \
  --data_dir data/toy \
  --image_size 32 \
  --batch_size 128 \
  --timesteps 1000 \
  --epochs 50 \
  --out_dir outputs/ddpm_toy32
```

## 추론(샘플 생성)

```bash
python sample.py \
  --ckpt outputs/ddpm_toy32/checkpoints/last.pt \
  --num_samples 64 \
  --batch_size 64 \
  --out_dir outputs/ddpm_toy32/samples
```

## 성능 측정

**1) Validation loss(노이즈 예측 MSE)**

```bash
python eval_loss.py \
  --ckpt outputs/ddpm_toy32/checkpoints/last.pt \
  --data_dir data/toy \
  --batch_size 128 \
  --num_batches 50
```

**2) Sampling 속도/메모리**

```bash
python benchmark.py \
  --ckpt outputs/ddpm_toy32/checkpoints/last.pt \
  --num_samples 64 \
  --batch_size 64 \
  --sample_steps 1000
```

