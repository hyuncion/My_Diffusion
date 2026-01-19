# My_Diffusion

PyTorch로 만든 **기본 DDPM(Diffusion Model)** 예제 코드입니다.

## Setup

```bash
cd My_Diffusion
pip install -r requirements.txt
```

## (Option) Toy Dataset 생성

외부 다운로드 없이 빠르게 파이프라인을 확인할 수 있도록 간단한 도형 이미지 데이터셋을 생성합니다.

```bash
python make_toy_dataset.py --out_dir data/toy --num_images 20000 --image_size 32
```

## Training

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

## Inference

```bash
python sample.py \
  --ckpt outputs/ddpm_toy32/checkpoints/last.pt \
  --num_samples 64 \
  --batch_size 64 \
  --out_dir outputs/ddpm_toy32/samples
```

## Evaluation

**1) Validation loss(MSE)**

```bash
python eval_loss.py \
  --ckpt outputs/ddpm_toy32/checkpoints/last.pt \
  --data_dir data/toy \
  --batch_size 128 \
  --num_batches 50
```

**2) Sampling speed/memory**

```bash
python benchmark.py \
  --ckpt outputs/ddpm_toy32/checkpoints/last.pt \
  --num_samples 64 \
  --batch_size 64 \
  --sample_steps 1000
```

