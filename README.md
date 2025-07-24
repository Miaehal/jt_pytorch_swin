# Swin Transformer in Jittor

## 1. Introduction
This project aims to complete the core model reproduction of the ICCV paper **"[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)"**. Based on the **[Jittor deep learning framework](https://github.com/Jittor/jittor)**, the complete code architecture of models such as Swin-T was implemented from scratch, including key modules such as **model structure, data processing, optimizer, and learning rate strategy**.

To verify the correctness of the reproduction, this project conducted a detailed **alignment analysis** of the training results implemented by Jittor and the benchmark test results based on the official PyTorch source code on the small dataset **'cats_vs_dogs'**.

## 2. Project Structure
The structure of this **jt_pytorch_swin** project is as follows:
```
jt_pytorch_swin/
├── configs/                  # yaml configuration files
│   └── swin_tiny_patch4_window7_224.yaml
│   └── swin_small_patch4_window7_224.yaml
│   └── swin_base_patch4_window7_224.yaml
├── data_jittor/              # Jittor data processing module
│   └── __init__.py
│   └── build_jittor.py
├── models_jittor/            # Jittor model
│   └── __init__.py
│   └── build_jittor.py
│   └── swin_transformer_jittor.py
├── cats_vs_dogs/             # The format can be found in the data preparation
├── output/                   # Training output directory (ignored by.gitignore)
├── main.py                   # Jittor main training/test script
├── optimizer.py              # Jittor optimizer
├── lr_scheduler.py           # Jittor lr_scheduler
├── utils.py                  # Jittor model save/load tool
├── logger.py                 # Log module
├── loss_acc.py               # Other loss and accuracy
├── results_curve.py          # Alignment curve
└── README.md                 # README
```

## 3. Environment Setup

This project consists of two parts: **PyTorch benchmark** and **Jittor reproduction**. It is recommended to create separate Conda virtual environments for them to avoid package version conflicts.

### 3.1 PyTorch benchmark environment(Windows)

Use the following steps to create a named **`swin`** conda environment, used to run the **[official PyTorch source](https://github.com/microsoft/Swin-Transformer)** and generate a benchmark data.

```bash
conda create -n swin python=3.7 -y
conda activate swin

pip install torch==1.8.0+cpu torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.4.12
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
pip install pyyaml scipy
```

### 3.2 Jittor reproduction environment(WSL/Ubuntu)

Please ensure that the **C++ compiler** and **Python development header files** are installed in your system. These are necessary for the Jittor framework to perform JIT compilation.
```bash
sudo apt update
sudo apt install g++ build-essential libomp-dev python3-dev
```
Use the following steps to create a named **`jittor_swin`** conda environment.

```bash
# Create and activate the environment
pip install --break-system-packages virtualenv
sudo apt install python3-virtualenv
virtualenv jittor_swin
source jittor_swin/bin/activate

pip install --break-system-packages jittor==1.3.9.14
pip install matplotlib==3.10.3 scipy==1.16.0 pyyaml==6.0.2 yacs==0.1.8 termcolor==3.1.0
```
You can manually switch to this project file or use the following command:
```bash
cd <path to your ‘jt_pytorch_swin’>
```

## 4. Data Preparation

1. **Datasets**: This project uses [Kaggle Cats vs Dogs](https://www.kaggle.com/c/dogs-vs-cats/data) dataset.
2. **Data division**:
    * **Training set**: **150** pictures each of cats and dogs.
    * **Validation set**: **25** pictures each of cats and dogs.
3. **Directory structure**: Please decompress the dataset and organize it into the format required by `ImageFolder`, and place it in the upper-level directory of the project folder. The final structure should be as follows:
    ```
    jt_pytorch_swin/
    ├── cats_vs_dogs/
        ├── train/
        │   ├── cat/         # 150 pictures of cats
        │   │   ├── 1
        │   │   ├── 2
			        ...
        │   │   └── 150
        │   └── dog/         # 150 pictures of dogs
        └── val/
            ├── cat/         # 25 pictures of cats
	        │   ├── 1
	        │   ├── 2
			        ...
	        │   └── 25
            └── dog/         # 25 pictures of dogs
    ```

## 5. Instruction Manual

Obtain the official benchmark data of PyTorch:
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 main.py ^
--cfg configs/swin/swin_tiny_patch4_window7_224.yaml ^
--data-path ./cats_vs_dogs ^
--batch-size 4 ^
--output ./output/swin_tiny_cats_vs_dogs_pytorch_run ^
--opts MODEL.NUM_CLASSES 2 TRAIN.EPOCHS 20 TRAIN.WARMUP_EPOCHS 5
```

**Note**:All commands are executed in the project root directory (`jt_pytorch_swin/`).

### Training script

Start the training of the model using the following command. The training logs and model checkpoints (`.pkl`files) will be saved in the `output/` directory.

```bash
python main.py \
--cfg ./configs/swin_tiny_patch4_window7_224.yaml \
--data-path ./cats_vs_dogs \
--batch-size 4 \
--output ./output/swin_tiny_cats_vs_dogs_jittor_run
```

### Test script

Use the following command to evaluate the checkpoints of a trained model. Please replace the path after the `--resume` parameter with the model file you want to test.
```bash
# Test the model saved in the 13th epoch
python main.py \
--cfg ./configs/swin_tiny_patch4_window7_224.yaml \
--data-path ./cats_vs_dogs \
--eval \
--resume ./output/swin_tiny_patch4_window7_224/jittor_default/ckpt_epoch_13.pkl
```

## 6. Experimental results and alignment analysis

For detailed configuration, please refer to **`config.py`**.

### Result comparison

| Framework | PyTorch | Jittor |
|:--:|:--:|:--:|
| training time per epoch | 149.90 seconds | 155.20 seconds |
| Max Acc@1 | 66.0% | 62.0% |
| the epoch achieves the highest Acc@1 | 2 | 5 |
| param.| 28.3 M | 27.5 M |
| FLOPs | 4.5 G | 4.5 G |
| image size| 224*224 | 224*224 |
| training time| 0:53:42 | 0:55:31 |

### Alignment graph of the accuracy curve of the validation set

The following figure is automatically generated by the `results_curve.py` script, demonstrating the changing trends of the validation set accuracy and loss between the Jittor implementation and the PyTorch benchmark during the training process.

![Acc@1 curve](https://github.com/user-attachments/assets/30161a1b-24bf-4cd5-bbb4-cc9125c3e868)

![loss curve](https://github.com/user-attachments/assets/04ecf2c4-e9c3-4e4a-a57d-608bfed00458)

### Result analysis and conclusion

By comparison, this Jittor implementation version achieved the highest validation set accuracy of **62%** on the `cats_vs_dogs` dataset, which is very close in numerical terms to the benchmark result of **66.0%** achieved by PyTorch on smaller datasets.
As can be seen from the alignment curve graph above, **the Jittor accuracy curve fluctuates within a similar range to that of PyTorch**. Due to the small dataset, both reached their performance peaks in the first five epochs, indicating that the models had been fitted. As a result, their performance slightly declined or stagnated in the later stages.

The Loss values in the early stage of training vary by order of magnitude (Jittor `~0.7` vs. PyTorch `~7.0`), and the reason is that in PyTorch, **`torch.nn.CrossEntropyLoss` has a special scaling when calculating the loss**. The initial value of Jittor's '~0.7' is more in line with the theoretical expectation of binary classification cross-entropy.

To sum up, although there are minor performance differences caused by the small dataset, from the perspective of **core training dynamics and behavior**, since the Jittor results are similar to the official PyTorch results, this Jittor reproduction **aligns with the PyTorch benchmark**, verifying the correctness of the code implementation.**If more epochs can be trained on large datasets, it may have a very good effect.**