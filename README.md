# VQMotion

VQMotion 是一套以 **Hierarchical Residual VQ-VAE（HRVQ-VAE）** 表示人體動作，並使用 **MaskGIT / BERT-style masked-token prediction** 完成文字驅動動作生成與編輯的系統。

![VQMotion model architecture](assets/vqmotion_architecture.png)

## 功能概覽（Features）

- Text-to-Motion Generation：由文字產生新動作。
- Text-Driven Motion Editing：依照文字指令修改既有動作。
- Evaluation：評估 generation 與 editing 的品質、語意對齊和推論速度。
- Visualization：輸出 skeleton 或 SMPL mesh 影片。
- Application：透過瀏覽器操作生成、編輯與 3D 預覽。

## 文件導覽（Contents）

| 想完成的工作 | 前往章節 |
| --- | --- |
| 第一次建立環境與資料集 | [安裝與資料準備](#installation) |
| 訓練或接續訓練模型 | [模型訓練](#training) |
| 重現論文評估結果 | [模型評估](#evaluation) |
| 產生 skeleton / mesh 影片 | [動作視覺化](#visualization) |
| 啟動互動展示平台 | [互動展示平台](#application) |

> [!NOTE]
> 正式模型使用 `configs/train_vqmotion_hml.yaml`。其他 `train_vqmotion_hml_*.yaml` 為論文 ablation 設定。

---

<a id="installation"></a>

## 1. 安裝與資料準備（Installation）

本章只建立 Python 環境與 dataset。Checkpoint、evaluation model 和 SMPL 檔案會在實際使用它們的章節下載。

### 1.1 取得程式（Clone repository）

```bash
git clone <REPOSITORY_URL> VQMotion
cd VQMotion
```

### 1.2 建立環境（Create environment）

測試環境：

| Component | Version |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.4.1 |
| CUDA | 12.1 |

```bash
conda create -n vqmotion python=3.11 -y
conda activate vqmotion

pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

確認 PyTorch 與 CUDA：

```bash
python cuda_check.py
```

> [!NOTE]
> 本專案使用 PyTorch，不需要 TensorFlow、PR-VIPE 或 POEM。

### 1.3 準備資料集（Prepare dataset）

**下載：** [VQMotion dataset package（Google Drive）](https://drive.google.com/file/d/1hgv3IZdf40FHCJ_ihgX9qyyUie2srMNf/view?usp=drive_link)

解壓縮後，建議保持以下結構：

```text
<DATA_ROOT>/
└── HumanML3D/
    ├── train.txt
    ├── val.txt
    ├── test.txt
    ├── Mean.npy
    ├── Std.npy
    ├── new_joint_vecs/
    ├── new_joints/
    └── texts/
```

HumanML3D evaluator 使用的 GloVe 詞向量已包含在 repository 的 `glove/`，不需要另外下載。

> [!IMPORTANT]
> 在 [configs/train_vqmotion_hml.yaml](configs/train_vqmotion_hml.yaml) 設定 dataset 路徑。

```yaml
data:
  root_dir: '/path/to/DATA_ROOT'
  feat_dir: '/path/to/DATA_ROOT/HumanML3D/new_joint_vecs'
```

> [!WARNING]
> `data.root_dir` 必須指向包含 `HumanML3D/` 的上一層目錄，不是直接指向 `HumanML3D/`。

### 1.4 驗證安裝（Verify installation）

```bash
python -m py_compile model.py VQMotion_trainer.py Train_vqmotion.py
```

完成後，可依需求進入 Training、Evaluation、Visualization 或 Application 章節。

---

<a id="training"></a>

## 2. 模型訓練（Training）

Training 會同時使用 HumanML3D generation samples 與 MotionFix-style editing pairs，訓練同一個 VQMotion 模型完成生成和編輯。

### 2.1 本章所需檔案（Required files）

**下載：** [Train.zip（Google Drive）](https://drive.google.com/file/d/1pChGlbZCHVhBzjJ_ouw_EFYJjYohJSVR/view?usp=drive_link)

> [!IMPORTANT]
> `Train.zip` 的根目錄必須直接包含 `checkpoint_dir/` 與 `TMR/`。請在專案根目錄解壓。

```bash
unzip Train.zip -d .
```

解壓後的主要結構如下：

```text
VQMotion/
├── checkpoint_dir/humanml3d/
│   ├── vq/hml_hrvq_nq4_263_nb512_fk0_gl0_ex0.5_attn/
│   │   └── model/net_best_fid.tar
│   ├── Comp_v6_KLD005/
│   ├── text_mot_match/model/finest.tar
│   └── VQMotion/
│       ├── model/                      # 接續訓練與評估使用
│       └── train_vqmotion_hml.yaml
└── TMR/
    ├── models/tmr_humanml3d_guoh3dfeats/
    └── stats/humanml3d/
```

| Directory | 用途 | 從頭訓練是否需要 |
| --- | --- | --- |
| `checkpoint_dir/humanml3d/vq/...` | Frozen HRVQ-VAE tokenizer / decoder | 必要 |
| `Comp_v6_KLD005/` | HumanML3D evaluator 設定與統計資料 | 必要 |
| `text_mot_match/` | Generation validation evaluator | 必要 |
| `TMR/` | Editing validation evaluator 與 pretrained weights | 必要 |
| `VQMotion/model/` | 已訓練 checkpoint | 僅接續訓練需要 |
| `VQMotion/animation/`、`diagnose/`、`eval/` | 舊訓練輸出與紀錄 | 不需要額外設定 |

### 2.2 設定路徑（Configure paths）

> [!IMPORTANT]
> 請先修改 [configs/train_vqmotion_hml.yaml](configs/train_vqmotion_hml.yaml) 中的本機路徑。

| Field | 設定內容 |
| --- | --- |
| `exp.root_ckpt_dir` | `<PROJECT_ROOT>/checkpoint_dir` |
| `exp.root_log_dir` | `<PROJECT_ROOT>/checkpoint_dir/log` |
| `vq_cfg_dir` | `<PROJECT_ROOT>`，不是 `configs/` |
| `vq_name` | `residual_vqvae_hml.yaml` |
| `vq_ckpt` | HRVQ-VAE checkpoint 檔名 |
| `data.root_dir` | 包含 `HumanML3D/` 的 dataset 上層目錄 |
| `data.feat_dir` | `<DATA_ROOT>/HumanML3D/new_joint_vecs` |
| `tmr.root` | `<PROJECT_ROOT>/TMR` |
| `tmr.run_dir` | `<PROJECT_ROOT>/TMR/models/tmr_humanml3d_guoh3dfeats` |

同時將 [configs/residual_vqvae_hml.yaml](configs/residual_vqvae_hml.yaml) 的 `exp.root_ckpt_dir` 設為相同的 checkpoint 根目錄。其餘目錄保持上述標準結構時，不需移動或重新命名。

> [!WARNING]
> 接續訓練會讀取 `checkpoint_dir/humanml3d/VQMotion/train_vqmotion_hml.yaml`。換到新機器後，請同步更新該檔案中的 dataset、checkpoint、專案與 TMR 路徑。

### 2.3 從頭訓練（Start training）

```bash
python Train_vqmotion.py \
  --config ./configs/train_vqmotion_hml.yaml
```

訓練輸出：

```text
checkpoint_dir/humanml3d/VQMotion/model/
```

### 2.4 接續訓練（Resume training）

```bash
python Train_vqmotion.py \
  --config ./configs/train_vqmotion_hml.yaml \
  --OnGoing_model latest.tar
```

---

<a id="evaluation"></a>

## 3. 模型評估（Evaluation）

Evaluation 可分別評估 generation、editing，或一次執行兩者。

### 3.1 評估設定（Configure evaluation）

正式設定位於 [configs/evaluation_motion_editing_hml.yaml](configs/evaluation_motion_editing_hml.yaml)：

| Parameter | Value |
| --- | ---: |
| Sampling steps | 10 |
| Generation CFG | 4.0 |
| Editing CFG | 1.75 |
| Source CFG | 0.9 |
| Delta strength | 0.4 |
| Repeats | 5 |

### 3.2 執行評估（Run evaluation）

同時評估 generation 與 editing：

```bash
python -m evaluator.eval_MotionEditing_HumanML3D \
  --trans_cfg ./configs/train_vqmotion_hml.yaml \
  --eval_cfg ./configs/evaluation_motion_editing_hml.yaml \
  --ckpt best.tar
```

只評估 editing：

```bash
python -m evaluator.eval_MotionEditing_HumanML3D \
  --trans_cfg ./configs/train_vqmotion_hml.yaml \
  --eval_cfg ./configs/evaluation_motion_editing_hml.yaml \
  --ckpt best.tar \
  --eval_editing
```

### 3.3 評估指標（Reported metrics）

| Task | Metrics |
| --- | --- |
| Generation | R-Precision、Matching、FID、Diversity、AITS |
| Editing | G2T、G2S、TMR-FID、TMR-Diversity、AITS |

評估紀錄會寫入：

```text
checkpoint_dir/humanml3d/<STAGE_DIR>/eval/
```

---

<a id="visualization"></a>

## 4. 動作視覺化（Visualization）

Visualization 支援快速 skeleton 動畫，以及經 SMPL fitting 產生的 mesh 影片。

### 4.1 準備 SMPL 與設定

**下載：** [SMPL model package](https://drive.google.com/file/d/1o0q80GeV6zZjx5CUahLB-EwV25mYJaQ2/view?usp=drive_link)，解壓到 `visualize/smpl_models/`。

確認 SMPL 目錄至少包含：

```text
visualize/smpl_models/
├── gmm_08.pkl
├── neutral_smpl_mean_params.h5
├── smpl/SMPL_NEUTRAL.pkl
├── smpl/J_regressor_extra.npy
└── smplx/SMPLX_MALE.npz
```

> [!IMPORTANT]
> 請確認 [utils/visual_config.yaml](utils/visual_config.yaml) 的 SMPL 路徑。

```yaml
SMPL_MODEL_DIR: './visualize/smpl_models/'
```

接著在 [configs/visualize_motion_editing_hml.yaml](configs/visualize_motion_editing_hml.yaml) 設定：

- `trans_cfg`：VQMotion training config。
- `ckpt`：要載入的 checkpoint。
- `mode`：`gen`、`edit` 或 `all`。
- `num_samples`：輸出樣本數。
- `output_dir`：影片輸出目錄。

### 4.2 執行視覺化

**Mesh**

```bash
python -m utils.visualize_motion_editing_hml \
  --config ./configs/visualize_motion_editing_hml.yaml \
  --mode edit
```

**Skeleton only**

```bash
python -m utils.visualize_motion_editing_hml \
  --config ./configs/visualize_motion_editing_hml.yaml \
  --mode edit \
  --skeleton-only
```

結果預設輸出至 `visualize/motion_editing_demo/`。

---

<a id="application"></a>

## 5. 互動展示平台（Application）

Application 是 Flask + Three.js 的互動展示平台，可在瀏覽器執行文字生成、動作編輯與 3D mesh 預覽。

### 5.1 準備與啟動

**下載：** [SMPL model package](https://drive.google.com/file/d/1o0q80GeV6zZjx5CUahLB-EwV25mYJaQ2/view?usp=drive_link)，解壓到 `visualize/smpl_models/`。

> [!IMPORTANT]
> Application 預設使用 `MESH_MODE=own`，因此必須準備 `visualize/smpl_models/smplx/SMPLX_MALE.npz`，並確認 [utils/visual_config.yaml](utils/visual_config.yaml) 的 `SMPL_MODEL_DIR` 指向 `./visualize/smpl_models/`。若只使用 `MESH_MODE=joints`，則不需要 SMPL model package。

```bash
cd Application
CKPT=best.tar python server.py
```

> [!TIP]
> 啟動後開啟 `http://localhost:5000`。服務狀態可由 `http://localhost:5000/health` 確認。

### 5.2 執行選項（Runtime options）

| Environment variable | Default | 說明 |
| --- | --- | --- |
| `TRANS_CFG` | `configs/train_vqmotion_hml.yaml` | Training config |
| `CKPT` | `best.tar` | Checkpoint 名稱或絕對路徑 |
| `PORT` | `5000` | Server port |
| `USE_EMA` | `1` | 使用 checkpoint 的 EMA 權重 |
| `MESH_MODE` | `own` | `own`：SMPL-X mesh；`joints`：skeleton |
