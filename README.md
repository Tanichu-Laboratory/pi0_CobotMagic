# pi0 for CobotMagic

## Introduction

このドキュメントは、CobotMagic で `pi0` / `pi05` を動かす方法と、ファインチューニング手順をまとめたものです。

- 現在、デプロイ可能な GPU サーバーは `ginkaku` のみです。
- CobotMagic 向けのメイン実装は `cobotmagic_deployment/` 配下に集約しています。

## Preparation

### 1. ginkaku で Docker コンテナを作成

`[container_name]` と `[your_mount_dir]` は環境に合わせて置き換えてください。

```bash
docker run -it --gpus all \
  --name [container_name] \
  --shm-size=32g \
  -v [your_mount_dir]:/workspace \
  pi0_bridge_env
```

### 注意

`ginkaku` の IP アドレスをコンテナ側に割り当てるため、コンテナは `root` で起動します。  
`root` で同時起動できるコンテナは 1 つのみのため、競合時はエラーになります。

### 2. githubからこのリポジトリをクローン
```bash
git clone --recurse-submodules \
https://github.com/Tanichu-Laboratory/pi0_CobotMagic.git
```

### 3. ROSのURI設定ファイルを作成する
```bash
cat > ~/.setup_ros.sh <<'EOF'
# > export ROS_MASTER_URI=http://10.228.162.34:11311
# > export ROS_IP=10.228.162.222
# > EOF
```
### 注意

ROS_MASTER_URIはCobotMagicのIPアドレスなので、使用時に変わっている可能性があります。その都度変更してください。

## Deployment pi0 on CobotMagic

### 1. CobotMagic 側

ROS を起動して、必要なノードを立ち上げます。

```bash
roscore
cd cobot_magic/Piper_ros_private-ros-noetic/
bash can_config.sh
# >>> agx
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
roslaunch astra_camera multi_camera.launch

# D405 を使う場合
roslaunch realsense2_camera rs_camera.launch camera:=camera_r serial_id:=218622277086
roslaunch realsense2_camera rs_camera.launch camera:=camera_l serial_id:=218622277131
```

### 2. GPU サーバー（ginkaku）側


トピックが見えていることを確認します。

```bash
rostopic list
```

`puppet_right`, `puppet_left` などが表示されれば OK です。  
タスクプロンプトは `cobotmagic/config.yaml` の先頭で指定します。

ターミナルを 2 つ用意して実行します。

#### ターミナル1（Python 3.8 / conda）

```bash
conda activate aloha
python cobotmagic/ros_bridge_node.py --config cobotmagic/config.yaml
```

#### ターミナル2（Python 3.11 / uv）

```bash
cd /workspace/project/openpi
source .venv/bin/activate
uv run python cobotmagic/policy_server.py --config cobotmagic/config.yaml
```

初期姿勢への移動が終わった後、任意キー入力で推論開始します。

## Fine-Tuning

### 1. 変換前データの確認
以下のように`data/dir/episode_*.hdf5` を配置します。
```text
/workspace/project
├── openpi/                         # このREADMEの作業ディレクトリ
│   └── scripts/
│       └── convert_local_mobile_aloha_to_lerobot.py
├── data/                           # 変換前データ（--data-root）
│   ├── <dir1>/episode_*.hdf5
│   └── <dir2>/episode_*.hdf5
└── buffer/                         # 変換後データ保存先（HF_LEROBOT_HOME）
    └── <repo-id>/                  # 例: mobile_aloha_test
```

### 2. uv 環境を有効化

```bash
cd /workspace/project/openpi
source .venv/bin/activate
```

### 3. データ変換（LeRobot 形式）

```bash
HF_LEROBOT_HOME=/workspace/project/buffer \
uv run python scripts/convert_local_mobile_aloha_to_lerobot.py \
  --data-root /workspace/project/data \
  --repo-id mobile_aloha_test
```

出力先: `/workspace/project/buffer/mobile_aloha_test`  
### 重要
タスクプロンプトは、変換時に`data/`に対して辞書を用いて割り振られます
`scripts/convert_local_mobile_aloha_to_lerobot.py`の冒頭でその辞書を定義する箇所があるので、使用するデータに合わせて変更してください。

### 4. 正規化統計を作成

```bash
HF_LEROBOT_HOME=/workspace/project/buffer \
uv run python scripts/compute_norm_stats.py --config-name pi0_mobile_aloha_local
```

出力先: `assets/pi0_mobile_aloha_local/<asset_id>`  
`asset_id` は設定値に合わせてください。

### 5. 学習開始

```bash
HF_LEROBOT_HOME=/workspace/project/buffer \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run python scripts/train.py pi0_mobile_aloha_local \
  --data.repo-id mobile_aloha_test \
  --exp-name mobile_aloha_lora \
  --overwrite
```

## Notes

- `repo-id` は `HF_LEROBOT_HOME` 配下のサブディレクトリ名として指定します。
- 変換・学習・norm stats の 3 ステップで同じ `repo-id` を使うことが重要です。
- `pi0_mobile_aloha_local` / `pi05_mobile_aloha_local` は LoRA 設定済みです。
