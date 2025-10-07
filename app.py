import argparse
import random
import tempfile
from datetime import datetime
from pathlib import Path

import requests
from omegaconf import OmegaConf
from tqdm import tqdm

# 假设核心推理函数位于 'scripts/inference.py' 中
try:
    from scripts.inference import main
except ImportError:
    print("错误：无法导入 'scripts.inference.main'。请确保您的运行环境可以找到该模块。")
    print("您可以尝试将项目根目录添加到 PYTHONPATH 中。")
    exit(1)


# --- 脚本配置 ---
CONFIG_PATH = Path("configs/unet/stage2.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")
ASSETS_PATH = Path("assets")

# --- 默认资源 ---
DEFAULT_VIDEO_URLS = [
    "https://github.com/anotherjesse/LatentSync/raw/main/assets/yuxin.mp4",
]
DEFAULT_AUDIO_URLS = [
    "https://github.com/anotherjesse/LatentSync/raw/main/assets/audio_yuxin.wav",
]


def download_file(url: str, directory: str) -> str | None:
    """从URL下载文件到指定目录，并显示进度条"""
    local_filename = Path(directory) / Path(url).name
    print(f"正在从 {url} 下载至 {local_filename}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get("content-length", 0))
            block_size = 1024

            with tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True, desc=local_filename.name) as progress_bar:
                with open(local_filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        progress_bar.update(len(chunk))
                        f.write(chunk)

            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("错误: 下载可能不完整")
                return None

    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        return None

    print("下载完成。")
    return str(local_filename)


if __name__ == "__main__":
    # --- 1. 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description="使用URL或本地文件进行LatentSync唇形同步")
    parser.add_argument("--input_video", type=str, default=None, help="输入视频的URL或本地路径 (.mp4)")
    parser.add_argument("--input_audio", type=str, default=None, help="输入音频的URL或本地路径 (.wav, .mp3)")
    
    # --- 主要改动 1 ---
    # 将输出目录的默认值改为 './output'
    parser.add_argument("--output_dir", type=str, default="./output", help="输出视频的保存目录")
    
    parser.add_argument("--guidance_scale", type=float, default=1.5, help="引导系数")
    parser.add_argument("--inference_steps", type=int, default=20, help="推理步数")
    parser.add_argument("--seed", type=int, default=1247, help="随机种子")
    cli_args = parser.parse_args()

    # --- 2. 如果未提供输入，则从 assets 目录随机选择 ---
    if cli_args.input_video is None:
        print(f"未指定 --input_video，尝试从 '{ASSETS_PATH}' 目录中随机选择...")
        video_files = list(ASSETS_PATH.glob("*.mp4"))
        if video_files:
            cli_args.input_video = str(random.choice(video_files))
            print(f"已选择视频: {cli_args.input_video}")
        else:
            print(f"警告: '{ASSETS_PATH}' 目录中未找到 .mp4 文件。将使用默认的示例URL。")
            cli_args.input_video = random.choice(DEFAULT_VIDEO_URLS)
            print(f"已选择视频URL: {cli_args.input_video}")

    if cli_args.input_audio is None:
        print(f"未指定 --input_audio，尝试从 '{ASSETS_PATH}' 目录中随机选择...")
        audio_files = list(ASSETS_PATH.glob("*.wav")) + list(ASSETS_PATH.glob("*.mp3"))
        if audio_files:
            cli_args.input_audio = str(random.choice(audio_files))
            print(f"已选择音频: {cli_args.input_audio}")
        else:
            print(f"警告: '{ASSETS_PATH}' 目录中未找到音频文件。将使用默认的示例URL。")
            cli_args.input_audio = random.choice(DEFAULT_AUDIO_URLS)
            print(f"已选择音频URL: {cli_args.input_audio}")

    # --- 3. 检查配置文件是否存在 ---
    if not CONFIG_PATH.exists() or not CHECKPOINT_PATH.exists():
        print(f"错误: 找不到必要的配置文件或模型检查点。")
        print(f"请确保 '{CONFIG_PATH}' 和 '{CHECKPOINT_PATH}' 存在。")
        exit(1)

    # --- 4. 创建临时目录并准备输入文件 ---
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"创建临时目录: {temp_dir}")
        local_video_path, local_audio_path = None, None

        if cli_args.input_video.startswith("http"):
            local_video_path = download_file(cli_args.input_video, temp_dir)
        else:
            local_video_path = cli_args.input_video
            print(f"使用本地视频文件: {local_video_path}")

        if cli_args.input_audio.startswith("http"):
            local_audio_path = download_file(cli_args.input_audio, temp_dir)
        else:
            local_audio_path = cli_args.input_audio
            print(f"使用本地音频文件: {local_audio_path}")

        if not (local_video_path and Path(local_video_path).exists()):
            print("错误：视频文件下载失败或路径无效，程序终止。")
            exit(1)
        if not (local_audio_path and Path(local_audio_path).exists()):
            print("错误：音频文件下载失败或路径无效，程序终止。")
            exit(1)

        # --- 5. 准备输出路径 ---
        output_dir = Path(cli_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # --- 主要改动 2 ---
        # 将输出路径固定为 'output/output.mp4'
        output_path = str(output_dir / "output.mp4")

        # --- 6. 加载并更新配置 ---
        config = OmegaConf.load(CONFIG_PATH)
        config["run"].update(
            {
                "guidance_scale": cli_args.guidance_scale,
                "inference_steps": cli_args.inference_steps,
            }
        )

        # --- 7. 构建传递给核心函数的参数 ---
        inference_args = argparse.Namespace(
            inference_ckpt_path=CHECKPOINT_PATH.absolute().as_posix(),
            video_path=local_video_path,
            audio_path=local_audio_path,
            video_out_path=output_path,
            inference_steps=cli_args.inference_steps,
            guidance_scale=cli_args.guidance_scale,
            seed=cli_args.seed,
            temp_dir=temp_dir,
            enable_deepcache=False,
        )

        # --- 8. 调用核心推理函数 ---
        try:
            print("\n模型配置完成，开始推理...")
            main(config=config, args=inference_args)
            print("=" * 50)
            print(f"推理成功完成！输出文件已保存至: {output_path}")
            print("=" * 50)
        except Exception as e:
            print(f"\n处理过程中发生错误: {e}")

    print("临时文件已清理。")
