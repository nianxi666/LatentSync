import argparse
import random
import tempfile
from datetime import datetime
from pathlib import Path

import requests
from omegaconf import OmegaConf
from tqdm import tqdm

# 假设核心推理函数位于 'scripts/inference.py' 中
from scripts.inference import main

# --- 脚本配置 ---
CONFIG_PATH = Path("configs/unet/stage2_512.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")


def download_file(url: str, directory: str) -> str | None:
    """从URL下载文件到指定目录，并显示进度条"""
    local_filename = Path(directory) / Path(url).name
    print(f"正在从 {url} 下载至 {local_filename}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get("content-length", 0))
            block_size = 1024  # 1 KB

            with tqdm(
                total=total_size_in_bytes, unit="iB", unit_scale=True, desc=local_filename.name
            ) as progress_bar:
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
    parser = argparse.ArgumentParser(description="使用视频和音频进行LatentSync唇形同步")
    parser.add_argument("--input_video", type=str, required=True, help="输入视频的直接URL链接 (.mp4)")
    # --- 这里是修改的地方：input_audio 不再是必须的 ---
    parser.add_argument(
        "--input_audio", type=str, default=None, help="[可选] 输入音频的直接URL链接。如果留空，则从 'assets' 目录中随机选择一个。"
    )
    parser.add_argument("--output_dir", type=str, default="./temp", help="输出视频的保存目录")
    parser.add_argument("--guidance_scale", type=float, default=1.5, help="引导系数")
    parser.add_argument("--inference_steps", type=int, default=20, help="推理步数")
    parser.add_argument("--seed", type=int, default=1247, help="随机种子")
    cli_args = parser.parse_args()

    # --- 2. 检查配置文件是否存在 ---
    if not CONFIG_PATH.exists() or not CHECKPOINT_PATH.exists():
        print(f"错误: 找不到必要的配置文件或模型检查点。")
        print(f"请确保 '{CONFIG_PATH}' 和 '{CHECKPOINT_PATH}' 存在。")
        exit(1)

    # --- 3. 创建临时目录并处理文件 ---
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"创建临时目录: {temp_dir}")

        # 下载视频 (视频总是需要下载)
        local_video_path = download_file(cli_args.input_video, temp_dir)
        if not local_video_path:
            print("视频文件下载失败，程序终止。")
            exit(1)

        local_audio_path = None

        # --- 这里是新增的核心逻辑 ---
        if cli_args.input_audio:
            # 如果用户提供了音频URL，则下载它
            print("从提供的URL下载音频...")
            local_audio_path = download_file(cli_args.input_audio, temp_dir)
        else:
            # 如果用户未提供，则从 assets 目录随机选择
            print("未提供音频URL，正在从 'assets' 目录中随机选择...")
            assets_dir = Path("./assets")
            if not assets_dir.is_dir():
                print(f"错误: 'assets' 目录不存在，无法随机选择音频。请确保该目录存在。")
                exit(1)

            # 查找所有 .wav 和 .mp3 文件
            audio_files = list(assets_dir.glob("*.wav")) + list(assets_dir.glob("*.mp3"))
            if not audio_files:
                print(f"错误: 在 'assets' 目录中未找到任何 .wav 或 .mp3 文件。")
                exit(1)

            selected_audio_path = random.choice(audio_files)
            # 直接使用本地文件的绝对路径
            local_audio_path = str(selected_audio_path.absolute())
            print(f"已随机选择音频文件: {selected_audio_path.name}")

        if not local_audio_path:
            print("未能获取有效的音频文件路径，程序终止。")
            exit(1)

        # --- 4. 准备输出路径 ---
        output_dir = Path(cli_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_stem = Path(local_video_path).stem
        output_path = str(output_dir / f"{video_stem}_{current_time}.mp4")

        # --- 5. 加载并更新配置 ---
        config = OmegaConf.load(CONFIG_PATH)
        config["run"].update(
            {"guidance_scale": cli_args.guidance_scale, "inference_steps": cli_args.inference_steps}
        )

        # --- 6. 构建传递给核心函数的参数 ---
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

        # --- 7. 调用核心推理函数 ---
        try:
            print("\n模型配置完成，开始推理...")
            main(config=config, args=inference_args)
            print("=" * 50)
            print(f"推理成功完成！输出文件已保存至: {output_path}")
            print("=" * 50)
        except Exception as e:
            print(f"\n处理过程中发生错误: {e}")

    print("临时文件已清理。")
