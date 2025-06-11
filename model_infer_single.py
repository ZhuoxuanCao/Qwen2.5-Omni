# model_infer_single.py
"""
单张图片推理脚本 - 符合官方README要求的版本
用法示例：
    python model_infer_single.py --image ./test/xxx.jpg --prompt "请描述图片中方块的数量、颜色、堆叠和相对位置"
"""

import argparse
from PIL import Image
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import os
import gc


def load_model(model_name="./models--Qwen--Qwen2.5-Omni-3B/snapshots/f75b40e3da2003cdd6e1829b1f420ca70797c34e"):
    """加载本地模型和处理器（使用 FP16 减少显存占用）"""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

    print(f"正在加载模型 {model_name} ...")

    # 清理显存
    torch.cuda.empty_cache()
    gc.collect()

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    # 如果只需要文本输出，可以禁用talker以节省显存
    model.disable_talker()

    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_name,
        local_files_only=True
    )

    print("模型加载完成。")
    return model, processor


def run_inference(model, processor, image_path, prompt, max_new_tokens=10):
    """对单张图片和prompt进行推理，返回文本描述"""

    # 推理前清理显存
    torch.cuda.empty_cache()
    gc.collect()

    # 加载并处理图片
    try:
        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        # 限制图片大小
        max_size = 360
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"图片从 {original_size} 缩放至: {image.size}")

        # 保存处理后的图片（官方示例使用文件路径）
        temp_image_path = "temp_resized_image.jpg"
        image.save(temp_image_path)

    except Exception as e:
        raise ValueError(f"图片加载失败: {e}")

    # 构建对话格式 - 这是关键！
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text",
                 "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": temp_image_path},  # 使用文件路径
                {"type": "text", "text": prompt}
            ],
        },
    ]

    # 设置是否使用视频中的音频（对于图片设置为False）
    USE_AUDIO_IN_VIDEO = False

    # 应用聊天模板 - 官方要求的步骤
    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

    # 处理多模态信息 - 官方要求的步骤
    audios, images, videos = process_mm_info(
        conversation,
        use_audio_in_video=USE_AUDIO_IN_VIDEO
    )

    print(
        f"处理后的信息: audios={len(audios) if audios else 0}, images={len(images) if images else 0}, videos={len(videos) if videos else 0}")

    # 准备输入
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO
    )

    # 移动到设备
    inputs = inputs.to(model.device).to(model.dtype)

    print(f"输入形状: {inputs['input_ids'].shape}")

    # 生成参数
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": 1,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "pad_token_id": processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id else processor.tokenizer.eos_token_id,
        "use_cache": True,
        "use_audio_in_video": USE_AUDIO_IN_VIDEO,
        "return_audio": False,  # 只返回文本
    }

    # 推理
    print("开始生成...")
    with torch.no_grad():
        text_ids = model.generate(**inputs, **generation_kwargs)

    # 解码输出
    output_text = processor.batch_decode(
        text_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # 清理临时文件
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    # 清理内存
    del inputs, text_ids
    torch.cuda.empty_cache()
    gc.collect()

    return output_text


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni 单张图片推理脚本")
    parser.add_argument('--image', type=str, required=True, help='图片路径')
    parser.add_argument('--prompt', type=str, default="Please describe what you see in the image.", help='输入提示词')
    parser.add_argument('--model', type=str,
                        default="./models--Qwen--Qwen2.5-Omni-3B/snapshots/f75b40e3da2003cdd6e1829b1f420ca70797c34e",
                        help='模型名称或本地模型路径')
    parser.add_argument('--max_new_tokens', type=int, default=10, help='生成最大 token 数')

    args = parser.parse_args()

    try:
        # 加载模型
        model, processor = load_model(args.model)

        # 运行推理
        print(f"\n推理图片: {args.image}")
        print(f"使用prompt: {args.prompt}")
        print("-" * 50)

        result = run_inference(model, processor, args.image, args.prompt, args.max_new_tokens)

        print("\n==== 推理结果 ====")
        print(result)

    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# 命令行调用示例：
# python model_infer_single.py --image ./image_test/BlueUp4.jpg --prompt "This is the workspace of a robotic arm. Based on the image, describe the current scene and infer a reasonable sequence of actions to complete the task: Pick up the green cube."