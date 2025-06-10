# model_infer_single.py
"""
单张图片推理脚本
用法示例：
    python model_infer_single.py --image ./test/xxx.jpg --prompt "请描述图片中方块的数量、颜色、堆叠和相对位置"
"""

import argparse
from PIL import Image
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# 原版代码，对显存要求超过8G
# def load_model(model_name="Qwen/Qwen2.5-Omni-3B"):
#     """加载模型和处理器"""
#     print(f"正在加载模型 {model_name} ...")
#     model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#         model_name, device_map="auto", torch_dtype="auto"
#     )
#     processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
#     print("模型加载完成。")
#     return model, processor

def load_model(model_name="Qwen/Qwen2.5-Omni-3B"):
    """加载模型和处理器（使用 FP16 减少显存占用），并在运行脚本前添加环境变量，减少碎片"""
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 可选：减少显存碎片问题

    print(f"正在加载模型 {model_name} ...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    print("模型加载完成。")
    return model, processor


def run_inference(model, processor, image_path, prompt, max_new_tokens=128):
    """对单张图片和prompt进行推理，返回文本描述"""
    # 加载图片
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"图片加载失败: {e}")

    # 文本和图片一起处理
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )
    # 移动到模型设备
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 推理，关闭梯度计算以节省显存
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )
    # 解码输出
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return result

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni 单张图片推理脚本")
    parser.add_argument('--image', type=str, required=True, help='图片路径')
    parser.add_argument('--prompt', type=str, default="请描述图片中方块的数量、颜色、堆叠和相对位置", help='输入提示词')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-Omni-3B", help='模型名称或本地模型路径')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='生成最大 token 数')

    args = parser.parse_args()

    try:
        model, processor = load_model(args.model)
        print(f"推理图片: {args.image}")
        print(f"使用prompt: {args.prompt}")
        result = run_inference(model, processor, args.image, args.prompt, args.max_new_tokens)
        print("\n==== 推理结果 ====")
        print(result)
    except Exception as e:
        print(f"推理失败: {e}")

if __name__ == "__main__":
    main()

# 命令行调用示例：python model_infer_single.py --image ./image_test/BlueUp4.jpg --prompt "Please describe the number, color, stacking, and relative position of the blocks in the image."
