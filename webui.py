import os
import shutil
import sys
import threading
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import gradio as gr
from indextts.utils.webui_utils import next_page, prev_page

from indextts.infer import IndexTTS
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="zh_CN")
MODE = 'local'
# tts = IndexTTS(model_dir="checkpoints",cfg_path="checkpoints/config.yaml")
tts = IndexTTS(model_dir="checkpoints_v1.5",cfg_path="checkpoints_v1.5/config.yaml")

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)


def gen_single(prompt, text, infer_mode, speed, pitch, volume, progress=gr.Progress()):
    """
    提交语音合成任务
    prompt: 参考音频
    text: 待合成的目标文本
    infer_mode: 推理模式，普通推理或批次推理
    speed: 语速倍率，float，大于1时加快，小于1时放慢
    pitch: 语调调整，半音数，正值提升，负值降低
    volume: 音量倍率，float，大于1时放大，小于1时减小
    """
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
    tts.gr_progress = progress
    if infer_mode == "普通推理":
        # output = tts.infer(prompt, text, output_path) # 普通推理
        output = tts.my_infer(prompt, text, output_path, speed, pitch, volume) # 普通推理
    else:
        # output = tts.infer_fast(prompt, text, output_path) # 批次推理
        output = tts.my_infer_fast(prompt, text, output_path, speed, pitch, volume) # 批次推理
    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button


with gr.Blocks() as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System</h2>
    <h2><center>(一款工业级可控且高效的零样本文本转语音系统)</h2>

<p align="center">
<a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>
    ''')
    with gr.Tab("音频生成"):
        with gr.Row():
            os.makedirs("prompts",exist_ok=True)
            prompt_audio = gr.Audio(label="请上传参考音频",key="prompt_audio",
                                    sources=["upload","microphone"],type="filepath")
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                input_text_single = gr.TextArea(label="请输入目标文本",key="input_text_single")
                infer_mode = gr.Radio(choices=["普通推理", "批次推理"], label="选择推理模式（批次推理：更适合长句，性能翻倍）",value="普通推理")
                
                # —— 新增三条 Slider —— #
                speed_slider = gr.Slider(
                    minimum=0.5, maximum=2.0, step=0.1, value=1.0,
                    label="语速"
                )
                pitch_slider = gr.Slider(
                    minimum=-12, maximum=12, step=1, value=0,
                    label="语调"
                )
                volume_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, step=0.1, value=1.0,
                    label="音量"
                )
                
                gen_button = gr.Button("生成语音",key="gen_button",interactive=True)
            output_audio = gr.Audio(label="生成结果", visible=True,key="output_audio")

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    gen_button.click(gen_single,
                     inputs=[prompt_audio, input_text_single, infer_mode, speed_slider, pitch_slider, volume_slider],
                     outputs=[output_audio])


if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name="127.0.0.1")
