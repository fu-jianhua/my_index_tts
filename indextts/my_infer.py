import os
import re
import sys
import time
import json
from subprocess import CalledProcessError
from typing import Dict, List, Optional

import torch
import torchaudio
from pydub import AudioSegment
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

import warnings
import requests
import librosa

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures

from indextts.utils.front import TextNormalizer, TextTokenizer

def crop_kada(wav: torch.Tensor, sr=24000, sec=0.1):
    """
    自动剪裁尾部咔哒声
    """
    cut_samples = int(sec * sr)
    trimmed_waveform = wav[..., :-cut_samples]
    
    return trimmed_waveform


def split_text(text):
    # 用特殊标记临时替换所有 break_"%ds" 形式的内容，防止它被分割
    protected = {}
    def protect(match):
        key = f"__BREAK_PLACEHOLDER_{len(protected)}__"
        protected[key] = match.group(0)
        return key

    text = re.sub(r'break_"[\d.]+s"', protect, text)

    # 使用中文和英文标点进行分割
    segments = re.split('([。！？，；：\n,.!?;:])', text)

    # 组合标点和句子
    result = []
    for i in range(0, len(segments)-1, 2):
        result.append(segments[i] + segments[i+1])
    # 处理最后一个没有标点的片段
    if len(segments) % 2 == 1:
        result.append(segments[-1])

    # 恢复 break_"%ds"
    def restore(segment):
        for key, val in protected.items():
            segment = segment.replace(key, val)
        return segment.strip()

    return [restore(seg) for seg in result if restore(seg)]

class IndexTTS:
    def __init__(
        self, cfg_path="checkpoints_v1.5/config.yaml", model_dir="checkpoints_v1.5", is_fp16=True, device=None, use_cuda_kernel=None,voice_db_path="voices"
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            is_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
            voice_db_path (str): 音色模型的保存路径, 默认为voices
        """
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.is_fp16 = False # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        # Comment-off to load the VQ-VAE model for debugging tokenizer
        #   https://github.com/index-tts/index-tts/issues/34
        #
        # from indextts.vqvae.xtts_dvae import DiscreteVAE
        # self.dvae = DiscreteVAE(**self.cfg.vqvae)
        # self.dvae_path = os.path.join(self.model_dir, self.cfg.dvae_checkpoint)
        # load_checkpoint(self.dvae, self.dvae_path)
        # self.dvae = self.dvae.to(self.device)
        # if self.is_fp16:
        #     self.dvae.eval().half()
        # else:
        #     self.dvae.eval()
        # print(">> vqvae weights restored from:", self.dvae_path)
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.is_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)
        if self.is_fp16:
            try:
                import deepspeed

                use_deepspeed = True
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(f">> DeepSpeed加载失败，回退到标准推理: {e}")
                print("See more details https://www.deepspeed.ai/tutorials/advanced-install/")

            self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=True)
        else:
            self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load as anti_alias_activation_loader
                anti_alias_activation_cuda = anti_alias_activation_loader.load()
                print(">> Preload custom CUDA kernel for BigVGAN", anti_alias_activation_cuda)
            except Exception as e:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.", e, file=sys.stderr)
                print(" Reinstall with `pip install -e . --no-deps --no-build-isolation` to prebuild `anti_alias_activation_cuda` kernel.", file=sys.stderr)
                print(
                    "See more details: https://github.com/index-tts/index-tts/issues/164#issuecomment-2903453206", file=sys.stderr
                )
                self.use_cuda_kernel = False
        self.bigvgan = Generator(self.cfg.bigvgan, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location="cpu")
        self.bigvgan.load_state_dict(vocoder_dict["generator"])
        self.bigvgan = self.bigvgan.to(self.device)
        # remove weight norm on eval mode
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", self.bigvgan_path)
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)
        # 缓存参考音频mel：
        self.cache_audio_prompt = None
        self.cache_cond_mel = None
        # 进度引用显示（可选）
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

        # 添加音色库相关属性
        self.voice_db_path = voice_db_path
        os.makedirs(voice_db_path, exist_ok=True)
        self.voice_index_file = os.path.join(voice_db_path, "voice_index.json")
        self.voice_index: Dict = self._load_voice_index()
        
    def _load_voice_index(self) -> Dict:
        """加载音色索引"""
        if os.path.exists(self.voice_index_file):
            with open(self.voice_index_file, 'r', encoding='utf8') as f:
                return json.load(f)
        return {}
    
    def _save_voice_index(self):
        """保存音色索引"""
        with open(self.voice_index_file, 'w', encoding='utf8') as f:
            json.dump(self.voice_index, f, ensure_ascii=False, indent=2)
     
    def save_voice(
            self,
            voice_name: str,
            audio_source: str
        ) -> Dict:
            """
            保存新音色
            Args:
                voice_name: 音色名称
                audio_source: 音频URL或本地路径
            Returns:
                音色信息字典
            """
            # 1. 处理音频来源
            if audio_source.startswith(('http://', 'https://')):
                # 下载URL音频
                audio_dir = os.path.join(self.voice_db_path, voice_name)
                os.makedirs(audio_dir, exist_ok=True)
                audio_path = os.path.join(audio_dir, "reference.wav")
                audio_path = self._download_audio(audio_source, audio_path)
            else:
                # 使用本地音频
                audio_path = audio_source

            # 2. 加载并处理音频
            audio, sr = torchaudio.load(audio_path)
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            
            # 3. 提取mel特征
            cond_mel = MelSpectrogramFeatures()(audio)
            
            # 4. 准备存储路径
            voice_dir = os.path.join(self.voice_db_path, voice_name)
            os.makedirs(voice_dir, exist_ok=True)
            
            # 5. 保存音色数据
            voice_data = {
                "name": voice_name,
                "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ref_audio": audio_path,
                "mel_path": os.path.join(voice_dir, f"{voice_name}.pt")
            }
            
            # 6. 保存mel特征
            torch.save(cond_mel, voice_data["mel_path"])
            
            # 7. 更新索引
            self.voice_index[voice_name] = voice_data
            self._save_voice_index()
            
            return voice_data

    def load_voice(self, voice_name: str) -> torch.Tensor:
        """加载音色的mel特征"""
        if voice_name not in self.voice_index:
            raise KeyError(f"Voice {voice_name} not found in database!")
            
        voice_data = self.voice_index[voice_name]
        mel_path = voice_data["mel_path"]
        
        if not os.path.exists(mel_path):
            raise FileNotFoundError(f"Mel feature file not found: {mel_path}")
            
        # 加载mel特征并移到正确的设备上
        cond_mel = torch.load(mel_path).to(self.device)
        return cond_mel

    def list_voices(self) -> Dict:
        """列出所有可用音色"""
        return self.voice_index

    def remove_voice(self, voice_name: str):
        """删除音色"""
        if voice_name not in self.voice_index:
            raise KeyError(f"Voice {voice_name} not found in database!")
            
        voice_data = self.voice_index[voice_name]
        voice_dir = os.path.dirname(voice_data["mel_path"])
        
        # 删除音色文件
        if os.path.exists(voice_dir):
            import shutil
            shutil.rmtree(voice_dir)
            
        # 更新索引
        del self.voice_index[voice_name]
        self._save_voice_index()  
    
            
    def _download_audio(self, url: str, save_path: str):
        """
        下载音频文件并转换为WAV格式
        Args:
            url: 音频文件URL
            save_path: 保存路径(应该以.wav结尾)
        """
        # 1. 先下载到临时MP3文件
        temp_mp3 = save_path.replace('.wav', '.mp3')
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(temp_mp3, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            # 2. 使用librosa读取MP3并转换格式
            y, sr = librosa.load(temp_mp3, sr=None)
            
            # 3. 保存为WAV格式
            wav_data = (y * 32767).astype(np.int16)
            wav_tensor = torch.tensor(wav_data)
            if len(wav_tensor.shape) == 1:
                wav_tensor = wav_tensor.unsqueeze(0)
            torchaudio.save(save_path, wav_tensor, sr)
            
            return save_path
            
        except Exception as e:
            print(f"下载或转换音频失败: {str(e)}")
            raise
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        Shrink special tokens (silent_token and stop_mel_token) in codes
        codes: [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                # code = code.cpu().tolist()
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[k] != self.stop_mel_token, f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                # new code
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # unchanged
            pass
        # clip codes to max length
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def bucket_sentences(self, sentences, bucket_max_size=4) -> List[List[Dict]]:
        """
        Sentence data bucketing.
        if ``bucket_max_size=1``, return all sentences in one bucket.
        """
        outputs: List[Dict] = []
        for idx, sent in enumerate(sentences):
            outputs.append({"idx": idx, "sent": sent, "len": len(sent)})
       
        if len(outputs) > bucket_max_size:
            # split sentences into buckets by sentence length
            buckets: List[List[Dict]] = []
            factor = 1.5
            last_bucket = None
            last_bucket_sent_len_median = 0

            for sent in sorted(outputs, key=lambda x: x["len"]):
                current_sent_len = sent["len"]
                if current_sent_len == 0:
                    print(">> skip empty sentence")
                    continue
                if last_bucket is None \
                        or current_sent_len >= int(last_bucket_sent_len_median * factor) \
                        or len(last_bucket) >= bucket_max_size:
                    # new bucket
                    buckets.append([sent])
                    last_bucket = buckets[-1]
                    last_bucket_sent_len_median = current_sent_len
                else:
                    # current bucket can hold more sentences
                    last_bucket.append(sent) # sorted
                    mid = len(last_bucket) // 2
                    last_bucket_sent_len_median = last_bucket[mid]["len"]
            last_bucket=None
            # merge all buckets with size 1
            out_buckets: List[List[Dict]] = []
            only_ones: List[Dict] = []
            for b in buckets:
                if len(b) == 1:
                    only_ones.append(b[0])
                else:
                    out_buckets.append(b)
            if len(only_ones) > 0:
                # merge into previous buckets if possible
                # print("only_ones:", [(o["idx"], o["len"]) for o in only_ones])
                for i in range(len(out_buckets)):
                    b = out_buckets[i]
                    if len(b) < bucket_max_size:
                        b.append(only_ones.pop(0))
                        if len(only_ones) == 0:
                            break
                # combined all remaining sized 1 buckets
                if len(only_ones) > 0:
                    out_buckets.extend([only_ones[i:i+bucket_max_size] for i in range(0, len(only_ones), bucket_max_size)])
            return out_buckets
        return [outputs]

    def pad_tokens_cat(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        if self.model_version and self.model_version >= 1.5:
            # 1.5版本以上，直接使用stop_text_token 右侧填充，填充到最大长度
            # [1, N] -> [N,]
            tokens = [t.squeeze(0) for t in tokens]
            return pad_sequence(tokens, batch_first=True, padding_value=self.cfg.gpt.stop_text_token, padding_side="right")
        max_len = max(t.size(1) for t in tokens)
        outputs = []
        for tensor in tokens:
            pad_len = max_len - tensor.size(1)
            if pad_len > 0:
                n = min(8, pad_len)
                tensor = torch.nn.functional.pad(tensor, (0, n), value=self.cfg.gpt.stop_text_token)
                tensor = torch.nn.functional.pad(tensor, (0, pad_len - n), value=self.cfg.gpt.start_text_token)
            tensor = tensor[:, :max_len]
            outputs.append(tensor)
        tokens = torch.cat(outputs, dim=0)
        return tokens

    def torch_empty_cache(self):
        try:
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
            elif "mps" in str(self.device):
                torch.mps.empty_cache()
        except Exception as e:
            pass

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    def synthesize(
        self,
        voice_name: str,
        text: str,
        output_path: Optional[str] = None,
        speed: float = 1.0,  
        pitch: float = 0.0,
        volume: float = 1.0,
        verbose: bool = False
    ):
        """
        使用保存的音色合成语音
        Args:
            voice_name: 音色名称
            text: 要合成的文本
            output_path: 输出音频路径
            speed: 语速(0.5-2.0)
            pitch: 音调调整(-12~12)
            volume: 音量(0.0-2.0) 
            verbose: 是否打印详细信息
        Returns:
            如果指定output_path则返回保存路径,否则返回(采样率,波形数据)元组
        """
        # 1. 加载音色mel特征
        cond_mel = self.load_voice(voice_name)
        
        # 2. 设置缓存
        self.cache_audio_prompt = voice_name
        self.cache_cond_mel = cond_mel
        text_segments = split_text(text)
        
        silent_time = 0.0
        timestamps = []         # 用于存储每段文本的时间戳
        current_time_ms = 0     # 当前时间戳（毫秒）
        silent_segments = []    # 用于收集每段 wav 的列表
        for i, segment in tqdm(enumerate(text_segments)):
            start_time_ms = current_time_ms
            # 3. 处理文本，处理静默时间
            parts = re.split(r'(break_"[\d\.]+s")', segment)
            sampling_rate = 24000
        
            for part in parts:
                m = re.match(r'break_"([\d\.]+)s"', part)
                if m:
                    # 3.1 静音段：生成 [1, num_samples] 的零张量
                    dur = float(m.group(1))
                    silent_time = dur
                    num_samples = int(dur * sampling_rate)
                    if verbose:
                        print(f">> 插入静音：{dur}s → {num_samples} 样本")
                    silence = torch.zeros(1, num_samples)
                    silent_segments.append(silence)
                else:
                    # 3.2 普通文字段：batch 推理逻辑不变，只输出二维 wav [1, T]
                    if not part.strip():
                        continue
                    if verbose:
                        print(f">> 合成文字段：\"{part}\"")
                        
                    wav_seg = self._synthesize_with_mel_fast(
                        cond_mel=cond_mel,
                        text=part,
                        cond_mel_lengths=torch.tensor([cond_mel.shape[-1]], device=self.device),
                        verbose=verbose,
                        output_path=output_path,
                        speed=speed,
                        pitch=pitch,
                        volume=volume
                    )
                    if wav_seg.dim() == 1:
                        wav_seg = wav_seg.unsqueeze(0)
                        
                    wav_seg = crop_kada(wav_seg, sr=sampling_rate)
                    
                    # # 保存每段
                    # torchaudio.save(f"{i}.wav", wav_seg.type(torch.int16), sampling_rate)
                    
                    silent_segments.append(wav_seg)
                    
            # 3.3 计算当前段的结束时间包含对应的静音段
            end_time_ms = current_time_ms + int(((wav_seg.shape[-1]/ sampling_rate) + silent_time) * 1000) 
            silent_time = 0.0  # 重置静音时间
            timestamps.append({
                "segment": segment,
                "start": start_time_ms,
                "end": end_time_ms
            })
            # 每句话换气，更新当前时间戳
            current_time_ms = end_time_ms + 80
            
        # wav audio outputs
        self._set_gr_progress(0.9, "save audio...")
        wav = torch.cat(silent_segments, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        
        # save audio
        wav = wav.cpu()  # to cpu
                
        # 使用 librosa 改变语速但不改变语调
        wav_np = wav.squeeze().cpu().numpy().astype(np.float32)
        
        wav_np = librosa.effects.preemphasis(wav_np, coef=0.97)
        
        # 1. 语速处理
        if speed != 1.0:
            wav_np = librosa.effects.time_stretch(y=wav_np, rate=speed)
        # 2. 语调处理
        if pitch != 0.0:
            wav_np = librosa.effects.pitch_shift(wav_np, sr=sampling_rate, n_steps=pitch)
        
        wav = torch.tensor(wav_np).unsqueeze(0)
        
   
        # 直接保存音频到指定路径中
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
        
        # 3. 音量处理
        if volume != 1.0:
            def map_volume_to_db(volume: float) -> float:
                if volume == 0:
                    return -100  # 静音（近似-60 dB）
                else:
                    # 1→0 dB（原音量），2→+6 dB（最大音量）
                    return (volume - 1) * 100
            audio = AudioSegment.from_file(output_path)
            audio = audio - map_volume_to_db(volume)  # volume 为分贝值（如 +6 表示增大6dB）
            audio.export(output_path, format="wav")
            
        print(">> wav file saved to:", output_path)
        return output_path, timestamps


    def _synthesize_with_mel_fast(self, cond_mel, text, cond_mel_lengths, verbose=False, output_path=None, speed=1.0, pitch=0.0, volume=1.0, max_text_tokens_per_sentence=100, sentences_bucket_max_size=4, **generation_kwargs):
        text_tokens_list = self.tokenizer.tokenize(text)

        sentences = self.tokenizer.split_sentences(text_tokens_list, max_tokens_per_sentence=max_text_tokens_per_sentence)
        if verbose:
            print(">> text token count:", len(text_tokens_list))
            print("   splited sentences count:", len(sentences))
            print("   max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")
            
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 1.0)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 600)
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0

        # text processing
        all_text_tokens: List[List[torch.Tensor]] = []
        self._set_gr_progress(0.1, "text processing...")
        bucket_max_size = sentences_bucket_max_size if self.device != "cpu" else 1
        all_sentences = self.bucket_sentences(sentences, bucket_max_size=bucket_max_size)
        bucket_count = len(all_sentences)
        if verbose:
            print(">> sentences bucket_count:", bucket_count,
                  "bucket sizes:", [(len(s), [t["idx"] for t in s]) for s in all_sentences],
                  "bucket_max_size:", bucket_max_size)
        for sentences in all_sentences:
            temp_tokens: List[torch.Tensor] = []
            all_text_tokens.append(temp_tokens)
            for item in sentences:
                sent = item["sent"]
                text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
                text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
                if verbose:
                    print(text_tokens)
                    print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                    # debug tokenizer
                    text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                    print("text_token_syms is same as sentence tokens", text_token_syms == sent) 
                temp_tokens.append(text_tokens)
        
            
        # Sequential processing of bucketing data
        all_batch_num = sum(len(s) for s in all_sentences)
        all_batch_codes = []
        processed_num = 0
        for item_tokens in all_text_tokens:
            batch_num = len(item_tokens)
            if batch_num > 1:
                batch_text_tokens = self.pad_tokens_cat(item_tokens)
            else:
                batch_text_tokens = item_tokens[0]
                
            auto_conditioning = torch.cat([cond_mel] * batch_num, dim=0)
            processed_num += batch_num
            # gpt speech
            self._set_gr_progress(0.2 + 0.3 * processed_num/all_batch_num, f"gpt inference speech... {processed_num}/{all_batch_num}")
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(batch_text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    temp_codes = self.gpt.inference_speech(auto_conditioning, batch_text_tokens,
                                        cond_mel_lengths=cond_mel_lengths,
                                        # text_lengths=text_len,
                                        do_sample=do_sample,
                                        top_p=top_p,
                                        top_k=top_k,
                                        temperature=temperature,
                                        num_return_sequences=autoregressive_batch_size,
                                        length_penalty=length_penalty,
                                        num_beams=num_beams,
                                        repetition_penalty=repetition_penalty,
                                        max_generate_length=max_mel_tokens,
                                        **generation_kwargs)
                    all_batch_codes.append(temp_codes)
            gpt_gen_time += time.perf_counter() - m_start_time

        # gpt latent
        self._set_gr_progress(0.5, "gpt inference latents...")
        all_idxs = []
        all_latents = []
        has_warned = False
        for batch_codes, batch_tokens, batch_sentences in zip(all_batch_codes, all_text_tokens, all_sentences):
            for i in range(batch_codes.shape[0]):
                codes = batch_codes[i]  # [x]
                if not has_warned and codes[-1] != self.stop_mel_token:
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True
                codes = codes.unsqueeze(0)  # [x] -> [1, x]
                if verbose:
                    print("codes:", codes.shape)
                    print(codes)
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                if verbose:
                    print("fix codes:", codes.shape)
                    print(codes)
                    print("code_lens:", code_lens)
                text_tokens = batch_tokens[i]
                all_idxs.append(batch_sentences[i]["idx"])
                m_start_time = time.perf_counter()
                with torch.no_grad():
                    with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                        latent = \
                            self.gpt(auto_conditioning, text_tokens,
                                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                        code_lens*self.gpt.mel_length_compression,
                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                        return_latent=True, clip_inputs=False)
                        gpt_forward_time += time.perf_counter() - m_start_time
                        all_latents.append(latent)
        del all_batch_codes, all_text_tokens, all_sentences
        # bigvgan chunk
        chunk_size = 2
        all_latents = [all_latents[all_idxs.index(i)] for i in range(len(all_latents))]
        if verbose:
            print(">> all_latents:", len(all_latents))
            print("  latents length:", [l.shape[1] for l in all_latents])
        chunk_latents = [all_latents[i : i + chunk_size] for i in range(0, len(all_latents), chunk_size)]
        chunk_length = len(chunk_latents)
        latent_length = len(all_latents)

        # bigvgan chunk decode
        self._set_gr_progress(0.7, "bigvgan decode...")
        tqdm_progress = tqdm(total=latent_length, desc="bigvgan")
        for items in chunk_latents:
            tqdm_progress.update(len(items))
            latent = torch.cat(items, dim=1)
            with torch.no_grad():
                with torch.amp.autocast(latent.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    m_start_time = time.perf_counter()
                    wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)
                    pass
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            wavs.append(wav.cpu()) # to cpu before saving
            
        # clear cache
        tqdm_progress.close()  # 确保进度条被关闭
        del all_latents, chunk_latents
        end_time = time.perf_counter()
        self.torch_empty_cache()
        
        # 在结尾添加音频处理和保存逻辑
        try:
            wav = torch.cat(wavs, dim=1)
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            wav = torch.cat(wavs, dim=0)
        return wav
        
if __name__ == "__main__":
    tts = IndexTTS()
    
    # tts.save_voice(
    #     "test_1",
    #     "https://nfc-links.oss-cn-chengdu.aliyuncs.com/virtuai/tianqizhenhao.wav"  # 音频url
    #     # "/root/autodl-tmp/project/index-tts/assets/lunwen.wav"  # 本地路径
    # )
    
    # tts.save_voice(
    #     "test_3",
    #     "/root/autodl-tmp/project/index-tts/assets/tmpv2ciwh1_.wav"  # 本地路径
    # )
    
    
    print(tts.synthesize(
        voice_name="test_1",
        text='党的十八大以来，习近平break_"1.5s"总书记高度重视网络文明建设。2025年中国网络文明大会举办之际，让我们一起重温总书记的“家园”之喻。',
        output_path="output1-1.wav",
    ))
    
    # print(tts.synthesize(
    #     voice_name="test_3",
    #     text='曾几何时，招呼人无须刻意措辞，一声“同志”便可，听者坦然、舒泰。后来，经济发展、文化多元，称呼也花样百出，“先生”“小姐”“老板”满天飞，开初尚觉新鲜，久之不免腻味，尤其是“小姐”这一称呼，常给人轻薄之感。如今，“小姐”似乎被“美女”取代，后者也逐渐失掉了赞美之意，变成了泛称。',
    #     output_path="output4-1.wav",volume=2
    # ))
    