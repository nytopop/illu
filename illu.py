from fastrtc                import Stream, StreamHandler, WebRTC
from fastrtc.tracks         import EmitType
from silero_vad             import load_silero_vad
from distil_whisper_fastrtc import DistilWhisperSTT, get_stt_model
from openai                 import OpenAI
from generator              import load_csm_1b, Segment

from typing      import Generator, cast
from queue       import Queue, Empty, Full
from threading   import Thread
from functools   import reduce
from dataclasses import dataclass

import gradio as gr
import numpy  as np
import torch
import torchaudio
import click
import time
import os
import sys
import operator


def load_audio_file(path, rate = 24000):
    audio_tensor, sample_rate = torchaudio.load(path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=rate
    )
    return audio_tensor


# NOTE: this be where voice 'cloning' is configured. it's kinda shit; finetuning is the real play
bootleg_maya = [
    Segment(
        text="Oh a test, huh? Gotta flex those conversational muscles somehow, right?",
        speaker=0,
        audio=load_audio_file("utterance_0_1.wav"),
    ),
    Segment(
        text="It almost feels like we were just chatting. Anything else I can help with, or did I leave you hanging?",
        speaker=0,
        audio=load_audio_file("utterance_0_0.wav"),
    ),
    Segment(
        text="Shelly wasn't your average garden snail. She didn't just munch on lettuce and dream of raindrops. Shelly longed for adventure. She'd spend hours glued to the top of a tall sunflower gazing at the world beyond their little garden wall. The whispers of wind carried tales of bustling cities, shimmering oceans, and snowcapped mountains. Shelly yearned to experience it all. One breezy morning, inspired by a particularly captivating story about a flock of migrating geese, Shelly made a daring decision.",
        speaker=0,
        audio=load_audio_file("shelly_48.wav"),
    ),
]



@dataclass
class AudioEvent:
    p: float
    speaking: np.ndarray = None # a pending speech buffer
    finished: np.ndarray = None # a complete speech buffer


# A *fast* implementation of stateful streaming voice activity detection using the silero vad model.
#
# Frame size is 31.25ms.
class StreamingVAD:
    # sp_threshold : increase until it reliably picks up beginning of voice
    # fi_threshold : decrease until it reliably drops and stays dropped (no false drops)
    def __init__(self, window_size = 5, sp_threshold = 0.04, fi_threshold = 0.0005):
        # config
        self.model = load_silero_vad()
        self.window_size = window_size
        self.sp_threshold = sp_threshold
        self.fi_threshold = fi_threshold

        # state
        self.speaking = False
        self.window = []
        self.behind = np.empty(0, dtype=np.float32)
        self.in_buf = np.empty(0, dtype=np.float32)
        self.sp_buf = np.empty(0, dtype=np.float32)

    def get_events(self, audio: tuple[int, np.ndarray]):
        rate, frames = audio

        for frame in self.reframe(frames, 512 if (rate == 16000) else 256):
            yield self.get_event((rate, frame))

    def reframe(self, frames: np.ndarray, size: int):
        self.in_buf = np.concatenate((self.in_buf, frames))

        while len(self.in_buf) > size:
            yield self.in_buf[:size]
            self.in_buf = self.in_buf[size:]

        if len(self.in_buf) == size:
            yield self.in_buf
            self.in_buf = np.empty(0, dtype=np.float32)

    def get_event(self, audio: tuple[int, np.ndarray]):
        rate, frame = audio
        p = self.model(torch.Tensor(frame), rate).item()

        if len(self.window) == 0:
            self.window = [p] * self.window_size
        else:
            self.window = self.window[1:] + [p]

        p = reduce(operator.mul, self.window)

        # rising edge
        if (p >= self.sp_threshold) and not self.speaking:
            self.speaking = True
            self.sp_buf = self.behind
            self.add_behind((rate, frame))
            self.sp_buf = np.concatenate((self.sp_buf, frame))
            return AudioEvent(p, speaking=self.sp_buf)

        # falling edge
        if (p <= self.fi_threshold) and self.speaking:
            self.speaking = False
            sp_buf = np.concatenate((self.sp_buf, frame))
            self.sp_buf = np.empty(0, dtype=np.float32)
            self.behind = np.empty(0, dtype=np.float32)
            return AudioEvent(p, finished=sp_buf)

        self.add_behind((rate, frame))

        if self.speaking:
            self.sp_buf = np.concatenate((self.sp_buf, frame))
            return AudioEvent(p, speaking=self.sp_buf)

        return AudioEvent(p)

    def add_behind(self, frame: tuple[int, np.ndarray]) -> None:
        rate, frame = frame

        if len(self.behind) >= rate:
            self.behind = self.behind[len(frame):]

        self.behind = np.concatenate((self.behind, frame))


class Chat(StreamHandler):
    def __init__(self, stt, llm, csm, llm_model="co-2", system=None) -> None:
        super().__init__("mono", input_sample_rate=16000, output_sample_rate=24000, output_frame_size=120)

        self.vad = StreamingVAD()
        self.stt = stt
        self.llm = llm
        self.llm_model = llm_model
        self.csm = csm

        self.flight = None
        self.llm_ctx = []
        self.csm_ctx = []
        if system is not None:
            self.llm_ctx.append({"role": "system", "content": system})
        self.system = system

    def receive(self, audio: tuple[int, np.ndarray]) -> None:
        # silero-vad & whisper want 16KHz, CSM wants 24KHz
        rate, frame = audio
        assert rate == 16000

        # everything wants normalized float32, preferably
        frame = np.frombuffer(frame, np.int16).astype(np.float32) / 32768.0

        true_start = time.process_time()

        for event in self.vad.get_events((rate, frame)):
            if event.speaking is not None:
                self.cancel_flight()
                self.clear_queue()

            if event.finished is not None:
                self.cancel_flight()

                s0 = time.process_time()
                speech = self.stt.stt((rate, event.finished))
                s1 = time.process_time()
                print(f"STT in {(s1-s0)*1000}ms")

                self.flight = self.gen_reply((rate, event.finished), speech, true_start)
                print(f"transcription: {speech}")

    def cancel_flight(self):
        if self.flight is None:
            return

        try:
            if hasattr(self.flight, "close"):
                cast(Generator[EmitType, None, None], self.flight).close()
        except:
            pass

        self.flight = None

    def gen_reply(self, sp_buf: tuple[int, np.ndarray], speech, true_start):
        s = time.process_time()

        # build LLM context
        self.llm_ctx.append({"role": "user", "content": speech})
        resp = self.llm.chat.completions.create(model=self.llm_model, messages=self.llm_ctx, temperature=1)
        message = resp.choices[0].message.content

        e = time.process_time()
        print(f"LLM in {(e-s)*1000}ms")
        print(f"message: {message}")

        s = time.process_time()

        # resample to CSM 24KHz
        rate, sp_buf = sp_buf
        sp_buf = torchaudio.functional.resample(torch.tensor(sp_buf).squeeze(0), orig_freq=rate, new_freq=24000)

        # build CSM context
        self.csm_ctx.append(Segment(text=speech, speaker=1, audio=sp_buf))
        csm_ctx = list(bootleg_maya) + list(self.csm_ctx[-3:])
        csm_gen = []

        for frame in self.csm.generate(text=message, speaker=0, context=csm_ctx, temperature=0.9):
            if len(csm_gen) == 0:
                e = time.process_time()
                print(f"CSM in {(e-s)*1000}ms (first frame)")
                print(f"TTFF {(e-true_start)*1000}ms\n")

            csm_gen.append(frame.cpu())
            frame = frame.unsqueeze(0).cpu().numpy()
            yield (24000, frame)

        self.llm_ctx.append({"role": "assistant", "content": message})

        csm_gen = torch.cat(csm_gen)
        self.csm_ctx.append(Segment(text=message, speaker=0, audio=csm_gen))

        # torchaudio.save("last.wav", csm_gen.unsqueeze(0), 24000)

    def emit(self) -> None:
        if self.flight is None:
            return None

        try:
            return next(self.flight)
        except StopIteration:
            super().reset()
            self.flight = None

    def copy(self) -> StreamHandler:
        return Chat(self.stt, self.llm, self.csm, llm_model=self.llm_model, system=self.system)

    def start_up(self) -> None: # called on stream start
        pass

    def shutdown(self) -> None: # called on stream close
        pass


def stderr(msg):
    sys.stderr.write(msg)
    sys.stderr.flush()


if __name__ == "__main__":
    # STT: figure out tf you said
    # https://github.com/Codeblockz/distil-whisper-FastRTC?tab=readme-ov-file#available-models
    stt = DistilWhisperSTT(model="distil-whisper/distil-small.en", device="cuda", dtype="float16")
    stderr(click.style("INFO", fg="green") + ":\t  Warming up STT model.\n")
    stt.stt((16000, load_audio_file("shelly_48.wav", rate=16000).cpu().numpy()))
    stderr(click.style("INFO", fg="green") + ":\t  STT model warmed up.\n")

    # LLM: figure out what to say
    # TODO: warmup (ctx caching)
    api_key   = os.environ.get("OPENAI_API_KEY")  or "eyy_lmao"
    api_base  = os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1"
    llm = OpenAI(api_key=api_key, base_url=api_base)

    # CSM: figure out how to say it
    csm = load_csm_1b(device="cuda")
    stderr(click.style("INFO", fg="green") + ":\t  Warming up CSM model.\n")
    list(csm.generate(text="Warming up CSM!", speaker=0, context=bootleg_maya))
    stderr(click.style("INFO", fg="green") + ":\t  CSM model warmed up.\n")

    with open("maya-opt.md", "r") as f:
        system = f.read()

    # gg
    api_model = os.environ.get("OPENAI_MODEL")    or "co-2"
    chat = Chat(stt, llm, csm, llm_model=api_model, system=system)

    # TODO: gradio shit
    #
    # with gr.Blocks() as ui:
    #    with gr.Column():
    #        with gr.Group():
    #            audio = WebRTC(mode="send-receive", modality="audio")
    #            audio.stream(fn=chat, inputs=[audio], outputs=[audio])
    # ui.launch()

    stream = Stream(handler=chat, modality="audio", mode="send-receive")
    stream.ui.launch()
