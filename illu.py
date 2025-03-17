from fastrtc                import Stream, StreamHandler
from fastrtc.tracks         import EmitType
from silero_vad             import load_silero_vad, get_speech_timestamps
from distil_whisper_fastrtc import get_stt_model
from openai                 import OpenAI
from typing                 import Generator, cast
from generator              import load_csm_1b, Segment
from queue                  import Queue, Empty

# import gradio as gr
import numpy  as np
import torch
import torchaudio
import click
import time
import os
import sys

def load_audio_file(path):
    audio_tensor, sample_rate = torchaudio.load(path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=24000
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

class Chat(StreamHandler):
    def __init__(self, vad, stt, llm, csm, llm_model="co-2", system=None) -> None:
        # output at 50Hz or 20ms/frame for alignment with CSM and a reasonable amount of buffering
        super().__init__("mono", input_sample_rate=16000, output_sample_rate=24000, output_frame_size=240)

        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.llm_model = llm_model
        self.csm = csm

        self.paused_at = time.process_time()
        self.rx_buf = np.empty(0, dtype=np.float32)
        self.sp_buf = []
        self.speech = ""
        self.flight = None
        self.queued = Queue()

        self.llm_ctx = []
        self.csm_ctx = []
        if system is not None:
            self.llm_ctx.append({"role": "system", "content": system})

        self.system = system

    def receive(self, audio: tuple[int, np.ndarray]) -> None:
        # NOTE: silero-vad & whisper want 16KHz, CSM wants 24KHz, all want normalized float32
        rate, frame = audio
        assert rate == 16000
        frame = np.frombuffer(frame, np.int16).astype(np.float32) / 32768.0

        true_start = time.process_time()

        # are there any new speech chunks from running VAD over this frame and the past window?
        if self.buffer_frame_with_vad((rate, frame)):
            # re-transcribe the whole inflight speech buffer
            sp_buf = np.concatenate(self.sp_buf)
            try:
                speech = self.stt.stt((rate, sp_buf))
            except:
                speech = self.speech

            # does it change the transcription? some 'speech' sounds don't transcribe intelligibly
            if speech != self.speech:
                self.cancel_flight()
                self.speech = speech
                self.flight = self.gen_reply((rate, sp_buf), speech, true_start)
                self.queued = Queue()
                print(f"transcription: {speech}")

        # once we transition into the paused state, prune sp_buf and speech
        if len(self.sp_buf) > 0 and self.input_paused():
            self.llm_ctx.append({"role": "user", "content": self.speech})

            sp_buf = torch.tensor(np.concatenate(self.sp_buf)).squeeze(0)
            sp_buf = torchaudio.functional.resample(sp_buf, orig_freq=rate, new_freq=24000)
            self.csm_ctx.append(Segment(text=self.speech, speaker=1, audio=sp_buf))

            self.sp_buf = []
            self.speech = ""

    def cancel_flight(self):
        if self.flight is None:
            return

        try:
            if hasattr(self.flight, "close"):
                cast(Generator[EmitType, None, None], self.flight).close()
        except Exception as e:
            print(f"error closing generator: {e}")

        self.flight = None

    def gen_reply(self, sp_buf: tuple[int, np.ndarray], speech, true_start):
        # build LLM context
        llm_ctx = self.llm_ctx
        llm_ctx.append({"role": "user", "content": speech})

        s = time.process_time()

        resp = self.llm.chat.completions.create(model=self.llm_model, messages=llm_ctx, temperature=1.3)
        message = resp.choices[0].message.content

        e = time.process_time()
        print(f"LLM in {(e-s)*1000}ms")
        print(f"message: {message}")

        # resampling to CSM 24KHz
        rate, sp_buf = sp_buf
        sp_buf = torchaudio.functional.resample(torch.tensor(sp_buf).squeeze(0), orig_freq=rate, new_freq=24000)

        # build CSM context
        csm_ctx = bootleg_maya + self.csm_ctx[-7:]
        csm_ctx.append(Segment(text=speech, speaker=1, audio=sp_buf))
        csm_gen = []

        s = time.process_time()
        for frame in self.csm.generate(text=message, speaker=0, context=csm_ctx):
            if len(csm_gen) == 0:
                e = time.process_time()
                print(f"CSM in {(e-s)*1000}ms (first frame)")
                print(f"TTFF {(e-true_start)*1000}ms")

            csm_gen.append(frame)
            frame = frame.unsqueeze(0).cpu().numpy()
            yield (24000, frame)

        self.llm_ctx.append({"role": "assistant", "content": message})
        self.csm_ctx.append(Segment(text=message, speaker=0, audio=torch.cat(csm_gen)))

    def input_paused(self, threshold_ms: float = 600) -> bool:
        if self.paused_at is None:
            return False

        elapsed_ms = (time.process_time() - self.paused_at) * 1000

        return elapsed_ms > threshold_ms

    # returns True if a new speech chunk was buffered, and False otherwise
    def buffer_frame_with_vad(self, audio: tuple[int, np.ndarray]) -> bool:
        rate, frame = audio
        self.rx_buf = np.concatenate((self.rx_buf, np.squeeze(frame)))

        # min_silence_duration of 0 ensures we can discard the rest without problems
        ts = get_speech_timestamps(self.rx_buf, self.vad, min_speech_duration_ms=150, min_silence_duration_ms=0)

        if len(ts) > 0:
            # stopped speaking? flush and clear buffer
            if ts[-1]['end'] != len(self.rx_buf):
                for i in ts:
                    self.sp_buf.append(self.rx_buf[i['start'] : i['end']])
                self.rx_buf = np.empty(0, dtype=np.float32)
                self.paused_at = time.process_time()
                return True
            else:
                self.paused_at = None
        # prune a frame if silence is too long (last bit may be the start of some unrecognized speech)
        elif (len(self.rx_buf) / rate) >= 0.5:
            self.rx_buf = self.rx_buf[len(frame):]

        return False

    def emit(self) -> None:
        paused = self.input_paused()

        if paused:
            try:
                return self.queued.get_nowait()
            except Empty:
                pass

        if self.flight is not None:
            try:
                item = next(self.flight)
                if paused:
                    return item
                else:
                    self.queued.put(item)
            except StopIteration:
                super().reset()
                self.flight = None

    def copy(self) -> StreamHandler:
        return Chat(self.vad, self.stt, self.llm, self.csm, llm_model=self.llm_model, system=self.system)

    def start_up(self) -> None: # called on stream start
        pass

    def shutdown(self) -> None: # called on stream close
        pass

def stderr(msg):
    sys.stderr.write(msg)
    sys.stderr.flush()

if __name__ == "__main__":
    # VAD: figure out if you're talking
    vad = load_silero_vad()

    # STT: figure out tf you said
    # https://github.com/Codeblockz/distil-whisper-FastRTC?tab=readme-ov-file#available-models
    stt = get_stt_model("distil-whisper/distil-small.en", device="cuda", dtype="float16")

    # LLM: figure out what to say
    # TODO: warmup (ctx caching)
    api_key  = os.environ.get("OPENAI_API_KEY")  or "eyy_lmao"
    api_base = os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1"
    llm = OpenAI(api_key=api_key, base_url=api_base)

    # CSM: figure out how to say it
    csm = load_csm_1b(device="cuda")
    stderr(click.style("INFO", fg="green") + ":\t  Warming up CSM model.\n")
    list(csm.generate(text="Warming up CSM!", speaker=0, context=bootleg_maya))
    stderr(click.style("INFO", fg="green") + ":\t  CSM model warmed up.\n")

    with open("maya.md", "r") as f:
        system = f.read()

    # gg
    chat = Chat(vad, stt, llm, csm, llm_model="co-2", system=system)

    # TODO: gradio shit
    #
    # with gr.Blocks() as ui:
    #     with gr.Column():
    #         with gr.Group():
    #             audio = WebRTC(mode="send-receive", modality="audio")
    #             audio.stream(fn=chat, inputs=[audio], outputs=[audio])
    # ui.launch()

    stream = Stream(handler=chat, modality="audio", mode="send-receive")
    stream.ui.launch()
