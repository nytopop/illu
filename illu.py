# impl notes in no particular order
#
# performance:
# - stream transcription instead of sequentially batching pause detection into transcription ops
#   then generate immediately on each transcribed word, but only yield the stream upon sufficient pause
#   improves TTFF significantly, as we can utilize ~600ms pause duration as part of our gen budget
# - CSM: torch compile
# - CSM: supposedly kv caches for backbone and decoder are on CPU (???)
# - latency is the enemy; kill it with fire
#
# processes:
# - silence-fill: treat silence as tacit permission to steer conversation arbitrarily
# - profiler: active/passive probes for interests, relationships, likes, dislikes, etc
#   in short, it should actively try to get to know you, and store that information for richer context
from fastrtc import Stream, StreamHandler
from queue   import Queue, Empty

from silero_vad             import load_silero_vad, get_speech_timestamps
from distil_whisper_fastrtc import get_stt_model
from openai                 import OpenAI
from generator              import load_csm_1b

import numpy  as np
import gradio as gr
import time
import os

class Chat(StreamHandler):
    def __init__(self, vad, stt, llm, csm) -> None:
        # output at 12.5Hz or 80ms/frame for alignment with CSM (let's avoid any unnecessary buffering)
        super().__init__("mono", input_sample_rate=16000, output_sample_rate=24000, output_frame_size=1920)

        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.csm = csm

        self.paused_at = time.process_time()
        self.rx_buf = np.empty(0, dtype=np.float32)
        self.sp_buf = []
        self.speech = ""
        self.queued = Queue()

    def receive(self, audio: tuple[int, np.ndarray]) -> None:
        # NOTE: silero-vad & whisper want 16KHz, CSM wants 24KHz, all want normalized float32
        rate, frame = audio
        assert rate == 16000
        frame = np.frombuffer(frame, np.int16).astype(np.float32) / 32768.0

        # are there any new speech chunks from running VAD over this frame and the past window?
        if self.buffer_frame_with_vad((rate, frame)):
            # re-transcribe the whole inflight speech buffer
            speech = self.stt.stt((rate, np.concatenate(self.sp_buf)))

            # does it change the transcription? some 'speech' sounds don't transcribe intelligibly
            if speech != self.speech:
                # TODO: actually do something with this
                self.speech = speech
                print(f"transcription: {speech}")
            else:
                # TODO: idk, we probably don't need to do anything in this case
                print("NOTHING CHANGED, U GOT SCAMMED")

        match self.input_status():
            case "speaking":
                # we don't need to do anything in this case
                print("still speaking")

            case "pausing", after:
                # informational?
                print(f"pausing after {after}ms")

            case "paused":
                # TODO: if we're not already yielding response frames, start now
                pass

    def input_status(self, threshold_ms: float = 600) -> str:
        if self.paused_at is None:
            return "speaking"

        elapsed_ms = (time.process_time() - self.paused_at) * 1000

        if elapsed_ms > threshold_ms:
            return "paused"

        return "pausing", threshold_ms - elapsed_ms

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
        try:
            return self.queued.get_nowait()
        except Empty:
            return None

    def copy(self) -> StreamHandler:
        return Chat(self.vad, self.stt, self.llm, self.csm)

    def start_up(self) -> None: # called on stream start
        pass

    def shutdown(self) -> None: # called on stream close
        pass

if __name__ == "__main__":
    # VAD: figure out if you're talking
    vad = load_silero_vad()

    # STT: figure out tf you said
    # https://github.com/Codeblockz/distil-whisper-FastRTC?tab=readme-ov-file#available-models
    stt = get_stt_model("distil-whisper/distil-small.en", device="cuda", dtype="float16")

    # LLM: figure out what to say
    api_key  = os.environ.get("OPENAI_API_KEY")  or "eyy_lmao"
    api_base = os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1"
    llm = OpenAI(api_key=api_key, base_url=api_base)

    # CSM: figure out how to say it
    csm = load_csm_1b(device="cuda")

    # gg
    chat = Chat(vad, stt, llm, csm)

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
