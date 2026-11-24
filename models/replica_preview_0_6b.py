"""Replica Preview 0.6B — TokenAI Arabic TTS model.

Model: https://huggingface.co/tokenaii/Replica-Preview-0.6B
Uses the OmniVoice inference API with Egyptian Arabic as the target language.
"""

import modal
from models import BaseTTSModel, arena_cls, register_model
from app import LOCAL_MODULES

MODEL_REPO = "tokenaii/Replica-Preview-0.6B"
LANGUAGE = "arz"  # Egyptian Arabic (ISO 639-3)
VOICE_INSTRUCT = "female"


replica_preview_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy",
        "soundfile",
        "huggingface_hub",
        "omnivoice",
        "requests",
    )
    .run_commands(
        "python3 -c \""
        "from omnivoice import OmniVoice; "
        "import torch; "
        "model = OmniVoice.from_pretrained("
        f"    '{MODEL_REPO}',"
        "    device_map='cpu',"
        "    dtype=torch.float32,"
        "); "
        "print(f'Replica Preview downloaded, sampling_rate={model.sampling_rate}')"
        "\"",
        secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@arena_cls(
    image=replica_preview_image,
    gpu="A10G",
    scaledown_window=60,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class ReplicaPreview06BModel(BaseTTSModel):
    """TokenAI Replica Preview 0.6B for Egyptian Arabic synthesis."""

    model_id = "replica_preview_0_6b"
    display_name = "Replica Preview 0.6B"
    model_url = "https://huggingface.co/tokenaii/Replica-Preview-0.6B"
    gpu = "A10G"

    @modal.enter()
    def load_model(self):
        import torch
        from omnivoice import OmniVoice

        self.model = OmniVoice.from_pretrained(
            MODEL_REPO,
            device_map="cuda",
            dtype=torch.bfloat16,
        )
        self.sample_rate = self.model.sampling_rate
        print(
            f"✅ Replica Preview 0.6B loaded on CUDA "
            f"(sr={self.sample_rate}, language={LANGUAGE})"
        )

    @modal.method()
    def synthesize(self, text: str) -> dict:
        from omnivoice import OmniVoiceGenerationConfig

        gen_config = OmniVoiceGenerationConfig(
            num_step=32,
            guidance_scale=2.0,
            denoise=True,
            postprocess_output=True,
        )
        audios = self.model.generate(
            text=text,
            language=LANGUAGE,
            instruct=VOICE_INSTRUCT,
            generation_config=gen_config,
        )
        wav_np = audios[0].squeeze().cpu().numpy()
        result = self.success_response(
            self.audio_to_base64(wav_np, self.sample_rate),
            self.sample_rate,
        )
        result["voice_mode"] = "design"
        result["language"] = LANGUAGE
        return result
