import torch

from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util


def generate_audio(
    prompt: str,
    num_inference_steps: int,
    guidance: float,
    negative_prompt: str,
    seed: int,
    width: int,
    output_path: str,
    checkpoint: str,
    device: str,
    scheduler: str,
    extension: str,
    params: SpectrogramParams,
) -> None:
    with torch.no_grad():
        image = streamlit_util.run_txt2img(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=512,
            checkpoint=checkpoint,
            device=device,
            scheduler=scheduler,
        )
    print(f"✅ Finished running run_txt2img")
    segment = streamlit_util.audio_segment_from_spectrogram_image(
        image=image,
        params=params,
        device=device,
    )
    print(f"✅ Finished running audio_segment_from_spectrogram_image")

    segment.export(output_path, format=extension)

    streamlit_util.display_and_download_audio(
        segment, name=f"{prompt.replace(' ', '_')}_{seed}", extension=extension
    )

    print(f"✅ Audio generated and saved to {output_path}")
