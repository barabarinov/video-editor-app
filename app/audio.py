import logging

import torch
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


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
    logging.info(
        f"Starting generate_audio with parameters:\n"
        f"  prompt: {prompt}\n"
        f"  num_inference_steps: {num_inference_steps}\n"
        f"  guidance: {guidance}\n"
        f"  negative_prompt: {negative_prompt}\n"
        f"  seed: {seed}\n"
        f"  width: {width}\n"
        f"  output_path: {output_path}\n"
        f"  checkpoint: {checkpoint}\n"
        f"  device: {device}\n"
        f"  scheduler: {scheduler}\n"
        f"  extension: {extension}"
    )
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
    logging.info("✅ Finished running run_txt2img")

    segment = streamlit_util.audio_segment_from_spectrogram_image(
        image=image,
        params=params,
        device=device,
    )
    logging.info("✅ Finished running audio_segment_from_spectrogram_image")

    segment.export(output_path, format=extension)

    streamlit_util.display_and_download_audio(
        segment, name=f"{prompt.replace(' ', '_')}_{seed}", extension=extension
    )
    logging.info(f"✅ Audio segment exported successfully to {output_path}")
