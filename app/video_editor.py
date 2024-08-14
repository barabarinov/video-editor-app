import logging
import shutil
import typing as T
from pathlib import Path

import moviepy.editor as mp
import streamlit as st
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.audio import generate_audio

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_video() -> None:
    st.title("Video Manipulation App")
    logging.info("Video Manipulation App started")

    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

    if video_file:
        video_path = save_uploaded_file(video_file)
        st.video(video_path)

        device = streamlit_util.select_device(st.sidebar)
        extension = streamlit_util.select_audio_extension(st.sidebar)
        checkpoint = streamlit_util.select_checkpoint(st.sidebar)

        num_clips = st.number_input(
            "Number of clips to split into", min_value=1, max_value=10, value=1
        )
        prompt = st.text_input("Enter a prompt for audio generation")

        selected_clip = st.selectbox(
            "Select clip to add generated audio", range(1, num_clips + 1)
        )
        num_columns = st.number_input(
            "Number of columns for displaying clips", min_value=1, max_value=5, value=3
        )
        audio_params = get_audio_params(device, extension, checkpoint)

        if st.button("Process"):
            process_and_download_clips(
                video_path=video_path,
                num_clips=num_clips,
                selected_clip=selected_clip,
                prompt=prompt,
                audio_params=audio_params,
                num_columns=num_columns,
            )


def save_uploaded_file(uploaded_file: UploadedFile) -> str:
    upload_dir = Path("uploads")
    recreate_directory(upload_dir)
    video_path = upload_dir / uploaded_file.name

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logging.info(f"✅ Uploaded video file saved to {video_path}")

    return video_path.as_posix()


def get_audio_params(device: str, extension: str, checkpoint: str) -> dict:
    with st.expander("Advanced Settings"):
        negative_prompt = st.text_input("Negative prompt")
        starting_seed = T.cast(
            int,
            st.number_input(
                "Seed",
                value=42,
                help="Change this to generate different variations",
            ),
        )
        num_inference_steps = T.cast(int, st.number_input("Inference steps", value=30))
        guidance = st.number_input(
            "Guidance",
            value=7.0,
            help="How much the model listens to the text prompt",
        )
        scheduler = st.selectbox(
            "Scheduler",
            options=streamlit_util.SCHEDULER_OPTIONS,
            index=0,
            help="Which diffusion scheduler to use",
        )
        assert scheduler is not None

        use_20k = st.checkbox("Use 20kHz", value=False)

        params = SpectrogramParams(
            min_frequency=10 if use_20k else 0,
            max_frequency=20000 if use_20k else 10000,
            sample_rate=44100,
            stereo=use_20k,
        )
    return {
        "negative_prompt": negative_prompt,
        "device": device,
        "extension": extension,
        "checkpoint": checkpoint,
        "seed": starting_seed,
        "num_inference_steps": num_inference_steps,
        "guidance": guidance,
        "scheduler": scheduler,
        "params": params,
    }


def process_and_download_clips(
    video_path: str,
    num_clips: int,
    selected_clip: int,
    prompt: str,
    audio_params: dict[str, any],
    num_columns: int,
) -> None:
    video = mp.VideoFileClip(video_path)
    clip_duration = video.duration / num_clips
    clips = [
        video.subclip(i * clip_duration, (i + 1) * clip_duration)
        for i in range(num_clips)
    ]

    audio_dir = Path("generated_audio")
    recreate_directory(audio_dir)
    audio_path = audio_dir / f"generated_audio.{audio_params['extension']}"

    if prompt:
        generate_audio(
            prompt=prompt,
            width=calculate_width(clip_duration),
            output_path=audio_path.as_posix(),
            **audio_params,
        )
    clips[selected_clip - 1] = clips[selected_clip - 1].set_audio(
        mp.AudioFileClip(audio_path.as_posix())
    )
    logging.info(f"✅ Added generated audio to clip {selected_clip}")

    output_dir = Path("output_clips")
    recreate_directory(output_dir)

    filename = get_video_file_name(video.filename)
    for i, clip in enumerate(clips):
        clip.write_videofile(
            (output_dir / f"{filename}_clip_{i + 1}.mp4").as_posix(), audio_codec="aac"
        )

    shutil.make_archive(base_name="clips", format="zip", root_dir=output_dir)
    logging.info("✅ Created zip archive of clips")
    st.success("Processing complete!")

    with open("clips.zip", "rb") as f:
        st.download_button(
            label="Download Clips",
            data=f,
            file_name="clips.zip",
            mime="application/zip",
        )

    clip_files = sorted(output_dir.iterdir(), key=lambda p: p.name)
    cols = st.columns(num_columns)

    for i, clip in enumerate(clip_files):
        with cols[i % num_columns]:
            st.video(clip.as_posix())


def get_video_file_name(filepath: str) -> str:
    return filepath.split("/")[1].split(".")[0]


def calculate_width(clip_duration: float) -> int:
    time_per_pixel = 512 / 44100
    width = int(clip_duration / time_per_pixel)
    return (width // 8) * 8


def recreate_directory(dir_path: Path) -> None:
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
