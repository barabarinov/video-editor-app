import os
import shutil
import typing as T

import moviepy.editor as mp
import streamlit as st
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util

from app.audio import generate_audio

AUDIO_OPTIONS_TITLE = "Audio generation options"


def recreate_directory(dir_path: str) -> None:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    os.makedirs(dir_path)


def process_video() -> None:
    st.title("Video Manipulation App")

    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

    if video_file:
        video_path = os.path.join("uploads", video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        st.video(video_path)

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
        with st.expander(AUDIO_OPTIONS_TITLE):
            st.subheader(f"{AUDIO_OPTIONS_TITLE}:")
            negative_prompt = st.text_input("Negative prompt")
            device = streamlit_util.select_device(st.sidebar)
            extension = streamlit_util.select_audio_extension(st.sidebar)
            checkpoint = streamlit_util.select_checkpoint(st.sidebar)
            starting_seed = T.cast(
                int,
                st.number_input(
                    "Seed",
                    value=42,
                    help="Change this to generate different variations",
                ),
            )
            num_inference_steps = T.cast(int, st.number_input("Inference steps", value=30))
            width = T.cast(int, st.number_input("Width", value=512))
            guidance = st.number_input(
                "Guidance", value=7.0, help="How much the model listens to the text prompt"
            )
            scheduler = st.selectbox(
                "Scheduler",
                options=streamlit_util.SCHEDULER_OPTIONS,
                index=0,
                help="Which diffusion scheduler to use",
            )
            assert scheduler is not None

            use_20k = st.checkbox("Use 20kHz", value=False)

        if use_20k:
            params = SpectrogramParams(
                min_frequency=10,
                max_frequency=20000,
                sample_rate=44100,
                stereo=True,
            )
        else:
            params = SpectrogramParams(
                min_frequency=0,
                max_frequency=10000,
                stereo=False,
            )

        if st.button("Process"):
            video = mp.VideoFileClip(video_path)
            duration = video.duration
            clip_duration = duration / num_clips
            clips = [
                video.subclip(i * clip_duration, (i + 1) * clip_duration)
                for i in range(num_clips)
            ]

            audio_dir = "generated_audio"
            audio_path = os.path.join(audio_dir, f"generated_audio.{extension}")
            recreate_directory(audio_dir)

            if prompt:
                with st.spinner("Processing..."):
                    generate_audio(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        guidance=guidance,
                        negative_prompt=negative_prompt,
                        seed=starting_seed,
                        width=width,
                        output_path=audio_path,
                        checkpoint=checkpoint,
                        device=device,
                        params=params,
                        scheduler=scheduler,
                        extension=extension,
                    )

            clips[selected_clip - 1] = clips[selected_clip - 1].set_audio(
                mp.AudioFileClip(audio_path)
            )

            output_dir = "output_clips"
            recreate_directory(output_dir)

            for i, clip in enumerate(clips):
                clip.write_videofile(
                    os.path.join(output_dir, f"clip_{i + 1}.mp4"), audio_codec="aac"
                )

            shutil.make_archive("clips", "zip", output_dir)
            st.success("Processing complete!")

            with open("clips.zip", "rb") as f:
                st.download_button(
                    label="Download Clips",
                    data=f,
                    file_name="clips.zip",
                    mime="application/zip",
                )

            cols = st.columns(num_columns)
            for i, clip in enumerate(clips):
                with cols[i % num_columns]:
                    st.video(
                        os.path.join(output_dir, f"clip_{i + 1}.mp4")
                    )
