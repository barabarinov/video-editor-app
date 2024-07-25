# video-editor-app
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)


[//]: # (This is a web application that allows users to manipulate video files. Users can upload a video file, specify the number of clips to split it into, enter a prompt for generating an audio track, and select which clip to add the generated audio track to. Optionally, users can enter the number of columns in which the clips will be displayed. The output is N video files, one of which will contain the generated audio track instead of the original one. Users can also download all video files as an archive.)
This application allows you to manipulate video files by splitting them into clips, generating audio tracks based on prompts, and replacing the original audio in a chosen clip. You can then download the edited clips individually or as a compressed archive.

## Features

* Upload a video file
* Specify the number of clips to split the video into
* Enter a prompt for generating an audio track with riffusion
* Select the clip to add the generated audio track to
* Optionally set the number of columns for displaying clips (prevents excessive scrolling)
* Download all generated video files as an archive
* (Optional) Additional riffusion settings for advanced users

### Quickstart:
```shell
git clone https://github.com/barabarinov/video-editor-app.git
python -m venv venv
pip install -r requirements.txt
```

### For install ffmpeg run command:
```shell
brew install ffmpeg
```
> [!IMPORTANT]
> ### There’s a version conflict in the file at the given path:
> *~/path/to/virtualenv/lib/python3.11/site-packages/riffusion/spectrogram_converter.py*
>
> 
> I didn’t have enough time to resolve it, so I suggest just commenting out the arguments passed to the InverseMelScale class ([Documentation link](https://pytorch.org/audio/stable/generated/torchaudio.transforms.InverseMelScale.html)), such as:
> 
> 
> ```python
> self.inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
>     n_stft=params.n_fft // 2 + 1,
>     n_mels=params.num_frequencies,
>     sample_rate=params.sample_rate,
>     f_min=params.min_frequency,
>     f_max=params.max_frequency,
>     # max_iter=params.max_mel_iters,
>     # tolerance_loss=1e-5,
>     # tolerance_change=1e-8,
>     # sgdargs=None,
>     norm=params.mel_scale_norm,
>     mel_scale=params.mel_scale_type,
> ).to(self.device)


### To run script use the following command:
```shell
streamlit run main.py
```
