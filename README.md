# video-editor-app
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)


This is a web application that allows users to manipulate video files. Users can upload a video file, specify the number of clips to split it into, enter a prompt for generating an audio track, and select which clip to add the generated audio track to. Optionally, users can enter the number of columns in which the clips will be displayed. The output is N video files, one of which will contain the generated audio track instead of the original one. Users can also download all video files as an archive.

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

### To run script use the following command:
```shell
streamlit run main.py
```
