import os
import uuid
import requests as req
from fastapi import UploadFile, File, HTTPException
from pytube import YouTube


class VideoHandler:
    def __init__(self):
        self.allowed_extensions = ('.mp4', '.avi', '.mov', '.mkv')

    def generate_path(self, vid_file: UploadFile = None):
        os.makedirs("Recieved_Videos", exist_ok=True)
        base_path = "Recieved_Videos/"

        if vid_file:
            save_path = base_path + vid_file.filename
        save_path = base_path + str(uuid.uuid4()) + ".mp4"

        return save_path

    @staticmethod
    def link_upload(video_url: str):
        base_path = "Recieved_Videos"
        try:
            yt = YouTube(url=video_url)
            video_stream = yt.streams.filter(res="480p", file_extension='.mp4').first()
            save_path = video_stream.download(output_path=base_path, filename=yt.title)
            print("----Video Downloaded and Saved Successfully----")
            return save_path
        except Exception as e:
            HTTPException(status_code=400, detail=f"Error downloading YouTube video: {str(e)}")

    def vid_upload(self, vid_file=File(...)):
        vid_path = self.generate_path(vid_file)
        file_extension = vid_file.filename[-4:].lower()

        if file_extension not in self.allowed_extensions:
            raise TypeError("Unsupported File Format!!")

        try:
            print("----Starting to Save the File----")
            with open(vid_path, "wb") as buffer:
                buffer.write(vid_file.file.read())
            return vid_path

        except Exception as e:
            raise Exception(e, "Error while saving the Uploaded file")
