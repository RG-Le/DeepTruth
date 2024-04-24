from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pytube import YouTube

from incoming_vid_handler import VideoHandler
from prediction_pipeline import PredictionPipeline
from pdf_generator import PDFGenerator

app = FastAPI()
inc_video_handler = VideoHandler()
pipeline = PredictionPipeline(num_classes=2)
out_class = ["FAKE", "REAL"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=["*"],
    allow_headers=["*"],
)


def analyse(vid_path, sequence_length):
    selected_frames, cropped_frames = pipeline.load_data(vid_paths=vid_path, sequence_length=sequence_length)
    results = pipeline.predict()

    report_file_path = "temp_files/detection_report.pdf"
    pdf = PDFGenerator(report_file_path, selected_frames, cropped_frames, results)

    response = {
        "message": "Successful Detection!!",
        "Cropped_Frames": cropped_frames,
        "Confidence": results[1],
        "Prediction": out_class[results[0]]
    }

    return response


@app.get("/download_pdf/")
async def download_pdf():
    pdf_file_path = "detection_report_1.pdf"
    try:
        return FileResponse(pdf_file_path, media_type="application/pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve PDF file")


@app.post("/upload_video_link/")
async def link_upload(vid_details: dict):
    base_path = "Recieved_Videos"
    try:
        vid_link = vid_details.get("vid_link")
        sequence_length = vid_details.get("sequence_length")
        yt = YouTube(vid_link)
        video_stream = yt.streams.filter(res="480p", file_extension="mp4").first()
        save_path = video_stream.download(output_path=base_path, filename=yt.title)
        print(f"---Video Downloaded and Saved Successfully----\n{save_path}")
        return analyse([save_path], sequence_length)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download video: {e}")


@app.post("/upload_video/")
async def upload_video(sequence_length: int, video: UploadFile = File(...)):
    try:
        save_path = [inc_video_handler.vid_upload(vid_file=video)]
        print("----File Saved Successfully---")
        result = analyse(save_path, sequence_length)
        return result
    except Exception as e:
        print("An error occured", e)
        raise HTTPException(status_code=400, detail="Bad Request!! Please Try Again")


@app.get("/home")
async def root():
    return {
        "message": "Hello!! Server is Running!!"
    }
