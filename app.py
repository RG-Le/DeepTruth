import os
import numpy as np
import uuid
import streamlit as st
from pytube import YouTube

from prediction_pipeline import PredictionPipeline
from pdf_generator import PDFGenerator

pipeline = PredictionPipeline(num_classes=2)
OUT_CLASS = ["FAKE", "REAL"]
base_vid_save_path = "Recieved_Videos/"
report_file_path = "temp_files/detection_report.pdf"


def analyse(vid_path, sequence_length):
    selected_frames, cropped_frames = pipeline.load_data(vid_paths=vid_path, sequence_length=sequence_length)
    results = pipeline.predict()

    pdf = PDFGenerator(report_file_path, selected_frames, cropped_frames, results)

    response = {
        "message": "Successful Detection!!",
        "Selected_Frames": selected_frames,
        "Cropped_Frames": cropped_frames,
        "Confidence": results[1],
        "Deepfake_Region_Image": results[2],
        "Prediction": OUT_CLASS[results[0]]
    }
    return response


def np_norm(frames, norm=True, np_al=False):
    if not np_al:
        frames = np.array(frames)
    if norm:
        min_val, max_val = frames.min(), frames.max()
        frames = (frames - min_val) / (max_val - min_val)
    return frames


def file_to_bytes(file_path):
    with open(file_path, "rb") as f:
        bytes_data = f.read()
    return bytes_data


def main():
    st.title("Deepfake Video Detection App")

    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
    video_link = st.text_input("Video Link")

    num_frames_options = [10, 20, 40, 60]
    num_frames = st.selectbox("Select Number of Frames", options=num_frames_options)

    video_path = ""
    if st.button("Analyse"):
        if uploaded_file is not None:
            # Read uploaded video
            with st.spinner("Temporary Saving File...."):
                vid_path = base_vid_save_path + str(uuid.uuid4()) + ".mp4"
                video_path = vid_path
                with open(vid_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

        elif video_link:
            # Download video from link
            with st.spinner("Temporary Saving File...."):
                yt = YouTube(video_link)
                video_stream = yt.streams.filter(res="480p", file_extension="mp4").first()
                vid_path = video_stream.download(output_path=base_vid_save_path, filename=yt.title)
                video_path = vid_path
        else:
            st.error("Please upload a video or provide a video link.")
            return

        with st.spinner("Detecting Video...."):
            results = analyse([video_path], sequence_length=num_frames)

        st.header("Detection Results")

        # Display Selected Frames
        selected_frames = np_norm(results["Selected_Frames"])
        # st.image(selected_frames[0])
        st.subheader("Selected Frames")
        st.image(selected_frames, width=150, use_column_width="never")

        # Display Cropped Frames
        st.subheader("Cropped Frames")
        cropped_frames = np_norm(results["Cropped_Frames"])
        st.image(cropped_frames, width=150, use_column_width="never")

        # Display Prediction Class and Confidence
        pred_class, conf = results["Prediction"], results["Confidence"]
        st.subheader("Prediction Result and Confidence")
        st.write(f"Prediction Class: {pred_class}")
        st.write(f"Prediction Confidence: {conf}")

        st.subheader("Probable Regions of Deepfake")
        ht_map_img = np_norm(results["Deepfake_Region_Image"], np_al=True)
        st.image(ht_map_img)

        # Download PDF report
        pdf_bytes = file_to_bytes("temp_files/detection_report.pdf")
        st.download_button(label="Download PDF Report",
                           data=pdf_bytes,
                           file_name="Detection_Report.pdf",
                           mime="application/pdf")


if __name__ == "__main__":
    main()
