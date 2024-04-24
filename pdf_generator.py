import os
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

out_class = ["FAKE", "REAL"]


class PDFGenerator:
    def __init__(self, filename, selected_frames, cropped_frames, results):
        self.filename = filename
        self.selected_frames = np.array(selected_frames, dtype=float)
        self.selected_frames = self.selected_frames/255.
        self.cropped_frames = np.array(cropped_frames, dtype=float)
        self.res_class = out_class[results[0]]
        self.confidence = results[1]
        self.heatmap_img = results[2]

        self.filename_selected_images = []
        self.filename_cropped_images = []
        self.save_images()
        self.create_pdf_with_content()

    def save_images(self):
        dir_name = "Detection_Result_Images"
        os.makedirs(dir_name, exist_ok=True)

        # Selected_Frames
        cnter = 0
        for i in range(0, len(self.selected_frames), 5):
            combined_frames = self.selected_frames[i:i+5]
            combined_image = np.concatenate(combined_frames, axis=1)
            img_name = f"{dir_name}/selected_combined_{cnter}.png"
            self.filename_selected_images.append(img_name)
            plt.imsave(img_name, combined_image)
            cnter += 1

        # Cropped_Frames
        cnter = 0
        min_val = self.cropped_frames.min()
        max_val = self.cropped_frames.max()
        for i in range(0, len(self.cropped_frames), 5):
            combined_frames = self.cropped_frames[i:i + 5]
            combined_frames = (combined_frames - min_val) / (max_val - min_val)
            combined_image = np.concatenate(combined_frames, axis=1)
            img_name = f"{dir_name}/cropped_combined_{cnter}.png"
            self.filename_cropped_images.append(img_name)
            plt.imsave(img_name, combined_image)
            cnter += 1

        min_val = self.heatmap_img.min()
        max_val = self.heatmap_img.max()
        normed_heatmap_img = (self.heatmap_img - min_val) / (max_val - min_val)
        plt.imsave(f"{dir_name}/heat_map_res.png", normed_heatmap_img)

    def create_pdf_with_content(self):
        c = canvas.Canvas(self.filename, pagesize=letter)

        # Set font size and leading for text
        c.setFont("Helvetica", 12)
        leading = 20

        # Add heading
        c.drawString(100, 750, "Deepfake Detection Report")
        c.drawString(100, 730, "-" * 50)

        # Add text explaining about the report
        text = f"""
        This report contains the results of the deepfake detection process. 
        This includes the frames that were selected by the model, the cropped frames that the model took as input.
        The results for detection report include the confidence and class of detection. 
        There is also a heatmap image of the detection showing the areas which were likely to be deepfake.
        ------------------------------------------------------------------------------------------------
        Resulting Class: {self.res_class}
        Confidence: {self.confidence}
        """
        text_lines = text.split("\n")
        for line in text_lines:
            c.drawString(100, 700 - leading, line.strip())
            leading += 20

        # Add Sub heading
        c.drawString(100, 700, "Frames Selected By Model")
        c.drawString(100, 690, "-" * 50)

        # Add Selected images to the PDF
        for i, image_filename in enumerate(self.filename_selected_images):
            x_offset = 100 if i % 2 == 0 else 300
            y_offset = 400 if i < 2 else 200
            c.drawImage(image_filename, x_offset, y_offset, width=200, height=150)
            if i % 2 == 1:
                c.showPage()  # Add a page after two images

        # Add Sub heading
        c.drawString(100, 700, "Cropped Frames Supplied To Model")
        c.drawString(100, 690, "-" * 50)

        # # Add Cropped images to the PDF
        # for i, image_filename in enumerate(self.cropped_frames):
        #     x_offset = 100 if i % 2 == 0 else 300
        #     y_offset = 400 if i < 2 else 200
        #     c.drawImage(image_filename, x_offset, y_offset, width=200, height=150)
        #     if i % 2 == 1:
        #         c.showPage()  # Add a page after two images

        c.drawString(100, 690, "-" * 50)

        # Add HeatMap Image to the PDF
        c.drawImage("Detection_Result_Images/heat_map_res.png", 100, 400, width=250, height=250)
        c.drawString(100, 690, "-" * 50)
        c.drawString(100, 690, "-" * 50)

        # Add concluding text
        conclusion_text = """
        This concludes the detection report. 
        """
        conclusion_lines = conclusion_text.split("\n")
        leading = 20
        for line in conclusion_lines:
            c.drawString(100, 750 - leading, line.strip())
            leading += 20

        c.save()
