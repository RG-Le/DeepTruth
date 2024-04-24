import cv2
import torch
import face_recognition
from torch.utils.data.dataset import Dataset


class DataLoader(Dataset):
    def __init__(self,video_names,sequence_length = 60, transform = None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length
        self.org_frames = []

    def reset_org_frames(self):
       self.org_frames = []

    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        # first_frame = np.random.randint(0,a)      
        for i,frame in enumerate(self.frame_extract(video_path)):
            #if(i % a == first_frame):
            faces = face_recognition.face_locations(frame)
            try:
              top,right,bottom,left = faces[0]
              orginal_frame = frame
              frame = frame[top:bottom,left:right,:]
            except:
              pass
            frames.append(self.transform(frame))
            print(f"Processed Frames: {len(frames)}")
            ## POSSIBEL ERROR FILE
            self.org_frames.append(orginal_frame.astype(int).tolist())
            if(len(frames) == self.count):
              break
        #print("no of frames",len(frames))
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    
    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path) 
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image