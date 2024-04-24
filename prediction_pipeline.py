import os
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from model_init import Model
from data_loader import DataLoader

class PredictionPipeline:

    def __init__(self, num_classes: int = 2, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225]):
        #self.vid_paths = vid_paths
        self.num_classes = num_classes
        self.im_size = 112
        self.mean = mean
        self.std = std
        
        self.model_10, self.md_10_init_state = Model(self.num_classes), False
        self.model_20, self.md_20_init_state = Model(self.num_classes), False
        self.model_40, self.md_40_init_state = Model(self.num_classes), False
        self.model_60, self.md_60_init_state = Model(self.num_classes), False
        self.wrk_model = None

        self.__sm = nn.Softmax()
        self.__initialize_model_wts()
        # self.vid_data_loader = DataLoader(self.vid_paths, self.sequence_length, transform=self.data_transforms)
        # self.vid_data, self.org_frames, self.crop_frames = self.load_data()
        self.vid_data = None

        self.data_transforms = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((self.im_size, self.im_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(self.mean, self.std)
                                    ])
    
    def __initialize_model_wts(self):
        model_wt_dir = "model_weights"
        model_wt_paths = os.listdir(model_wt_dir)
        
        if not self.md_10_init_state:
            self.model_10.load_state_dict(torch.load(model_wt_dir + "/" + model_wt_paths[0]))
            self.md_10_init_state = True

        if not self.md_20_init_state:
            self.model_20.load_state_dict(torch.load(model_wt_dir + "/" + model_wt_paths[1]))
            self.md_20_init_state = True
        
        if not self.md_40_init_state:
            self.model_40.load_state_dict(torch.load(model_wt_dir + "/" + model_wt_paths[2]))
            self.md_40_init_state = True
        
        if not self.md_60_init_state:
            self.model_60.load_state_dict(torch.load(model_wt_dir + "/" + model_wt_paths[3]))
            self.md_60_init_state = True
        
    def choose_model(self, sequence_length):
        if sequence_length == 10:
            self.wrk_model = self.model_10.to('cuda')
        elif sequence_length == 20:
            self.wrk_model = self.model_20.to('cuda')
        elif sequence_length == 40:
            self.wrk_model = self.model_40.to('cuda')
        elif sequence_length == 60:
            self.wrk_model = self.model_60.to('cuda')
        else:
            raise Exception("Out of Bound Sequence Length!!")

    def load_data(self, vid_paths, sequence_length):
        vid_data_loder = DataLoader(video_names=vid_paths, sequence_length=sequence_length, transform=self.data_transforms)
        print("------Processing the Video------")
        self.vid_data = vid_data_loder.__getitem__(0)
        orginal_frames = vid_data_loder.org_frames
        print("------Getting Cropped Frames--------")
        crop_frames = self.get_cropped_frame_data()

        self.choose_model(sequence_length)

        return orginal_frames, crop_frames

    def get_cropped_frame_data(self):
        cropped_images = []
        _, frame_cnt, _, _, _ = self.vid_data.shape
        for i in tqdm(range(frame_cnt)):
            cropped_images.append(self.get_img_data(self.vid_data[0][i]))
        return cropped_images

    @staticmethod
    def get_img_data(tensor):
        image = tensor.cpu().numpy().transpose(1,2,0)
        b,g,r = cv2.split(image)
        image = cv2.merge((r,g,b))
        image = image*[0.22803, 0.22145, 0.216989] +  [0.43216, 0.394666, 0.37645]
        image = image*255.0
        # print(image.shape)
        return image.astype(int).tolist()

    def predict(self):
        if not self.wrk_model:
            raise Exception("Model Not Loaded!! Pls use function load_data() first!!!")
        
        print("------Starting Prediction-------")
        
        fmap,logits = self.wrk_model(self.vid_data.to('cuda'))
        params = list(self.wrk_model.parameters())
        weight_softmax = self.wrk_model.linear1.weight.detach().cpu().numpy()
        logits = self.__sm(logits)
        _,prediction = torch.max(logits,1)
        confidence = logits[:,int(prediction.item())].item()*100
        # print('confidence of prediction:',logits[:,int(prediction.item())].item()*100)
        idx = np.argmax(logits.detach().cpu().numpy())
        bz, nc, h, w = fmap.shape
        out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
        predict = out.reshape(h,w)
        predict = predict - np.min(predict)
        predict_img = predict / np.max(predict)
        predict_img = np.uint8(255*predict_img)
        out = cv2.resize(predict_img, (self.im_size, self.im_size))
        heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
        img = self.im_convert(self.vid_data[:,-1,:,:,:])
        result = heatmap * 0.5 + img*0.8*255
        #cv2.imwrite('/content/1.png',result)
        result1 = heatmap * 0.5/255 + img*0.8
        r,g,b = cv2.split(result1)
        result1 = cv2.merge((r,g,b))
        # plt.imshow(result1)
        # plt.show()
        self.wrk_model.to('cpu')

        print("-----Completed Prediciton------")
        return [int(prediction.item()),confidence, result1]
    
    def im_convert(self, tensor):
        """ Display a tensor as an image. """
        inv_normalize =  transforms.Normalize(mean=-1*np.divide(self.mean, self.std),std=np.divide([1,1,1],self.std))

        image = tensor.to("cpu").clone().detach()
        image = image.squeeze()
        image = inv_normalize(image)
        image = image.numpy()
        image = image.transpose(1,2,0)
        image = image.clip(0, 1)
        cv2.imwrite('./2.png',image*255)

        return image