import cv2
import torch

# (height, width, color)
img1 = cv2.imread('./1.jpg', cv2.COLOR_BGR2RGB) # ndarray: (480, 640, 3)
img2 = cv2.imread('./2.jpg', cv2.COLOR_BGR2RGB) # ndarray: (480, 640, 3)
img3 = cv2.imread('./3.jpg', cv2.COLOR_BGR2RGB) # ndarray: (480, 640, 3)

class TensorManipulator:
    #TODO: 작성 필요
    def __init__(self, img1, img2, img3):
        self.img1 = torch.Tensor(img1)
        self.img2 = torch.Tensor(img2)
        self.img3 = torch.Tensor(img3)

    def concatenation(self):
        """
        (3, 3, 480, 640) 크기의 텐서를 반환
        입력된 이미지 3장을 dim=0을 기준으로 연결한 결과물
        dim=1은 이미지 채널 수, dim=2는 이미지 세로길이, dim=3은 이미지 가로길이
        """
        return torch.cat((self.img1, self.img2, self.img3), dim=0)
    
    def flatten(self, tensor):
        return tensor.view(-1)
    
    def average(self, tensor):
        return torch.mean(tensor)
    
    def print(self):
        print(self.img1, self.img1.shape)

obj = TensorManipulator(img1, img2, img3)
out = obj.concatenation()
out_flt = obj.flatten(out)
out_avg = obj.average(out_flt)
print(out.shape, out_flt.shape, out_avg)