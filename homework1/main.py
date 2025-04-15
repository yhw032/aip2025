import cv2
import torch

# (height, width, color)
img1 = cv2.imread('./1.jpg', cv2.COLOR_BGR2RGB) # ndarray: (480, 640, 3)
img2 = cv2.imread('./2.jpg', cv2.COLOR_BGR2RGB) # ndarray: (480, 640, 3)
img3 = cv2.imread('./3.jpg', cv2.COLOR_BGR2RGB) # ndarray: (480, 640, 3)

class TensorManipulator:
    #TODO: 작성 필요
    def __init__(self, img1, img2, img3):
        """
        이미지들을 입력받아 pytorch tensor로 변환
        """
        # permute: (480, 640, 3) -> (3, 480, 640)
        # unsqueeze: (3, 480, 640) -> (1, 3, 480, 640)
        self.img1 = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        self.img2 = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        self.img3 = torch.tensor(img3, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


    def concatenation(self):
        """
        (3, 3, 480, 640) 크기의 텐서를 반환
        입력된 이미지 3장을 dim=0을 기준으로 연결한 결과물
        dim=1은 이미지 채널 수, dim=2는 이미지 세로길이, dim=3은 이미지 가로길이
        """
        return torch.cat((self.img1, self.img2, self.img3), dim=0)
    
    def flatten(self, tensor):
        """
        concatenation() 함수 결과물인 (3, 3, 480, 640) 크기의 텐서를 (3, 921600) 크기로 변환하여 반환
        3*480*640 = 921600
        """
        return torch.flatten(tensor, start_dim=1)
    
    def average(self, tensor):
        """
        flatten()함수의 결과물인 (3, 921600) 크기의 텐서를 입력받아 각 이미지 텐서들의 평균을 반환
        - 이미지 1번 값들의 평균은 144.0534, 이미지 2번 값들의 평균은 136.8974, 이미지 3번 값들의 평균은 85.5971
        - 값의 형태가 실수형임에 유의
        - 해당 평균값들은 라이브러리 및 연산 방식에 따라 조금씩 다를 수 있음
        """
        return torch.mean(tensor, dim=1)

obj = TensorManipulator(img1, img2, img3)

out = obj.concatenation()
out_flt = obj.flatten(out)
out_avg = obj.average(out_flt)

print(out.shape)
print(out_flt.shape)
print(out_avg)