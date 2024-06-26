
#1 VGG 기본 블록 정의

import torch
import torch.nn as nn

class BasicBlock(nn.Module): # 기본 블록 정의
   
   # 기본 블록을 구성하는 층 정의
   def __init__(self, in_channels, out_channels, hidden_dim):
    
       # nn.Module 클래스의 요소 상속
       super(BasicBlock, self).__init__()

       # 합성곱층 정의
       self.conv1 = nn.Conv2d(in_channels, hidden_dim,
                              kernel_size=3, padding=1)
       self.conv2 = nn.Conv2d(hidden_dim, out_channels, 
                              kernel_size=3, padding=1)
       self.relu = nn.ReLU()

       # stride는 커널의 이동 거리
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

   def forward(self, x): #  기본 블록의 순전파 정의
       x = self.conv1(x)
       x = self.relu(x)
       x = self.conv2(x)
       x = self.relu(x)
       x = self.pool(x)

       return x
   

#2 VGG 모델 정의

class CNN(nn.Module):
   def __init__(self, num_classes): # num_classes는 클래스 개수
       super(CNN, self).__init__()

       # 합성곱 기본 블록 정의
       self.block1 = BasicBlock(in_channels=3, out_channels=32, hidden_dim=16)
       self.block2 = BasicBlock(in_channels=32, out_channels=128, hidden_dim=64)
       self.block3 = BasicBlock(in_channels=128, out_channels=256, 
                                hidden_dim=128)

       # 분류기 정의
       self.fc1 = nn.Linear(in_features=4096, out_features=2048)
       self.fc2 = nn.Linear(in_features=2048, out_features=256)
       self.fc3 = nn.Linear(in_features=256, out_features=num_classes)


       # 분류기의 활성화 함수
       self.relu = nn.ReLU()

   def forward(self, x):
       x = self.block1(x)
       x = self.block2(x)
       x = self.block3(x)  # 출력 모양: (-1, 256, 4, 4) 
       x = torch.flatten(x, start_dim=1) # 2차원 특징 맵을 1차원으로

       x = self.fc1(x)
       x = self.relu(x)
       x = self.fc2(x)
       x = self.relu(x)
       x = self.fc3(x)

       return x


#3 전이 학습 모델 VGG로 분류하기

    #3-1 데이터 전처리

from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
from torch.optim.adam import Adam

transforms = Compose([
   RandomCrop((32, 32), padding=4),  # 랜덤 크롭핑
   RandomHorizontalFlip(p=0.5),      # y축으로 좌우대칭
   ToTensor(),                       # 텐서로 변환

   # 이미지 정규화
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])


    #3-2 데이터 로드 및 모델 정의

from torchvision.datasets import CIFAR10

# 학습용 데이터와 평가용 데이터 불러오기
training_data = CIFAR10(root="./", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./", train=False, download=True, transform=transforms)


# 데이터로더 정의
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# 학습을 진행할 프로세서 설정
device = "cuda" if torch.cuda.is_available() else "cpu"


# CNN 모델 정의
model = CNN(num_classes=10)

# 모델을 device로 보냄
model.to(device)


    #3-3 학습 루프
# 학습률 정의
lr = 1e-3

# 최적화 기법 정의
optim = Adam(model.parameters(), lr=lr)

# 학습 루프 정의
for epoch in range(100):
   for data, label in train_loader:  # 데이터 호출
       optim.zero_grad()  # 기울기 초기화

       preds = model(data.to(device)) # 모델의 예측

       # 오차역전파와 최적화
       loss = nn.CrossEntropyLoss()(preds, label.to(device)) 
       loss.backward() 
       optim.step() 

   if epoch==0 or epoch%10==9:  # 10번마다 손실 출력
       print(f"epoch{epoch+1} loss:{loss.item()}")


# 모델 저장
torch.save(model.state_dict(), "CIFAR.pth")


    #3-4 모델 성능 평가하기
model.load_state_dict(torch.load("CIFAR.pth", map_location=device))

num_corr = 0

with torch.no_grad():
   for data, label in test_loader:

       output = model(data.to(device))
       preds = output.data.max(1)[1]
       corr = preds.eq(label.to(device).data).sum().item()
       num_corr += corr

   print(f"Accuracy:{num_corr/len(test_data)}")


#4 전이 학습 모델 VGG로 분류하기

    #4-1 사전학습된 모델 준비
import torch
import torch.nn as nn

from torchvision.models.vgg import vgg16

device = "cuda" if torch.cuda.is_available() else "cpu"

model = vgg16(pretrained=True) # vgg16 모델 객체 생성
fc = nn.Sequential(            # 분류층 정의
       nn.Linear(512 * 7 * 7, 4096),
       nn.ReLU(),
       nn.Dropout(),           # 드롭아웃층 정의
       nn.Linear(4096, 4096),
       nn.ReLU(),
       nn.Dropout(),
       nn.Linear(4096, 10),
   )

model.classifier = fc          # VGG의 classifier를 덮어씀
model.to(device)



    # 4-2 데이터 전처리
from tqdm import tqdm

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader

from torch.optim.adam import Adam

transforms = Compose([
   Resize(224),
   RandomCrop((224, 224), padding=4),
   RandomHorizontalFlip(p=0.5),
   ToTensor(),
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])



    # 4-3 학습 루프 구성

# 데이터로더 정의
training_data = CIFAR10(root="./", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# 학습 루프 정의
lr = 1e-4
optim = Adam(model.parameters(), lr=lr)

for epoch in range(30):
   iterator = tqdm.tqdm(train_loader)   # 학습 로그 출력
   for data, label in iterator:
       optim.zero_grad()

       preds = model(data.to(device))   # 모델의 예측값 출력

       loss = nn.CrossEntropyLoss()(preds, label.to(device))
       loss.backward()
       optim.step()

       # tqdm이 출력할 문자열
       iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")

torch.save(model.state_dict(), "CIFAR_pretrained.pth") # 모델 저장



    # 4-4 모델 성능 평가
model.load_state_dict(torch.load("CIFAR_pretrained.pth", map_location=device))

num_corr = 0

with torch.no_grad():
   for data, label in test_loader:

       output = model(data.to(device))
       preds = output.data.max(1)[1]
       corr = preds.eq(label.to(device).data).sum().item()
       num_corr += corr

   print(f"Accuracy:{num_corr/len(test_data)}")
