import os, random, numpy as np, torch
from pathlib import Path
from PIL import Image
from collections import Counter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.nn as nn, torch.optim as optim
from torchvision import transforms
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE', DEVICE)

DATA_DIR=Path('ultrasound_data')
TRAIN_DIR=DATA_DIR/'train'
TEST_DIR=DATA_DIR/'test'
CLASS_NAMES=sorted([p.name for p in TRAIN_DIR.iterdir() if p.is_dir()])
print('classes', CLASS_NAMES)

IMG_SIZE=128
means=[]; stds=[]
for cls in CLASS_NAMES:
    for i,p in enumerate((TRAIN_DIR/cls).glob('*.*')):
        if i>=100: break
        arr=np.array(Image.open(p).convert('RGB'))/255.0
        means.append(arr.mean()); stds.append(arr.std())
mean=(float(np.mean(means)),)*3
std=(float(np.mean(stds)),)*3
print('norm mean', mean, 'std', std)

train_transforms=transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2,contrast=0.2),
    transforms.RandomAffine(degrees=0,translate=(0.05,0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)
])
val_transforms=transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)
])

class PCOSImageDataset(Dataset):
    def __init__(self, root_dir, class_names, transform=None):
        self.transform=transform
        self.samples=[]
        self.class_to_idx={cls:(1 if 'infected' in cls.lower() and 'not' not in cls.lower() else 0) for cls in class_names}
        for cls in class_names:
            folder=Path(root_dir)/cls
            for ext in ['*.jpg','*.jpeg','*.png','*.bmp']:
                self.samples += [(p,self.class_to_idx[cls]) for p in folder.glob(ext)]
        random.shuffle(self.samples)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p,label=self.samples[idx]
        try:
            img=Image.open(p).convert('RGB')
        except Exception:
            img=Image.new('RGB',(IMG_SIZE,IMG_SIZE),(0,0,0))
        if self.transform: img=self.transform(img)
        return img, torch.tensor(label,dtype=torch.float32)
    def get_labels(self): return [l for _,l in self.samples]

train_ds=PCOSImageDataset(TRAIN_DIR, CLASS_NAMES, transform=train_transforms)
full_test_ds=PCOSImageDataset(TEST_DIR, CLASS_NAMES, transform=val_transforms)
val_size=int(0.5*len(full_test_ds)); test_size=len(full_test_ds)-val_size
val_ds,test_ds=torch.utils.data.random_split(full_test_ds,[val_size,test_size],generator=torch.Generator().manual_seed(SEED))
print('dataset sizes', len(train_ds), len(val_ds), len(test_ds))

BATCH_SIZE=32
weights={cls:1.0/count for cls,count in Counter(train_ds.get_labels()).items()}
train_weights=[weights[label] for label in train_ds.get_labels()]
train_sampler=WeightedRandomSampler(torch.DoubleTensor(train_weights), num_samples=len(train_weights), replacement=True)

train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,sampler=train_sampler,num_workers=2,pin_memory=(DEVICE.type=='cuda'))
val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=(DEVICE.type=='cuda'))
test_loader=DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=(DEVICE.type=='cuda'))
print('loaders', len(train_loader), len(val_loader), len(test_loader))

class ConvBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__(); self.block=nn.Sequential(nn.Conv2d(in_ch,out_ch,3,padding=1,bias=False),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),nn.MaxPool2d(2,2))
    def forward(self,x): return self.block(x)

class PCOSConvNet(nn.Module):
    def __init__(self):
        super().__init__();
        self.features=nn.Sequential(ConvBlock(3,32),ConvBlock(32,64),ConvBlock(64,128),ConvBlock(128,256),ConvBlock(256,256))
        self.global_avg_pool=nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(256,512),nn.BatchNorm1d(512),nn.ReLU(inplace=True),nn.Dropout(0.4),nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(inplace=True),nn.Dropout(0.2),nn.Linear(256,1),nn.Sigmoid())
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear): nn.init.kaiming_normal_(m.weight, nonlinearity='relu'); nn.init.zeros_(m.bias)
    def forward(self,x): return self.classifier(self.global_avg_pool(self.features(x))).squeeze(1)

model=PCOSConvNet().to(DEVICE)
print('model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
criterion=nn.BCELoss(); optimizer=optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler=nn.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-6)

best_loss=1e9; patience=7; wait=0
for epoch in range(1,11):
    model.train(); train_loss=0.0; correct=0; total=0
    for x,y in train_loader:
        x,y=x.to(DEVICE),y.to(DEVICE)
        optimizer.zero_grad(); out=model(x); loss=criterion(out,y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        train_loss += loss.item() * x.size(0); correct += ((out>=0.5).float() == y).sum().item(); total += x.size(0)
    train_loss /= total; train_acc = correct / total
    model.eval(); val_loss=0.0; val_corr=0; val_tot=0
    with torch.no_grad():
        for x,y in val_loader:
            x,y=x.to(DEVICE),y.to(DEVICE); out=model(x); loss=criterion(out,y)
            val_loss += loss.item() * x.size(0); val_corr += ((out>=0.5).float()==y).sum().item(); val_tot += x.size(0)
    val_loss /= val_tot; val_acc = val_corr / val_tot
    scheduler.step(val_loss)
    print(f'Epoch {epoch:02d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')
    if val_loss < best_loss - 1e-4:
        best_loss = val_loss; wait = 0; torch.save(model.state_dict(), 'models/pcos_cnn_final.pt'); print('saved best cnn')
    else:
        wait += 1
        if wait >= patience: print('early stop at', epoch); break

model.load_state_dict(torch.load('models/pcos_cnn_final.pt', map_location=DEVICE)); model.eval()
probs=[]; preds=[]; trues=[]
with torch.no_grad():
    for x,y in test_loader:
        x,y=x.to(DEVICE),y.to(DEVICE); out=model(x)
        probs.extend(out.cpu().numpy()); preds.extend((out>=0.5).float().cpu().numpy()); trues.extend(y.cpu().numpy())
print('test acc', accuracy_score(trues,preds), 'f1', f1_score(trues,preds), 'auc', roc_auc_score(trues,probs))

class CNNFeatureExtractor(nn.Module):
    def __init__(self, full_model):
        super().__init__(); self.features=full_model.features; self.global_avg_pool=full_model.global_avg_pool
    def forward(self,x): return self.global_avg_pool(self.features(x)).view(x.size(0),-1)
extractor=CNNFeatureExtractor(model).to(DEVICE); extractor.eval()
with torch.no_grad():
    dummy=torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(DEVICE)
    print('extractor output shape', extractor(dummy).shape)

with open('models/cnn_extractor_full.pt', 'wb') as f:
    torch.save({'extractor_state': extractor.state_dict(), 'img_size': IMG_SIZE, 'norm_mean': mean, 'norm_std': std, 'feature_dim': 256}, f)
print('saved extractor')

def extract_feats(loader):
    feats=[]; labs=[]
    with torch.no_grad():
        for x,y in loader:
            feats.append(extractor(x.to(DEVICE)).cpu().numpy()); labs.append(y.numpy())
    return np.concatenate(feats,axis=0), np.concatenate(labs,axis=0)
train_clean=DataLoader(PCOSImageDataset(TRAIN_DIR, CLASS_NAMES, transform=val_transforms), batch_size=32, shuffle=False, num_workers=2, pin_memory=(DEVICE.type=='cuda'))
train_feats, train_lbls = extract_feats(train_clean)
val_feats, val_lbls = extract_feats(val_loader)
test_feats, test_lbls = extract_feats(test_loader)
with open('models/cnn_features.pkl','wb') as f:
    pickle.dump({'train_feats':train_feats,'val_feats':val_feats,'test_feats':test_feats,'train_labels':train_lbls,'val_labels':val_lbls,'test_labels':test_lbls,'feature_dim':256,'img_size':IMG_SIZE,'norm_mean':mean,'norm_std':std}, f)
print('saved cnn_features.pkl', train_feats.shape, val_feats.shape, test_feats.shape)
