import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# UPDATE: Updated imports for PyTorch 2.x+
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# --- CONFIGURATION ---
BATCH_SIZE = 4       
LEARNING_RATE = 0.0002
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4      

# Paths 
DATASET_DIR = "terrain_dataset_eroded"
METADATA_PATH = os.path.join(DATASET_DIR, "eroded_metadata.csv")
RAW_DIR = os.path.join(DATASET_DIR, "npy_raw")
CHECKPOINT_DIR = "gan_checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- 1. DATASET LOADING ---

class ErosionDataset(Dataset):
    def __init__(self, metadata_path, raw_dir):
        self.metadata = pd.read_csv(metadata_path)
        self.raw_dir = raw_dir
        
        # We filter out 'ID' and fixed parameters
        self.param_cols = ['p_inertia', 'p_capacity', 'p_evaporation', 'p_erosion', 'p_deposition']

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load Input (Original Perlin)
        input_path = os.path.join("terrain_dataset_10k", "npy_raw", f"terrain_{int(row['ID'])}.npy")
        target_path = os.path.join(self.raw_dir, f"eroded_{int(row['ID'])}.npy")
        
        input_map = np.load(input_path).astype(np.float32)
        target_map = np.load(target_path).astype(np.float32)

        # Normalize Maps to [-1, 1] 
        input_map = (input_map * 2.0) - 1.0
        target_map = (target_map * 2.0) - 1.0

        # Prepare Parameters
        params = row[self.param_cols].values.astype(np.float32)
        
        # Expand Parameters to Spatial Dimensions
        h, w = input_map.shape[0], input_map.shape[1]
        
        # Create parameter channels
        param_channels = []
        for p in params:
            channel = np.full((h, w), p, dtype=np.float32)
            param_channels.append(channel)
        
        # Stack: [Input_Terrain, Param1, Param2, ..., Param5]
        combined_input = np.stack([input_map] + param_channels, axis=0)
        
        # Target is just the eroded map [1, 512, 512]
        target_map = target_map[np.newaxis, :, :]

        return torch.from_numpy(combined_input), torch.from_numpy(target_map)

# --- 2. GENERATOR (U-NET) ---

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if act == "relu" else nn.LeakyReLU(0.2, inplace=True),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=6, features=64): 
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Encoder 
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), 
            nn.ReLU(inplace=True)
        )

        # Decoder 
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, 1, 4, 2, 1),
            nn.Tanh(), 
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        
        # Skip Connections 
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))

# --- 3. DISCRIMINATOR (PatchGAN) ---

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=7, features=[64, 128, 256, 512]):
        super().__init__()
        # This initial layer handles the 7 input channels
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_c = features[0]
        for feature in features[1:]:
            layers.append(CNNBlock(in_c, feature, stride=1 if feature == features[-1] else 2))
            in_c = feature

        layers.append(nn.Conv2d(in_c, 1, 4, 1, 1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        # FIX: Concatenate Input (x) and Target (y) -> 7 channels
        x = torch.cat([x, y], dim=1)
        # FIX: Pass through initial layer FIRST (7 -> 64 channels)
        x = self.initial(x)
        # Then pass through the rest of the model
        return self.model(x)

# --- 4. TRAINING LOOP ---

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def train_fn(disc, gen, loader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE) # Input (Map + Params)
        y = y.to(DEVICE) # Target (Eroded Map)

        # Train Discriminator
        # UPDATE: Updated syntax for autocast to fix Warning
        with autocast(device_type='cuda', dtype=torch.float16):
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with autocast(device_type='cuda', dtype=torch.float16):
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, y) * 100 
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 100 == 0:
            loop.set_postfix(
                D_loss=D_loss.item(),
                G_loss=G_loss.item(),
            )

def main():
    gen = Generator(in_channels=6).to(DEVICE)
    disc = Discriminator(in_channels=7).to(DEVICE)
    
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    dataset = ErosionDataset(METADATA_PATH, RAW_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    
    # UPDATE: Updated syntax for GradScaler to fix Warning
    g_scaler = GradScaler(device='cuda')
    d_scaler = GradScaler(device='cuda')

    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        train_fn(disc, gen, loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler)

        if (epoch + 1) % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=os.path.join(CHECKPOINT_DIR, f"gen_epoch_{epoch+1}.pth.tar"))
            save_checkpoint(disc, opt_disc, filename=os.path.join(CHECKPOINT_DIR, f"disc_epoch_{epoch+1}.pth.tar"))

if __name__ == "__main__":
    main()