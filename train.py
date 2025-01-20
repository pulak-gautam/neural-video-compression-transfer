import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

class TemporalVideoDataset(Dataset):
    def __init__(self, video_dir, max_videos=None, sequence_length=8):
        self.sequence_length = sequence_length
        self.videos = []
        self.sizes = []
        
        video_paths = list(Path(video_dir).glob('*.mp4')) + list(Path(video_dir).glob('*.avi'))
        if max_videos:
            video_paths = video_paths[:max_videos]
            
        logging.info(f"Loading {len(video_paths)} videos")
        
        for video_path in tqdm(video_paths, desc="Loading videos"):
            frames = []
            cap = cv2.VideoCapture(str(video_path))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = torch.FloatTensor(frame).permute(2, 0, 1) / 255.0
                frames.append(frame)
            
            cap.release()
            if frames:
                self.videos.append(frames)
                self.sizes.append((frames[0].shape[1], frames[0].shape[2]))
                
        logging.info(f"Loaded {len(self.videos)} videos with {sum(len(v) for v in self.videos)} total frames")
        
    def __len__(self):
        return sum(len(v) - self.sequence_length + 1 for v in self.videos)
        
    def __getitem__(self, idx):
        video_idx = 0
        while idx >= len(self.videos[video_idx]) - self.sequence_length + 1:
            idx -= len(self.videos[video_idx]) - self.sequence_length + 1
            video_idx += 1
        sequence = self.videos[video_idx][idx:idx + self.sequence_length]
        sequence = torch.stack(sequence)
        return sequence, self.sizes[video_idx]

class MotionEstimation(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow_net = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1)
        )
        
    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=1)
        flow = self.flow_net(concat)
        return flow

class VideoCompressor(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        self.motion_estimation = MotionEstimation()
        
        self.temporal_encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(2, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, latent_dim, kernel_size=(2, 3, 3), padding=(0, 1, 1)),
            nn.ReLU()
        )
        
        self.temporal_attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_dim + 2, 64, kernel_size=(2, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 3, kernel_size=(2, 3, 3), padding=(0, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, S, H, W]
        
        # Motion estimation
        motion_features = []
        for i in range(s-1):
            flow = self.motion_estimation(x[:, :, i], x[:, :, i+1])
            motion_features.append(flow)
        motion_features = torch.stack(motion_features, dim=2)
        
        # Temporal encoding
        encoded = self.temporal_encoder(x)  # [B, 128, S', H', W']
        
        # Get actual dimensions after encoding
        b_enc, c_enc, s_enc, h_enc, w_enc = encoded.size()
        
        # Reshape preserving feature information
        encoded_flat = encoded.permute(2, 0, 1, 3, 4)  # [S', B, C, H', W']
        encoded_flat = encoded_flat.reshape(s_enc, b_enc, c_enc * h_enc * w_enc)
        
        # Project to attention dimension
        projection_layer = nn.Linear(c_enc * h_enc * w_enc, 128).to(encoded.device)
        encoded_flat = projection_layer(encoded_flat)  # [S', B, 128]
        
        # Apply attention
        attended, _ = self.temporal_attention(encoded_flat, encoded_flat, encoded_flat)
        
        # Project back
        reverse_projection = nn.Linear(128, c_enc * h_enc * w_enc).to(encoded.device)
        attended = reverse_projection(attended)
        
        # Reshape back
        attended = attended.view(s_enc, b_enc, c_enc, h_enc, w_enc)
        attended = attended.permute(1, 2, 0, 3, 4)  # [B, C, S', H', W']
        
        # Resize motion features
        motion_features = nn.functional.interpolate(
            motion_features,
            size=(attended.size(2), attended.size(3), attended.size(4)),
            mode='trilinear',
            align_corners=False
        )
        
        # Combine and decode
        combined = torch.cat([attended, motion_features], dim=1)
        decoded = self.decoder(combined)
        
        # Match input size
        decoded = nn.functional.interpolate(
            decoded,
            size=(s, h, w),
            mode='trilinear',
            align_corners=False
        )
        
        return decoded.permute(0, 2, 1, 3, 4)

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.info(f"Using device: {device}")
    
    model = VideoCompressor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    
    dataset = TemporalVideoDataset(args.video_dir, max_videos=args.max_videos, sequence_length=8)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for sequences, _ in progress:
            sequences = sequences.to(device)
            optimizer.zero_grad()
            
            output = model(sequences)
            loss = criterion(output, sequences)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            logging.info(f"Saved new best model with loss: {best_loss:.6f}")
        
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pth")

def main():
    parser = argparse.ArgumentParser(description="Train video compression model")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of videos to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    train_model(args)

if __name__ == "__main__":
    main()
