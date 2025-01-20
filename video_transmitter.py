import torch
import cv2
import numpy as np
from pathlib import Path
import logging
from train import VideoCompressor
import argparse
from tqdm import tqdm

class VideoTransmissionSystem:
    def __init__(self, model_path, chunk_size=32, scale_factor=0.5):
#        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.chunk_size = chunk_size
        self.scale_factor = scale_factor
        
        # Load the model
        self.model = VideoCompressor().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def compress_chunk(self, frames):
        """Compress a chunk of frames using the neural network."""
        with torch.no_grad():
            # Convert frames to tensor
            frames_tensor = torch.stack([
                torch.FloatTensor(frame).permute(2, 0, 1) / 255.0 
                for frame in frames
            ]).unsqueeze(0).to(self.device)
            
            # Compress using the model
            compressed = self.model(frames_tensor)
            
            # Convert back to numpy arrays
            compressed_frames = [
                (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                for frame in compressed[0]
            ]
            
            return compressed_frames

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate new dimensions
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)  # Original size output
        )

        frames_buffer = []
        pbar = tqdm(total=frame_count, desc="Processing video")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Downsample
            small_frame = cv2.resize(frame, (new_width, new_height))
            frames_buffer.append(small_frame)

            if len(frames_buffer) == self.chunk_size:
                # Process chunk
                compressed_frames = self.compress_chunk(frames_buffer)
                
                # Write frames
                for compressed_frame in compressed_frames:
                    # Upsample back to original size
                    restored_frame = cv2.resize(compressed_frame, (width, height))
                    out.write(restored_frame)
                
                frames_buffer = []
                pbar.update(self.chunk_size)

        # Process remaining frames
        if frames_buffer:
            # Pad to chunk_size if necessary
            while len(frames_buffer) < self.chunk_size:
                frames_buffer.append(frames_buffer[-1])
            
            compressed_frames = self.compress_chunk(frames_buffer)
            
            # Only write the actual number of remaining frames
            for compressed_frame in compressed_frames[:len(frames_buffer)]:
                restored_frame = cv2.resize(compressed_frame, (width, height))
                out.write(restored_frame)
            
            pbar.update(len(frames_buffer))

        cap.release()
        out.release()
        pbar.close()

def main():
    parser = argparse.ArgumentParser(description="Neural video transmission system")
    parser.add_argument("--input_video", type=str, required=True, help="Input video path")
    parser.add_argument("--output_video", type=str, required=True, help="Output video path")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--chunk_size", type=int, default=32, help="Size of video chunks")
    parser.add_argument("--scale_factor", type=float, default=0.5, help="Scale factor for downsampling")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    transmitter = VideoTransmissionSystem(
        args.model_path,
        chunk_size=args.chunk_size,
        scale_factor=args.scale_factor
    )
    
    try:
        transmitter.process_video(args.input_video, args.output_video)
        print(f"Successfully processed video to: {args.output_video}")
    except Exception as e:
        print(f"Error processing video: {e}")
        raise

if __name__ == "__main__":
    main()
