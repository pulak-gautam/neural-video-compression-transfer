import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import argparse
import logging
from pytorch_msssim import ssim, ms_ssim
from tqdm import tqdm
from train import VideoCompressor  # Import model from training file

def calculate_metrics(original, compressed):
    """Calculate SSIM, PSNR and MSE metrics."""
    mse = torch.mean((original - compressed) ** 2).item()
    psnr = -10 * np.log10(mse + 1e-8)
    ssim_val = ssim(original, compressed, data_range=1.0).item()
    ms_ssim_val = ms_ssim(original, compressed, data_range=1.0).item()
    
    return {
        'SSIM': ssim_val,
        'MS-SSIM': ms_ssim_val,
        'PSNR': psnr,
        'MSE': mse
    }

def compress_video(args):
    device="cpu"
    logging.info(f"Using device: {device}")
    
    try:
        # Load model
        model = VideoCompressor().to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
        # Open video
        cap = cv2.VideoCapture(args.input_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {args.input_video}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup output video writer
        output_path = Path(args.output_dir) / f"compressed_{Path(args.input_video).name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Metrics storage
        metrics_list = []
        sequence_length = 8  # Same as training
        frame_buffer = []
        
        with torch.no_grad():
            pbar = tqdm(total=frame_count, desc="Compressing video")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert to tensor
                frame_tensor = torch.FloatTensor(frame).permute(2, 0, 1).unsqueeze(0) / 255.0
                frame_buffer.append(frame_tensor)
                
                if len(frame_buffer) == sequence_length:
                    # Process sequence
                    sequence = torch.cat(frame_buffer, dim=0).unsqueeze(0).to(device)
                    compressed_sequence = model(sequence)
                    
                    # Calculate metrics for middle frame
                    mid_idx = sequence_length // 2
                    metrics = calculate_metrics(
                        sequence[0, mid_idx:mid_idx+1].to(device),
                        compressed_sequence[0, mid_idx:mid_idx+1]
                    )
                    metrics_list.append(metrics)
                    
                    # Save middle frame
                    compressed_frame = (compressed_sequence[0, mid_idx] * 255).byte().permute(1, 2, 0).cpu().numpy()
                    out.write(compressed_frame)
                    
                    # Update buffer
                    frame_buffer = frame_buffer[1:]
                
                pbar.update(1)
        
        # Process remaining frames
        while frame_buffer:
            padding = [frame_buffer[-1] for _ in range(sequence_length - len(frame_buffer))]
            sequence = torch.cat(frame_buffer + padding, dim=0).unsqueeze(0).to(device)
            compressed_sequence = model(sequence)
            
            for i in range(len(frame_buffer)):
                compressed_frame = (compressed_sequence[0, i] * 255).byte().permute(1, 2, 0).cpu().numpy()
                out.write(compressed_frame)
            
            frame_buffer = []
        
        cap.release()
        out.release()
        
        # Calculate average metrics
        avg_metrics = {
            metric: np.mean([m[metric] for m in metrics_list])
            for metric in metrics_list[0].keys()
        }
        
        logging.info("\nCompression Results:")
        logging.info("-" * 50)
        for metric, value in avg_metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        # Calculate compression ratio
        original_size = Path(args.input_video).stat().st_size
        compressed_size = output_path.stat().st_size
        compression_ratio = original_size / compressed_size
        logging.info(f"Compression Ratio: {compression_ratio:.2f}x")
        
        return output_path, avg_metrics
        
    except Exception as e:
        logging.error(f"Error during compression: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Video compression inference")
    parser.add_argument("--input_video", type=str, required=True, help="Input video path")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('inference.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        Path(args.output_dir).mkdir(exist_ok=True)
        output_path, metrics = compress_video(args)
        print(f"\nSuccessfully compressed video to: {output_path}")
    except Exception as e:
        print(f"Failed to compress video: {e}")
        exit(1)

if __name__ == "__main__":
    main()
