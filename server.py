import socket
import pickle
import cv2
import numpy as np
from video_transmitter import VideoTransmissionSystem
from tqdm import tqdm
import struct
import argparse
from pathlib import Path

def get_video_writer_params(ext):
    codec_map = {
        '.avi': ('XVID', 'avi'),
        '.mp4': ('mp4v', 'mp4'),
        '.mkv': ('X264', 'mkv'),
        '.mov': ('MJPG', 'mov'),
        '.wmv': ('WMV2', 'wmv')
    }
    return codec_map.get(ext.lower(), ('XVID', 'avi'))

def send_chunk(client_socket, data):
    size = len(data)
    client_socket.send(struct.pack('!I', size))
    client_socket.send(data)

def start_server(video_path, model_path, port=9999, chunk_size=16, scale_factor=0.25):
    transmitter = VideoTransmissionSystem(
        model_path=model_path,
        chunk_size=chunk_size,
        scale_factor=scale_factor
    )
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', port))
    server_socket.listen(1)
    print(f"Server listening on port {port}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    client_socket, address = server_socket.accept()
    print(f"Connection from {address}")

    # Send video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ext = Path(video_path).suffix
    metadata = pickle.dumps((fps, width, height, total_frames, ext))
    send_chunk(client_socket, metadata)

    frames_buffer = []
    chunk_id = 0
    
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_buffer.append(frame)
            
            if len(frames_buffer) == chunk_size:
                compressed_frames = transmitter.compress_chunk(frames_buffer)
                chunk_data = pickle.dumps((chunk_id, compressed_frames))
                send_chunk(client_socket, chunk_data)
                
                frames_buffer = []
                chunk_id += 1
                pbar.update(chunk_size)

        # Handle remaining frames
        if frames_buffer:
            compressed_frames = transmitter.compress_chunk(frames_buffer)
            chunk_data = pickle.dumps((chunk_id, compressed_frames))
            send_chunk(client_socket, chunk_data)
            pbar.update(len(frames_buffer))

    cap.release()
    client_socket.close()
    server_socket.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video compression server')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--model', '-m', required=True, help='Path to trained model')
    parser.add_argument('--port', '-p', type=int, default=9999, help='Server port')
    parser.add_argument('--chunk-size', '-c', type=int, default=16, help='Frame chunk size')
    parser.add_argument('--scale', '-s', type=float, default=0.25, help='Scale factor')
    
    args = parser.parse_args()
    
    start_server(
        args.input,
        args.model,
        args.port,
        args.chunk_size,
        args.scale
    )
