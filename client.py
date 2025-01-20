import socket
import pickle
import cv2
import numpy as np
import struct
from pathlib import Path
from tqdm import tqdm
import argparse

def get_video_writer_params(ext):
    codec_map = {
        '.avi': ('XVID', 'avi'),
        '.mp4': ('mp4v', 'mp4'),
        '.mkv': ('X264', 'mkv'),
        '.mov': ('MJPG', 'mov'),
        '.wmv': ('WMV2', 'wmv')
    }
    return codec_map.get(ext.lower(), ('XVID', 'avi'))

def receive_chunk(client_socket):
    size_data = client_socket.recv(4)
    if not size_data:
        return None
    size = struct.unpack('!I', size_data)[0]
    
    data = b''
    while len(data) < size:
        packet = client_socket.recv(min(size - len(data), 4096))
        if not packet:
            return None
        data += packet
    return pickle.loads(data)

def receive_video(host='localhost', port=9999, output_path='output.avi'):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print(f"Connected to server at {host}:{port}")
    
    # Receive metadata
    metadata = receive_chunk(client_socket)
    if metadata is None:
        raise ConnectionError("Failed to receive metadata")
        
    fps, width, height, total_frames, src_ext = metadata
    
    # Use source extension if output_path has no extension
    out_path = Path(output_path)
    if not out_path.suffix:
        out_path = out_path.with_suffix(src_ext)
    
    # Setup video writer with original width and height
    codec, _ = get_video_writer_params(out_path.suffix)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    
    with tqdm(total=total_frames) as pbar:
        while True:
            chunk_data = receive_chunk(client_socket)
            if chunk_data is None:
                break
                
            chunk_id, frames = chunk_data
            
            for frame in frames:
                # Perform super-resolution to upscale to original size
                upscaled_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
                out.write(upscaled_frame)
                pbar.update(1)
    
    out.release()
    client_socket.close()
    print(f"Video saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video compression client')
    parser.add_argument('--host', default='localhost', help='Server hostname')
    parser.add_argument('--port', '-p', type=int, default=9999, help='Server port')
    parser.add_argument('--output', '-o', required=True, help='Output video path')
        
    args = parser.parse_args()
    
    receive_video(args.host, args.port, args.output)
