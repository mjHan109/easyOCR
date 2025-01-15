from fastapi import FastAPI, File, UploadFile, Form
import easyocr
import uvicorn
import os
import logging
import json
import io
import numpy as np
from PIL import Image
import torch
from typing import Dict
import asyncio

app = FastAPI()

# GPU 설정
use_gpu = torch.cuda.is_available()
readers = {}

def config_reading(config_name):
    """
    config 파일을 읽어오는 함수
    
    Args:
        config_name (str): config 파일 이름
        
    Returns:
        dict: config 파일의 내용을 담은 딕셔너리
    """
    current_dir = os.getcwd()
    config_path = os.path.join(current_dir, config_name)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"{config_name} 파일을 찾을 수 없습니다.")
        return None
    except json.JSONDecodeError:
        logging.error(f"{config_name} 파일의 JSON 형식이 올바르지 않습니다.")
        return None

def initialize_readers():
    """서버 시작 시 모든 언어의 Reader를 미리 초기화"""
    global readers

            
    config = config_reading('config.json')
    download_mode = config['ocr_info']['download_mode']

    if download_mode:
        try:
            readers['ko'] = easyocr.Reader(['ko', 'en'], gpu=use_gpu)
            readers['en'] = easyocr.Reader(['en'], gpu=use_gpu)
            readers['ja'] = easyocr.Reader(['ja', 'en'], gpu=use_gpu)
            readers['zh'] = easyocr.Reader(['ch_sim', 'en'], gpu=use_gpu)
            readers['unknown'] = easyocr.Reader(['ko', 'en'], gpu=use_gpu)
            logging.info("Successfully initialized all OCR readers")
        except Exception as e:
            logging.error(f"Error initializing readers: {str(e)}")
    else:
        try:
            model_path = os.path.expanduser("~/.EasyOCR/model")
            readers['ko'] = easyocr.Reader(['ko', 'en'], gpu=use_gpu, model_storage_directory=model_path)
            readers['en'] = easyocr.Reader(['en'], gpu=use_gpu, model_storage_directory=model_path)
            readers['ja'] = easyocr.Reader(['ja', 'en'], gpu=use_gpu, model_storage_directory=model_path)
            readers['zh'] = easyocr.Reader(['ch_sim', 'en'], gpu=use_gpu, model_storage_directory=model_path)
            readers['unknown'] = easyocr.Reader(['ko', 'en'], gpu=use_gpu, model_storage_directory=model_path)
            logging.info("Successfully initialized all OCR readers with local models")
        except Exception as e:
            logging.error(f"Error initializing readers with local models: {str(e)}")

async def process_image(image_data: bytes) -> np.ndarray:
    """이미지 전처리 함수
    
    이미지를 그레이스케일로 변환하고 크기가 너무 큰 경우 비율을 유지하며 축소합니다.
    """
    try:
        with Image.open(io.BytesIO(image_data)) as img:
            # RGBA를 RGB로 변환
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # 그레이스케일로 변환
            img = img.convert('L')
            
            # config.json에서 최대 크기 설정 읽어오기
            config = config_reading('config.json')
            max_width = config['ocr_info'].get('max_width')
            max_height = config['ocr_info'].get('max_height')
            width, height = img.size
            
            if width > max_width and height > max_height:
                # 비율 계산
                ratio = min(max_width/width, max_height/height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return np.array(img)
    except Exception as e:
        logging.error(f"이미지 처리 중 오류 발생: {str(e)}")
        raise

@app.post("/ocr/")
async def ocr_endpoint(
    file: UploadFile = File(...),
    language: str = Form(...),
    uuid_str: str = Form(...)
):
    try:
        # 이미지 데이터 읽기
        image_data = await file.read()
        if not image_data:
            raise ValueError("Empty image data")

        # 이미지 검증 및 전처리
        try:
            # BytesIO 객체 생성 전에 이미지 데이터 유효성 검사
            if not image_data.startswith(b'\xff\xd8') and not image_data.startswith(b'\x89PNG'):
                raise ValueError("Invalid image format")

            image_stream = io.BytesIO(image_data)
            image = Image.open(image_stream)
            # 이미지 포맷 확인 및 로깅

            logging.info(f"이미지 포맷: {image.format}")

            # RGBA -> RGB 변환
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # 이미지 크기 확인 및 조정
            max_size = 1920  # 최대 크기 설정
            width, height = image.size
            
            if width and height and (width > max_size or height > max_size):
                ratio = min(max_size/width, max_size/height)
                new_size = (int(width*ratio), int(height*ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # numpy 배열로 변환
            image_array = np.array(image)
            
            # 메모리 정리
            image_stream.close()
            
        except Exception as e:
            logging.error(f"Image processing error for UUID {uuid_str}: {str(e)}")
            raise ValueError(f"Image processing error: {str(e)}")

        # OCR 처리
        try:
            if language not in readers:
                language = 'unknown'
                
            reader = readers[language]
            result = reader.readtext(image_array)
            
            # OCR 결과 텍스트 추출
            text = ' '.join([item[1] for item in result])
            
            if not text.strip():
                logging.warning(f"No text detected in image for UUID {uuid_str}")
            
            return {
                "text": text,
                "uuid": uuid_str,
                "success": True,
                "language": language
            }
            
        except Exception as e:
            logging.error(f"OCR processing error for UUID {uuid_str}: {str(e)}")
            raise ValueError(f"OCR processing error: {str(e)}")

    except Exception as e:
        error_msg = str(e)
        logging.error(f"OCR processing error for UUID {uuid_str}: {error_msg}")
        return {
            "error": error_msg,
            "uuid": uuid_str,
            "success": False
        }
        
        
@app.get("/status")
async def get_server_status():
    return {
        "status": "running",
        "gpu_available": use_gpu,
        "active_readers": list(readers.keys())
    }

def main():
    try:
        # 설정 파일 읽기
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('ocr_server.log')
            ]
        )
        
        initialize_readers()
        
        uvicorn.run(
            app,
            host=config.get('host', '0.0.0.0'),
            port=config.get('port', 9000),
            workers=config.get('workers', 1)
        )
    except Exception as e:
        logging.error(f"Server startup failed: {str(e)}")

if __name__ == "__main__":
    main()