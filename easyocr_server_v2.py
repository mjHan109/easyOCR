from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import io
import logging
import multiprocessing
import threading
from PIL import Image as Image
from fastapi import File, Form, UploadFile
import numpy as np
import easyocr
import torch
import concurrent.futures
import uvicorn
import os
import json
from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

# GPU 사용 여부 확인
use_gpu = torch.cuda.is_available()

# OCR readers를 저장할 전역 딕셔너리
readers = {}

class OCRProcessor:
    def __init__(self):
        self.max_workers = min(multiprocessing.cpu_count(), 8)  # GPU 메모리 고려하여 제한
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers * 2)
        self.batch_size = 4  # GPU 메모리 사용량 고려

    @staticmethod
    def process_image_batch(image_batch):
        """배치 단위 이미지 처리"""
        results = []
        try:
            for img, lang, uuid in image_batch:
                reader = readers.get(lang, readers['unknown'])
                result = reader.readtext(img)
                extracted_texts = [item[1].strip() for item in result if item[1].strip()]
                results.append({
                    "uuid": uuid,
                    "text": ' '.join(extracted_texts),
                    "success": True
                })
        except Exception as e:
            logging.error(f"배치 처리 중 오류: {str(e)}")
        return results

def initialize_readers():
    """
    서버 시작 시 모든 언어의 Reader를 미리 초기화
    """
    global readers
    logging.info("Initializing OCR readers for all languages...")
    tika_config = config_reading('tika_config.json')
    download_mode = tika_config['ocr_info']['download_mode']
    
    if download_mode:
        try:
            # 기본 언어 조합들에 대한 Reader 초기화
            logging.info("Download mode is enabled. Downloading models...")
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
            # 오프라인 모드에서 로컬 모델 사용
            logging.info("Offline mode. Using local models from ~/.EasyOCR/model...")
            model_path = os.path.expanduser("~/.EasyOCR/model")
            readers['ko'] = easyocr.Reader(['ko', 'en'], gpu=use_gpu, model_storage_directory=model_path)
            readers['en'] = easyocr.Reader(['en'], gpu=use_gpu, model_storage_directory=model_path)
            readers['ja'] = easyocr.Reader(['ja', 'en'], gpu=use_gpu, model_storage_directory=model_path)
            readers['zh'] = easyocr.Reader(['ch_sim', 'en'], gpu=use_gpu, model_storage_directory=model_path)
            readers['unknown'] = easyocr.Reader(['ko', 'en'], gpu=use_gpu, model_storage_directory=model_path)
            logging.info("Successfully initialized all OCR readers with local models")
        except Exception as e:
            logging.error(f"Error initializing readers with local models: {str(e)}")

@app.post("/ocr/")
async def perform_ocr(
    file: UploadFile = File(...),
    language: str = Form(...),
    uuid_str: str = Form(...)
):
    """OCR 수행 함수 (병렬 처리 적용)"""
    try:
        # 이미지 전처리
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)

        # 이미지 크기에 따른 분할 처리
        height, width = image_np.shape[:2]
        if height * width > 4000 * 3000:  # 큰 이미지는 분할 처리
            segments = []
            segment_height = height // 2
            for i in range(2):
                y_start = i * segment_height
                y_end = (i + 1) * segment_height if i < 1 else height
                segment = image_np[y_start:y_end, :]
                segments.append((segment, language, f"{uuid_str}_{i}"))

            # 병렬 처리
            with ProcessPoolExecutor(max_workers=1) as executor:  # 이전: max_workers=2
                futures = [executor.submit(OCRProcessor.process_image_batch, [segment]) 
                          for segment in segments]
                results = []
                for future in concurrent.futures.as_completed(futures):
                    results.extend(future.result())

            # 결과 병합
            all_texts = []
            for result in sorted(results, key=lambda x: x['uuid']):
                if result['success']:
                    all_texts.append(result['text'])

            return {
                "uuid": uuid_str,
                "text": ' '.join(all_texts),
                "success": True
            }
        else:
            # 작은 이미지는 직접 처리
            reader = readers.get(language, readers['unknown'])
            result = reader.readtext(image_np)
            extracted_texts = [item[1].strip() for item in result if item[1].strip()]
            
            return {
                "uuid": uuid_str,
                "text": ' '.join(extracted_texts),
                "success": True
            }

    except Exception as e:
        logging.error(f"OCR 처리 중 오류 발생 (UUID: {uuid_str}): {str(e)}")
        return {
            "uuid": uuid_str,
            "text": str(e),
            "success": False
        }


def get_log_level(log_level):
    log_level = log_level.upper()
    if log_level == "DEBUG":
        return logging.DEBUG
    elif log_level == "INFO":
        return logging.INFO
    elif log_level == "WARNING":
        return logging.WARNING
    elif log_level == "ERROR":
        return logging.ERROR
    elif log_level == "CRITICAL":
        return logging.CRITICAL
    else:
        return logging.INFO

def config_reading(json_file_name):
    current_directory = os.getcwd()
    config_file = os.path.join(current_directory, json_file_name)

    if os.path.isfile(config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    else:
        logging.error(f"{current_directory} - {json_file_name} 파일을 찾을 수 없습니다.")
        return None


def main():
    try:
        # tika_config.json 파일 읽기
        json_data = config_reading('tika_config.json')
        if not json_data:  # config 파일 읽기 실패 시
            raise Exception("설정 파일을 읽을 수 없습니다.")

        # OCR 서버 설정 가져오기 
        ocr_config = json_data.get('ocr_info')
        if not ocr_config:  # ocr_info 섹션이 없을 경우
            raise Exception("설정 파일에서 'ocr_info' 섹션을 찾을 수 없습니다.")

        host = "0.0.0.0"
        port = ocr_config.get('ocr_server_port')
        workers = ocr_config.get('workers', 1)
        log_level = ocr_config.get('log_level', 'info')
        log_to_file = ocr_config.get('log_to_file')
        
        # 로그 파일 경로 설정 (날짜 포함)
        log_file_path = f"logs/ocr_server_{datetime.now().strftime('%Y%m%d')}.log"
        
        # logs 디렉토리가 없으면 생성
        os.makedirs('logs', exist_ok=True)

        # 로깅 핸들러 설정
        handlers = []
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')

        # 콘솔 핸들러 추가
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
        
        # 파일 로깅이 활성화된 경우 파일 핸들러 추가
        if log_to_file:
            try:
                file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
                file_handler.setFormatter(formatter)
                handlers.append(file_handler)
            except Exception as e:
                print(f"로그 파일 생성 중 오류 발생: {str(e)}")

        # 로깅 설정
        logging.basicConfig(
            level=get_log_level(log_level),
            format='%(asctime)s - [%(levelname)s] - %(message)s',
            handlers=handlers
        )

        logger = logging.getLogger('OCR_SERVER')
        
        # 서버 시작 로그
        logger.info(f"Starting OCR server on {host}:{port}")
        logger.info(f"GPU 사용 여부: {use_gpu}")
        logger.info(f"로그 파일 경로: {log_file_path}")
        
        # Reader 초기화
        initialize_readers()
        
        # 서버 실행
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers,
            log_level=log_level.lower(),
            access_log=True
        )
    except Exception as e:
        logger = logging.getLogger('OCR_SERVER')
        logger.error(f"서버 시작 실패: {str(e)}", exc_info=True)  # 스택 트레이스 포함
        raise  # 예외를 다시 발생시켜 프로그램 종료


if __name__ == "__main__":
    main()
