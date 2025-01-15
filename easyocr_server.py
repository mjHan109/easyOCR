from fastapi import FastAPI, File, UploadFile, Form
import easyocr
# import torch
import uvicorn
import os
import logging
import json
import io
import numpy as np
from PIL import Image
import torch
from functools import lru_cache

app = FastAPI()

# GPU 사용 여부 확인
use_gpu = torch.cuda.is_available()

# 글로벌 변수로 readers 선언
readers = {}

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
        

# 사전 로드 함수
def load_custom_dict(language: str):
    """
    특정 언어의 사전 파일을 로드합니다.
    """
    dict_dir = os.path.join(os.getcwd(), "dict")  # 사전 디렉토리
    dict_path = os.path.join(dict_dir, f"{language}.txt")

    if os.path.isfile(dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())  # 단어를 세트로 로드
    else:
        # logging.warning(f"{language} 사전 파일이 {dict_path}에 없습니다. 사전을 사용하지 않습니다.")
        return set()  # 사전이 없으면 빈 세트를 반환


custom_dicts = {lang: load_custom_dict(lang) for lang in ['ko', 'en', 'ja', 'ch_sim']}


@lru_cache(maxsize=4)  # 최대 4개 언어의 리더만 캐시
def get_reader(language: str):
    """
    언어에 따른 리더를 반환하는 함수 (캐시 사용)  

    Parameters:
        language (str): 언어 코드
    
    Returns:
        easyocr.Reader: 언어에 따른 리더 객체
    
    """
    try:
        if language == 'ko':
            return easyocr.Reader(['ko', 'en'])
        elif language == 'ja':
            return easyocr.Reader(['ja', 'en'])
        elif language == 'zh':
            return easyocr.Reader(['ch_sim', 'en'])
        elif language == 'en':
            return easyocr.Reader('en')
        elif language == 'unknown':
            return easyocr.Reader(['ko', 'en'])
        else:  # 지원하지 않는 언어인 경우 기본값(ko, en)으로 설정
            return easyocr.Reader(['ko', 'en'])
    except Exception as e:
        return easyocr.Reader(['ko', 'en'])

def config_reading(json_file_name):
    current_directory = os.getcwd()
    config_file = os.path.join(current_directory, json_file_name)

    if os.path.isfile(config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    else:
        logging.error(f"{current_directory} - {json_file_name} 파일을 찾을 수 없습니다.")
        return None


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

@app.post("/ocr/")
async def perform_ocr(
    file: UploadFile = File(...),
    language: str = Form(...),
    uuid_str: str = Form(...)
):
    """
    OCR 수행 함수
    
    Parameters:
        uuid_str (str): 요청 UUID
        language (str): 언어 코드 (ko, en, ja, ch_sim, unknown)
        file (UploadFile): 이미지 파일
    
    Returns:
        dict: OCR 결과
    """
    try:
        # 파일을 바이트로 읽기
        contents = await file.read()

        # 이미지 바이트 데이터를 PIL 이미지로 변환
        image = Image.open(io.BytesIO(contents))

        # PIL 이미지를 numpy array로 변환
        image_np = np.array(image)

        # 언어에 맞는 reader 가져오기
        reader = get_reader(language)

        # OCR 수행
        result = reader.readtext(image_np)
        
        # 결과 텍스트 추출 및 공백 처리
        extracted_texts = []
        for item in result:
            text = item[1].strip()
            if text:  # 빈 문자열이 아닌 경우만 추가
                extracted_texts.append(text)
        
        result_string = ' '.join(extracted_texts)

        # UUID와 결과를 함께 반환
        return {
            "uuid": uuid_str,
            "text": result_string,
            "success": True
        }

    except Exception as e:
        logging.error(f"OCR 처리 중 오류 발생 (UUID: {uuid_str}): {str(e)}")
        # 에러 발생시 실패 응답 반환
        return {
            "uuid": uuid_str,
            "text": str(e),
            "success": False
        }

def main():
    try:
        # tika_config.json 파일 읽기
        json_data = config_reading('tika_config.json')

        # OCR 서버 설정 가져오기 
        ocr_config = json_data['ocr_info']
        host = "0.0.0.0"
        port = ocr_config['ocr_server_port']
        workers = ocr_config.get('workers', 1)
        log_level = ocr_config.get('log_level', 'info')
        log_to_file = ocr_config.get('log_to_file')
        log_file_path = ocr_config.get('log_file_path')

        # 로깅 핸들러 설정
        handlers = []
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')

        # 콘솔 핸들러 추가
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
        
        # 파일 로깅이 활성화된 경우 파일 핸들러 추가
        if log_to_file:
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        # 로깅 설정
        logging.basicConfig(
            level=get_log_level(log_level),
            format='%(asctime)s - [%(levelname)s] - %(message)s'
        )
                
        # 서버 시작 로그
        logging.info(f"Starting OCR server on {host}:{port}")
        logging.info(f"GPU 사용 여부: {use_gpu}")
        
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
        logging.error(f"서버 시작 실패: {str(e)}")


if __name__ == "__main__":
    main()
