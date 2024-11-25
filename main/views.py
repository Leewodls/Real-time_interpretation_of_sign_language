from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
import openai
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import os
import atexit
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from PIL import ImageFont, ImageDraw, Image



model = tf.keras.models.load_model('')

actions = ['가렵다', '가스', '가슴', '각목', '갇히다', '감금', '감전', '강남구', '강동구', '강북구', '강서구', 
           '강풍', '개', '거실', '결박', '경운기', '경찰차', '계곡', '계단', '고속도로', '고압전선', '고열', '고장', 
           '부러지다', '골절', '탈골', '공사장', '공원', '놀이터', '공장', '관악구', '광진구', '교통사고', '구급대', 
           '구급대원', '구급차', '구로구', '구해주세요', '금요일', '금천구', '급류', '기절', '기절하다', '깔리다', 
           '끓는물', '남자친구', '남편', '남학생', '납치', '낯선남자', '낯선사람', '내년', '노원구', '논', '농약', 
           '누전', '누출', '눈', '다리', '다음', '대문앞', '도둑', '절도', '도로', '도봉구', '독버섯', '독사', 
           '동대문구', '동작구', '두드러기생기다', '딸', '떨어지다', '쓰러지다', '뜨거운물', '마당', '마포구', 
           '말려주세요', '말벌', '맹견', '멧돼지', '목요일', '무너지다', '붕괴', '문틈', '밑에', '아래', '바다', 
           '반점생기다', '발', '발가락', '발목', '발작', '방망이', '밭', '배고프다', '뱀', '범람', '벼락', '병원', 
           '보건소', '보내주세요(경찰)', '보내주세요(구급차)', '복통', '부엌', '불', '불나다', '붕대', '비닐하우스', 
           '비상약', '빌라', '뼈', '사이', '산', '살충제', '살해', '서대문구', '서랍', '서울시', '서초구', '선반', 
           '선생님', '성동구', '성북구', '성폭행', '소방관', '소방차', '소화기', '소화전', '손', '손가락', '손목', 
           '송파구', '수영장', '술취한 사람', '시청', '심장마비', '아기', '아이들', '어린이', '아내', '아저씨', 
           '아줌마', '아파트', '인대', '안방', '알려주세요', '앞집', '약국', '약사', '양천구', '엘리베이터', '여자친구', 
           '여학생', '연락해주세요', '열', '열나다', '열어주세요', '엽총', '영등포구', '옆집', '옆집 아저씨', '옆집 할아버지', 
           '옆집사람', '오늘', '오른쪽', '옥상', '올해', '왼쪽', '욕실', '용산구', '우리집', '운동장', '위에', '위협', 
           '윗집', '윗집사람', '유리', '유치원', '유치원 버스', '은평구', '음식물', '응급대원', '응급처리', '의사', 
           '이물질', '이번', '이상한사람', '이웃집', '일요일', '임산부', '임신한아내', '자동차', '자살', '자상', 
           '작년', '작은방', '장난감', '장단지', '절단', '제초제', '조난', '종로구', '중구', '중랑구', '지혈대', 
           '진통제', '질식', '집', '집단폭행', '차밖', '차안', '창문', '창백하다', '체온계', '총', '추락', '축사', 
           '출산', '출혈', '피나다', '친구', '침수', '칼', '코', '택시', '토요일', '토하다', '통학버스', '트랙터', 
           '트럭', '파도', '파편', '팔', '팔꿈치', '폭발', '폭탄', '폭우', '폭행', '학교', '학생', '함몰되다', 
           '해(연)', '해독제', '해열제', '현관', '현관앞', '협박', '호흡곤란', '홍수', '화상', '화약', '화재']

seq_length = 60
origin_cap = 0

font_path = ""
font = ImageFont.truetype(font_path, 30)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
def hand_gesture(request):
    user_id = request.session.session_key
    file_path = get_unique_user_file_path(user_id)
    result_file_path = get_result_file_path(user_id)

    @atexit.register
    def cleanup():
        if os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("")
        if os.path.exists(result_file_path):
            with open(result_file_path, 'w', encoding='utf-8') as f:
                f.write("")

    def gen_frames():
        seq_data = []  
        
        previous_action = None
        start_time = 0
        current_action = None
        seq_data_right = []
        seq_data_left = []


        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        user_id = request.session.session_key
        if not user_id:
            request.session.create()
            user_id = request.session.session_key

        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks is not None:
                joint = np.zeros((126 * 2,))  # 양손의 랜드마크 데이터 (21*3*2)
                hand_count = 0  # 손의 개수를 추적

                for hand_landmarks in result.multi_hand_landmarks:
                    for j, lm in enumerate(hand_landmarks.landmark):
                        joint[hand_count * 21 * 3 + j * 3] = lm.x
                        joint[hand_count * 21 * 3 + j * 3 + 1] = lm.y
                        joint[hand_count * 21 * 3 + j * 3 + 2] = lm.z
                    hand_count += 1

                    # 양손의 랜드마크 그리기
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # 최대 두 손만 처리하도록 설정
                    if hand_count == 2:
                        break

                # 실시간 데이터 추가
                seq_data.append(joint)

                # 시퀀스 길이 유지 (최대 seq_length 프레임 저장)
                if len(seq_data) > seq_length:
                    seq_data.pop(0)

                # 시퀀스가 충분히 쌓였을 때 예측
                if len(seq_data) == seq_length:
                    input_data = np.expand_dims(np.array(seq_data), axis=0)  # 모델 입력 형식에 맞게 차원 추가
                    y_pred = model.predict(input_data).squeeze()

                    # 예측 결과에 따라 동작 결정
                    action_idx = np.argmax(y_pred)

                    # 예측 인덱스가 유효한지 확인
                    if action_idx < len(actions):
                        action = actions[action_idx]
                        confidence = y_pred[action_idx]

                        # 예측 결과 출력
                        print(f'예측된 동작: {action} ({confidence:.2f})')

                        # 같은 동작이 1초 동안 유지되었을 때만 자막 업데이트
                        if action == previous_action:
                            if time.time() - start_time >= 1:
                                current_action = action
                        else:
                            previous_action = action
                            start_time = time.time()
                            current_action = None
                    else:
                        current_action = None

                    # 화면에 예측 결과 표시
                    if current_action:
                        img_pil = Image.fromarray(img)
                        draw = ImageDraw.Draw(img_pil)
                        text = f'{current_action} ({confidence:.2f})'
                        draw.text((10, 150), text, font=font, fill=(255, 0, 0, 0))
                        img = np.array(img_pil)

            
            ret, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        cap.release()

    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def handle_prompt(request):
    if request.method == 'POST':
        user_id = request.session.session_key
        file_path = get_unique_user_file_path(user_id)
        result_file_path = get_result_file_path(user_id)
        handle_data = read_text_file(file_path)

        if handle_data:
            result = handle_data + "수어 문장으로 해석"
            result_content = get_completion(result) 
            append_to_file(result_file_path, result_content)

            # 파일을 초기화
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("")

            return JsonResponse({'prompt' : handle_data, 'response': result_content})
        else:
            return JsonResponse({'response': 'No gesture detected'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def index(request) :
    start = time.time() # 시작
    user_id = request.session.session_key
    if not user_id:
        request.session.create()
        user_id = request.session.session_key
    file_path_result = get_result_file_path(user_id)
    result_content = read_text_file(file_path_result)
    
    return render(request, 'index.html', {'result_content': result_content})

@csrf_exempt
def ready(request):
    
    start = time.time()
    if request.method == 'POST':
        # 실행할 코드
        first_setting_path = get_setting_file_path()
        first_setting_content = read_text_file(first_setting_path)
        get_completion(first_setting_content, "gpt-4-turbo")
        
        third_setting_path = get_setting_file_path('third', 'third_setting.txt')
        third_setting_content = read_text_file(third_setting_path)
        get_completion(third_setting_content)
        
        time.sleep(1)
        print(f"{time.time() - start:.4f} sec")
        return JsonResponse({"status": "success"})
        
    return render(request, 'ready.html')
    

def get_setting_file_path(folder_name = 'first', file_name = 'first_setting.txt'):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, folder_name, file_name)
    return file_path

def get_unique_user_file_path(user_id):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_dir = os.path.join(base_dir, 'data', user_id)
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, 'example.txt')
    return file_path


# 24.05.31 gpt로 해석된 문장이 저장되는 경로
def get_result_file_path(user_id):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(base_dir, 'result', user_id)
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, 'getResponse.txt')
    return file_path

def append_to_file(file_path, text):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines and lines[-1].strip() == text.strip():
                    return
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(f"{text}\n")
    except Exception as e:
        print(f"파일 쓰기 오류: {e}")
       
def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return None

def get_completion(prompt, model = "gpt-3.5-turbo"):
    try:
        print(prompt)
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        message = response.choices[0].message['content']
        print(message)
        return message
    except openai.error.OpenAIError as e:
        if e.code == 'quota_exceeded':
            return "You have exceeded your quota. Please check your OpenAI plan and billing details."
        else:
            print(f"Error occurred: {e}")
            return "There was an error processing your request."
