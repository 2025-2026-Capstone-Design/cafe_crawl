import json
import os
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor
from playwright.sync_api import sync_playwright

def process_apollo_item(item_value, cafe_info_ref):
    if not isinstance(item_value, dict) or '__typename' not in item_value:
        cafe_info_ref = None
        return

    type_name = item_value.get('__typename')

    match type_name:
        case 'PlaceDetailBase':
            cafe_info_ref['name'] = item_value.get('name')
            cafe_info_ref['category'] = item_value.get('category')
            cafe_info_ref['micro_review'] = item_value.get('microReviews')
            cafe_info_ref['road_address'] = item_value.get('roadAddress')
            cafe_info_ref['address'] = item_value.get('address')
            cafe_info_ref['virtual_phone_number'] = item_value.get('virtualPhone')
            cafe_info_ref['payment_info'] = item_value.get('paymentInfo')
            cafe_info_ref['convenience'] = item_value.get('conveniences')
            coordinate = item_value.get('coordinate')
            if isinstance(coordinate, dict):
                cafe_info_ref['lat'] = coordinate.get('x') # 경도
                cafe_info_ref['lon'] = coordinate.get('y') # 위도
        case 'Query':
            for key, value in item_value.items():
                if(key.startswith("placeDetail")):
                    # value는 placeDetail
                    for deep_key, deep_value in value.items():
                        if(deep_key.startswith("newBusinessHours")):
                            # 영업 시간 관련
                            if(isinstance(deep_value, list) and len(deep_value) > 0 and isinstance(deep_value[0].get("businessHours"), list)):
                                for business_hour in deep_value[0].get("businessHours", []):
                                    cafe_info_ref['business_hours'].append({
                                        "day": business_hour.get("day"),
                                        "start": business_hour.get("businessHours").get("start") if isinstance(business_hour.get("businessHours"), dict) else None,
                                        "end": business_hour.get("businessHours").get("end") if isinstance(business_hour.get("businessHours"), dict) else None,
                                        "breakHours": business_hour.get("breakHours"),
                                        "description": business_hour.get("description"),
                                        "lastOrderTimes": business_hour.get("lastOrderTimes")
                                    })
                        # 이미지 관련
                        if(deep_key.startswith("images")):
                            for image in deep_value.get("images"):
                                cafe_info_ref['image_url'].append(image.get("origin"))
                        # description
                        if(deep_key.startswith("description")):
                            cafe_info_ref['description'] = deep_value
                        # url
                        if(deep_key.startswith("homepages")):
                            cafe_info_ref['url'] = deep_value.get("repr").get("url") if isinstance(deep_value.get("repr"), dict) else None
                        # parking
                        if(deep_key.startswith("informationTab")):
                            cafe_info_ref['parking_info'] = deep_value.get("parkingInfo").get("basicParking") if isinstance(deep_value.get("parkingInfo"), dict) else None


        case 'Menu':
            menu_item = {
                'name': item_value.get('name'),
                'price': item_value.get('price'),
                'description': item_value.get('description'),
                'images': item_value.get('images')
            }
            cafe_info_ref['menu'].append(menu_item)
        case 'InformationFacilities':
            cafe_info_ref['Information_facilitie'].append(item_value.get('name'))
        case _:
            # 위 case에 해당하지 않는 나머지 모든 경우 (기본값)
            pass

def extract_apollo_state(script_content):
    match = re.search(r"window\.__APOLLO_STATE__\s*=\s*({.*?});", script_content, re.DOTALL)
    if not match:
        print("스크립트에서 APOLLO_STATE 패턴을 찾지 못했습니다.")
        return None

    json_string = match.group(1)
    try:
        data = json.loads(json_string)
        print("APOLLO_STATE 파싱 성공!")
        return data
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return None


def crawl_cafe_basic_info(business_id):
    target_url = f"https://pcmap.place.naver.com/restaurant/{business_id}/home"
    cafe_info = {
        "id": business_id,
        "name": None,
        "category": None,
        "micro_review": None,
        "road_address": None,
        "address": None,
        "lat": None,
        "lon": None,
        "business_hours": [],
        "virtual_phone_number": None,
        "url": None,
        "convenience": None,
        "description": None,
        "Information_facilitie": [],
        "parking_info": None,
        "payment_info": [],
        "menu": [],
        "image_url": [],
    }

    with sync_playwright() as p:
        browser = None
        try:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            page.goto(target_url, wait_until="networkidle", timeout=30000)
            time.sleep(random.uniform(2.0, 6.0)) # 이상 탐지 방지

            js_code = """
            () => {
                const scripts = document.querySelectorAll('script');
                const searchPattern = 'var naver=typeof naver';
                for (const script of scripts) {
                    if (script.textContent && script.textContent.includes(searchPattern)) {
                        return script.textContent;
                    }
                }
                return null;
            }
            """
            script_content = page.evaluate(js_code)
            apollo_state = extract_apollo_state(script_content)

            if(not apollo_state):
                return None

            for key, value in apollo_state.items():
                process_apollo_item(value, cafe_info)
            
        except Exception as e:
            print(f"[{business_id}] 크롤링 중 심각한 오류 발생: {e}")
            cafe_info = None # All or Nothing
        finally:
            if browser and browser.is_connected():
                browser.close()
    return cafe_info

def save_cafe_info_to_json(cafe_info, directory="./data/cafe_info"):
    if not cafe_info or not cafe_info.get('id'):
        print("유효하지 않은 카페 정보입니다. 저장하지 않습니다.")
        return False

    business_id = cafe_info['id']
    filename = f"{directory}/{business_id}_info.json"
    
    try:
        if not os.path.exists(directory):
            print(f"폴더 생성: {directory}")
            os.makedirs(directory, exist_ok=True)    
        # "w" 모드: 파일이 있으면 덮어쓰고, 없으면 새로 생성
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(cafe_info, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")
        return False

def load_cafe_ids_from_jsonl(filename):
    cafe_ids = []
    
    if not os.path.exists(filename):
        print(f"오류: '{filename}' 파일을 찾을 수 없습니다.")
        return cafe_ids # 빈 리스트

    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    cafe_id = data.get('id')
                    if cafe_id:
                        cafe_ids.append(cafe_id)
                except json.JSONDecodeError:
                    print(f"경고: JSON 파싱 오류 발생. 건너뜀: {line}")
                    continue                   
    except Exception as e:
        print(f"파일 읽기 중 오류 발생: {e}")
        return cafe_ids # 빈 리스트
    
    print(f"총 {len(cafe_ids)}개의 카페 ID를 로드했습니다.")
    return cafe_ids

def process_single_cafe(business_id):
    output_dir = "./data/cafe_info"
    output_file = f"{output_dir}/{business_id}_info.json"
    
    try:
        # 0 바이트 쓰레기 파일도 크롤링 하기 위함
        if os.path.exists(output_file) and os.path.getsize(output_file) > 100:
            # print(f"SKIPPED: {business_id}")
            return
            
        basic_info = crawl_cafe_basic_info(business_id)
        
        if basic_info:
            if save_cafe_info_to_json(basic_info, directory=output_dir):
                print(f"SUCCESS: {business_id}")
            else:
                print(f"FAILED: {business_id}")
        else:
            print(f"FAILED: {business_id}")
            
    except Exception as e:
        print(f"[{business_id}] 처리 중 예외 발생: {e}")

if __name__ == "__main__":
    MAX_THREADS = 5
    CAFE_LIST_FILE = "./data/cafe_list.jsonl"
    
    cafe_ids_to_process = load_cafe_ids_from_jsonl(CAFE_LIST_FILE)
    
    if not cafe_ids_to_process:
        print("수집할 카페 ID가 없습니다. 프로그램을 종료합니다.")
    else:
        print(f"총 {len(cafe_ids_to_process)}개 ID 로드. {MAX_THREADS}개 스레드로 작업 시작...")
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            # iterable 데이터를 받아서 내부적으로 순회
            # 각 경쟁은 원자적으로 이뤄짐
            results = executor.map(process_single_cafe, cafe_ids_to_process)
                        
        print("--- 모든 작업 완료 ---")