import cv2
import math
import edge_tts
import mediapipe as mp
from flask import Flask, jsonify, render_template, Response
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import asyncio
import json
import threading
import time

load_dotenv()

MIN_VELOCITY = 0.015

gemini_api_key_1 = os.environ.get("GEMINI_API_KEY_1")
gemini_api_key_2 = os.environ.get("GEMINI_API_KEY_2")
gemini_api_key_3 = os.environ.get("GEMINI_API_KEY_3")
gemini_api_key_4 = os.environ.get("GEMINI_API_KEY_4")
gemini_api_key_5 = os.environ.get("GEMINI_API_KEY_5")

gemini_client_1 = genai.Client(api_key=gemini_api_key_1)
gemini_client_2 = genai.Client(api_key=gemini_api_key_2)
gemini_client_3 = genai.Client(api_key=gemini_api_key_3)
gemini_client_4 = genai.Client(api_key=gemini_api_key_4)
gemini_client_5 = genai.Client(api_key=gemini_api_key_5)


app = Flask(__name__)

global ai_model
ai_model = "gemini_1"

global latest_feedback_text, ai_running
latest_feedback_text = ""

ai_running = False

with open("instructions.txt", "r") as f:
    instructions = f.read()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    result = abs(ang)
    if result > 180.0:
        result = 360 - result
    return result

def get_frames(open):
    global right_elbow_angles, left_elbow_angles, previous_distances, shooting, right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist, hip, knee, ankle, ai_response, beginning_angle, ending_angle, ai_running, last_shot_time
    last_shot_time = 0
    camera = cv2.VideoCapture(0)
    dominant_hand = "right"
    current_frame = 0
    reset()
    while open:
        success, frame = camera.read()
        if not success:
            break
        else:
            current_frame += 1
            frame=cv2.flip(frame,1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w, _ = image.shape
            results = pose.process(image)
            
            if results.pose_landmarks:
               
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                # cv2.imshow("FixMyJumper Feed", frame)

                landmarks = results.pose_landmarks.landmark

                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


                current_right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
                current_right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
                current_right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]

                current_right_elbow_angle = calculate_angle(current_right_shoulder, current_right_elbow, current_right_wrist)
                right_elbow_angles.append(current_right_elbow_angle)

                current_left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
                current_left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
                current_left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]

                current_left_elbow_angle = calculate_angle(current_left_shoulder, current_left_elbow, current_left_wrist)
                left_elbow_angles.append(current_left_elbow_angle)

                right_arm_range = max(right_elbow_angles) - min(right_elbow_angles)
                left_arm_range = max(left_elbow_angles) - min(left_elbow_angles)

              

                if shooting:
                    right_shoulder.append([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h])
                    right_elbow.append([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h])
                    right_wrist.append([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h])
                    hip.append([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h])
                    knee.append([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h])
                    ankle.append([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h])
                    left_shoulder.append([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h])
                    left_elbow.append([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h])
                    left_wrist.append([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h])
                    
                    stats = {
                        "right_shoulder [in format (x, y)]": right_shoulder,
                        "right_elbow [in format (x, y)]": right_elbow,
                        "right_wrist [in format (x, y)]": right_wrist,
                        "left_shoulder [in format (x, y)]": left_shoulder,
                        "left_elbow [in format (x, y)]": left_elbow,
                        "left_wrist [in format (x, y)]": left_wrist,
                        "hip [in format (x, y)]": hip,
                        "knee [in format (x, y)]": knee,
                        "ankle [in format (x, y)]": ankle,
                        "dominant_hand": dominant_hand,
                        "right_elbow_angles": right_elbow_angles,
                        "left_elbow_angles": left_elbow_angles,
                        "beginning_angle": beginning_angle,
                        "ending_angle": ending_angle
                    }


                if right_arm_range > left_arm_range:
                    dominant_hand = "right"
                    beginning_angle = min(right_elbow_angles)
                    ending_angle = max(right_elbow_angles)
                else:
                    dominant_hand = "left"
                    beginning_angle = min(left_elbow_angles)
                    ending_angle = max(left_elbow_angles)
                    
                if dominant_hand == "right":
                    w_x, w_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                    s_x, s_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                    y_wrist, y_shoulder = current_right_wrist[1], current_right_shoulder[1]
                else:
                    w_x, w_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                    s_x, s_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                    y_wrist, y_shoulder = current_left_wrist[1], current_left_shoulder[1]
                print("y_shoulder: ", y_shoulder)
                print("y_wrist: ", y_wrist)

                if dominant_hand == "right":
                    current_distance = math.sqrt((w_x - s_x) ** 2 + (w_y - s_y) ** 2)
                    print("Current distance: ", current_distance)
                    if len(previous_distances) >= 2:
                        velocity1 = current_distance - previous_distances[-1]
                        velocity2 = current_distance - previous_distances[-2]
                        print("Velocity1: ", velocity1)
                        print("Velocity2: ", velocity2)
                        current_time = time.time()
                        
                        if velocity1 > 0.15 and velocity2 > 0.15 and not shooting and y_wrist < y_shoulder and (current_time - last_shot_time > 2.5):
                            shooting = True

                        elif velocity1 < 0.15 and velocity2 < 0.15 and shooting and y_wrist > y_shoulder:
                            shooting = False



                            if not ai_running:
                                ai_running = True
                                ai_response = True
                                last_shot_time = current_time
                                print("Dominant hand: ", dominant_hand)
                                print("SHOT DETECTED")
                                threading.Thread(target=run_ai_and_tts, args=(stats.copy(),), daemon=True).start()
                                reset()

                    previous_distances.append(current_distance)
                    if len(previous_distances) > 5:
                        previous_distances.pop(0)

                else:
                    
                    current_distance = math.sqrt((w_x - s_x) ** 2 + (w_y - s_y) ** 2)
                    print("Current distance: ", current_distance)

                    if len(previous_distances) >= 2:
                        
                        velocity1 = current_distance - previous_distances[-1]
                        velocity2 = current_distance - previous_distances[-2]

                        print("Velocity1: ", velocity1)
                        print("Velocity2: ", velocity2)

                        current_time = time.time()

                        if velocity1 > 0.15 and velocity2 > 15.015 and not shooting and y_wrist < y_shoulder and (current_time - last_shot_time) > 2.5:
                            shooting = True

                        elif velocity1 < 0.15 and velocity2 < 0.15 and shooting and y_wrist > y_shoulder:
                            shooting = False
                            if not ai_running:
                                ai_running = True
                                last_shot_time = current_time
                                print("SHOT DETECTED")
                                threading.Thread(target=run_ai_and_tts, args=(stats.copy(),), daemon=True).start()
                                reset()
                                
                    previous_distances.append(current_distance)
                    if len(previous_distances) > 5:
                        previous_distances.pop(0)

                    

    else:
        camera.release()

async def talk(text):
    try:
        communicate = edge_tts.Communicate(text, "en-US-AndrewNeural", rate="+10%")
        output_path = "output.mp3"
        await communicate.save(output_path)
        return output_path
    except Exception as e:
        pass

async def ai(stats):
    global ai_model
    if ai_model == "gemini_1":
        try:
            config = types.GenerateContentConfig(
                system_instruction=instructions, temperature=0.7)
            chat = gemini_client_1.aio.chats.create(model="gemini-3-flash-preview", config=config)
            response = await chat.send_message("\n\nStats:\n" +json.dumps(stats, indent=2))
            return response.text
        except Exception as e:
           print(f"Error with gemini_1: {e}")
           ai_model = "gemini_2"
    if ai_model == "gemini_2":

        try:

            config = types.GenerateContentConfig(

                system_instruction=instructions, temperature=0.7)

            chat = gemini_client_2.aio.chats.create(model="gemini-3-flash-preview", config=config)

            response = await chat.send_message("\n\nStats:\n" +json.dumps(stats, indent=2))

            return response.text

        except Exception as e:
            print(f"Error with Gemini 2: {e}")
            ai_model = "gemini_3"
    if ai_model == "gemini_3":

        try:

            config = types.GenerateContentConfig(

                system_instruction=instructions, temperature=0.7)

            chat = gemini_client_3.aio.chats.create(model="gemini-3-flash-preview", config=config)

            response = await chat.send_message("\n\nStats:\n" +json.dumps(stats, indent=2))

            return response.text

        except Exception as e:
           print(f"Error with Gemini 3: {e}")
           ai_model = "gemini_4"
    if ai_model == "gemini_4":
        try:

            config = types.GenerateContentConfig(

                system_instruction=instructions, temperature=0.7)

            chat = gemini_client_4.aio.chats.create(model="gemini-3-flash-preview", config=config)

            response = await chat.send_message("\n\nStats:\n" +json.dumps(stats, indent=2))

            return response.text
        
        except Exception as e:
            print(f"Error with Gemini 4: {e}")
            ai_model = "gemini_5"
    if ai_model == "gemini_5":
        try:

            config = types.GenerateContentConfig(

                system_instruction=instructions, temperature=0.7)

            chat = gemini_client_5.aio.chats.create(model="gemini-3-flash-preview", config=config)

            response = await chat.send_message("\n\nStats:\n" +json.dumps(stats, indent=2))

            return response.text
        
        except Exception as e:
            print(f"Error with Gemini 5: {e}")
            pass

def run_ai_and_tts(stats):
    global latest_feedback_text, ai_running

    latest_feedback_text = "Shot Detected! AI Coach is analysing your shot..."

    response = asyncio.run(ai(stats))
    if response:
        asyncio.run(talk(response))
        latest_feedback_text = response


    ai_running = False

def reset():
    global right_elbow_angles, left_elbow_angles, previous_distances, shooting, right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist, hip, knee, ankle, ai_response, beginning_angle, ending_angle
    ai_response = False
    beginning_angle = None
    ending_angle = None
    right_elbow_angles = []
    left_elbow_angles = []
    previous_distances = []
    shooting = False
    right_shoulder = []
    right_elbow = []
    right_wrist = []
    left_shoulder = []
    left_elbow = []
    left_wrist = []
    hip = []
    knee = []
    ankle = []

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(get_frames(True), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_latest_feedback')
def get_latest_feedback():
    global latest_feedback_text
    return jsonify({"feedback": latest_feedback_text})


@app.route('/get_audio')
def get_audio():
    try:
        with open("output.mp3", "rb") as f:
            audio_data = f.read()
        return Response(audio_data, mimetype="audio/mpeg")
    except FileNotFoundError:
        return jsonify({"error": "Audio file not found"}), 404

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)