import cv2
import math
import edge_tts
import mediapipe as mp
from flask import Flask, render_template, Response
from google import genai, types
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

gemini_api_key_1 = os.environ.get("GEMINI_API_KEY_1")
gemini_api_key_2 = os.environ.get("GEMINI_API_KEY_2")
gemini_api_key_3 = os.environ.get("GEMINI_API_KEY_3")
gemini_api_key_4 = os.environ.get("GEMINI_API_KEY_4")

gemini_client_1 = genai.Client(api_key=gemini_api_key_1)
gemini_client_2 = genai.Client(api_key=gemini_api_key_2)
gemini_client_3 = genai.Client(api_key=gemini_api_key_3)
gemini_client_4 = genai.Client(api_key=gemini_api_key_4)


app = Flask(__name__)

ai_model = "gemini_1"
history = []

with open("instructions.txt", "r") as f:
    instructions = f.read()


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


async def ai(user_input):

    if ai_model == "gemini_1":
        try:
            config = types.GenerateContentConfig(
                system_instruction=instructions, temperature=0.7)
            chat = gemini_client_1.aio.chats.create(model="gemini-3-flash-preview", config=config)
            response = await chat.send_message(user_input)
            return response.text
        except Exception as e:
           ai_model = "gemini_2"
    if ai_model == "gemini_2":

        try:

            config = types.GenerateContentConfig(

                system_instruction=instructions, temperature=0.7)

            chat = gemini_client_2.aio.chats.create(model="gemini-3-flash-preview", config=config)

            response = await chat.send_message(user_input)

            return response.text

        except Exception as e:
            ai_model = "gemini_3"
    if ai_model == "gemini_3":

        try:

            config = types.GenerateContentConfig(

                system_instruction=instructions, temperature=0.7)

            chat = gemini_client_3.aio.chats.create(model="gemini-3-flash-preview", config=config)

            response = await chat.send_message(user_input)

            return response.text

        except Exception as e:
           ai_model = "gemini_4"
    if ai_model == "gemini_4":
        try:

            config = types.GenerateContentConfig(

                system_instruction=instructions, temperature=0.7)

            chat = gemini_client_4.aio.chats.create(model="gemini-3-flash-preview", config=config)

            response = await chat.send_message(user_input)

            return response.text
        
        except Exception as e:
            return "All AI models failed. Please try again later."

def calculate_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    result = abs(ang)
    if result > 180.0:
        result = 360 - result
    return result

def get_frames(open):
    camera = cv2.VideoCapture(0)
    dominant_hand = "right"
    current_frame = 0
    shooting = False
    right_min_angle = 0
    right_max_angle = 0
    left_min_angle = 0
    left_max_angle = 0
    highest_wrist_y = 0
    frames_since_highest = 0
    right_elbow_angles = []
    left_elbow_angles = []
    while open:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame += 1
            frame=cv2.flip(frame,1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w, _ = image.shape
            results = pose.process(image)
            
            if results.pose_landmarks:
               
                mp.drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                landmarks = results.pose_landmarks.landmark

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]

                current_right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                right_elbow_angles.append(current_right_elbow_angle)

                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]

                current_left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                left_elbow_angles.append(current_left_elbow_angle)

                right_arm_range = max(right_elbow_angles) - min(right_elbow_angles)
                left_arm_range = max(left_elbow_angles) - min(left_elbow_angles)


                if right_arm_range > left_arm_range:
                    dominant_hand = "right"
                    beginning_angle = min(right_elbow_angles)
                    ending_angle = max(right_elbow_angles)
                else:
                    dominant_hand = "left"
                    beginning_angle = min(left_elbow_angles)
                    ending_angle = max(left_elbow_angles)

                    



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

