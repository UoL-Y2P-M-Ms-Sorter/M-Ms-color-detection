import speech_recognition as sr
import time
from queue import Queue
import sys
import os

listen_recognizer = sr.Recognizer()
process_recognizer = sr.Recognizer()

audios_to_process = Queue()

state = 0.5


def callback(recognizer, audio_data):
    if audio_data:
        audios_to_process.put(audio_data)


def listen():
    global state
    source = sr.Microphone()
    stop = listen_recognizer.listen_in_background(source, callback, 3)
    return stop


def process_thread_func(stop_listening):
    global state
    while True:
        if audios_to_process.empty():
            time.sleep(2)
            continue
        audio = audios_to_process.get()
        if audio:
            try:
                sys.stdout = open(os.devnull, 'w')
                text = process_recognizer.recognize_google(audio)
                sys.stdout = sys.__stdout__
                if 'start' in text:
                    stop_listening()
                    return 1
                if 'stop' in text:
                    stop_listening()
                    return 0
            except sr.UnknownValueError or sr.RequestError:
                pass


def voice_rec():
    return process_thread_func(listen())


if __name__ == '__main__':
    print(voice_rec())
