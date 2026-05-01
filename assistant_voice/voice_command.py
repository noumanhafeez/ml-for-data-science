import speech_recognition as sr
import pyttsx3

# Initialize recognizer and speaker
recognizer = sr.Recognizer()
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, could not understand.")
            return None
        except sr.RequestError:
            print("API error.")
            return None

# Main loop
while True:
    text = listen()
    if text:
        text_lower = text.lower()

        if text_lower in ["exit", "quit", "stop"]:
            speak("Goodbye!")
            break

        elif "hello" in text_lower:
            speak("Hello Nouman, how can I help you?")

        else:
            speak(text)