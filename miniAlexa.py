import speech_recognition as sr
import chat, pyjokes, wikipedia, datetime, pywhatkit, pyttsx3
from pywinauto import Application
import bot_eyes


listener = sr.Recognizer()
engine = pyttsx3.init()
engine.setProperty('volume', 1)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)


def talk(text):
    engine.say(text)
    engine.runAndWait()


def take_command():
    try:
        with sr.Microphone() as source:
            print('listening...')
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            
            if 'alexa' in command:
                command = command.replace('alexa', '')
                return command
            else:
                pass
    except Exception as e:
        print(e)
        pass
    return 'none'

def getReply(command):
    res = chat.chatbot_response(command)
    return res

'''def run_alexa():
    command = take_command()
    print(command)
    if 'play' in command:
        song = command.replace('play', '')
        talk('playing ' + song)
        pywhatkit.playonyt(song)
    elif 'time' in command:
        time = datetime.datetime.now().strftime('%I:%M %p')
        talk('Current time is ' + time)
    elif 'who the heck is' in command:
        person = command.replace('who the heck is', '')
        info = wikipedia.summary(person, 1)
        print(info)
        talk(info)
    #elif 'date' in command:
    #    talk('sorry, I have a headache')
    #elif 'are you single' in command:
    #    talk('I am in a relationship with wifi')
    elif 'joke' in command:
        talk(pyjokes.get_joke())
    else:
        res = getReply(command)
        talk(res)'''



talk('Starting Your Personal Assistant Alexa. Please Start the command by saying Alexa')
while True:
    #run_alexa()
    command = take_command()
    print(command)
    if 'play' in command:
        song = command.replace('play', '')
        talk('playing ' + song)
        pywhatkit.playonyt(song)
    elif 'time' in command:
        time = datetime.datetime.now().strftime('%I:%M %p')
        talk('Current time is ' + time)
    elif 'who is' in command:
        person = command.replace('who the heck is', '')
        info = wikipedia.summary(person, 1)
        print(info)
        talk(info)
    #elif 'date' in command:
    #    talk('sorry, I have a headache')
    #elif 'are you single' in command:
    #    talk('I am in a relationship with wifi')
    elif 'joke' in command:
        talk(pyjokes.get_joke())
    elif 'quit' in command:
        talk("Alright. It was nice speaking to you. Bye! Have a nice day.")
        break 
    elif 'none' in command:
        pass   
    elif 'notepad' in command:
        talk('opening notepad')
        app = Application(backend='uia').start('notepad.exe')
    else:
        res = getReply(command)
        talk(res)