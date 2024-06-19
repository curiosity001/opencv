import pyttsx3

text='罐笼有人'
voice=pyttsx3.init()
voice.say(text)
voice.save_to_file(text, 'sound.wav')
voice.runAndWait()
print(1111111)
# pygame.mixer.music.load('C:/yolov7/wav/1.wav')
#                             pygame.mixer.music.set_volume(0.5)
#                             pygame.mixer.music.play()