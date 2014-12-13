from pynaural.io.soundlib import *
from pynaural.signal.sounds import *
from time import sleep


sound = pinknoise(1.)

#serv = get_default_audioserver()
serv = JackServer()
serv.play(sound.atlevel(60))
sleep(1)

sound = whitenoise(1.)
serv.play(sound.atlevel(60))
sleep(1)
