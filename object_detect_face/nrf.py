BROCKER_ADDRESS = 'localhost'

OBJECT_DETECT_TOPIC = "/object"

ROBOT_STATUS = "/status"

NRF24_ADDRESS = 0xF0F0F0F0D2

#from RF24 import *
import time
# from config import *

class NRF24L01:
    def __init__(self):
        self.radio = RF24(22, 0)

        self.radio.begin()
        self.radio.enableDynamicPayloads()
        self.radio.setRetries(5,15)
        self.radio.enableAckPayload()
        self.radio.setDataRate(RF24_250KBPS)

        self.radio.openWritingPipe(NRF24_ADDRESS)

        self.radio.powerUp()
    
    def write(self, data):
        self.radio.write(data)



from pyrf24 import RF24, RF24_PA_MAX, RF24_250KBPS




class NRF24L01:
    def __init__(self):
        
        self.radio = RF24(22, 0)
        self.radio.begin()
        self.radio.setDataRate(RF24_250KBPS)
        self.radio.openWritingPipe(0xF0F0F0F0D2)
        self.radio.stopListening()
    
    def write(self, data):
        self.radio.write(data)
        time.sleep(0.01)


trashbin = False

rf24 = NRF24L01()

def button_press():
    global trashbin
    trashbin = not trashbin
    if trashbin:
        for _ in range(3):
            #rf24.write(b'11') #trash ID = 1; Open command = 1
            rf24.write(b'21') #trash ID = 2; Open command = 1
            #rf24.write(b'31') #trash ID = 3; Open command = 1
            time.sleep(0.1)
    else:
        for _ in range(3):
            #rf24.write(b'10') #trash ID = 1; Close command = 0
            rf24.write(b'20') #trash ID = 2; Close command = 0
            #rf24.write(b'30') #trash ID = 3; Close command = 0
            time.sleep(0.1)

while True:
	time.sleep(1)
	rf24.write(b'10') #
	rf24.write(b'20') #
	rf24.write(b'30') #
	rf24.write(b'40') #
	time.sleep(1)
	rf24.write(b'11') #
	rf24.write(b'21') #
	rf24.write(b'31') #
	rf24.write(b'41') #
	time.sleep(1)
	rf24.write(b'12') #
	rf24.write(b'22') #
	rf24.write(b'32') #
	rf24.write(b'42') #
	time.sleep(1)
	rf24.write(b'13') #
	rf24.write(b'23') #
	rf24.write(b'33') #
	rf24.write(b'43') #
    	#button_press()
