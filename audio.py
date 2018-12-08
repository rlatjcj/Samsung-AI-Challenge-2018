import pyaudio
import time
import numpy as np
import threading
from scipy.signal import butter, lfilter, freqz

########################################################
def getFFT(data,rate):
    """Given some data and rate, returns FFTfreq and FFT (half)."""
    data=data*np.hamming(len(data))
    fft=np.fft.fft(data)
    fft=np.abs(fft)
    freq=np.fft.fftfreq(len(fft),1.0/rate)
    return freq[:int(len(freq)/2)],fft[:int(len(fft)/2)]


def butter_bandpass(lowcut, highcut, rate, order=5):
    nyq = 0.5 * rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, rate, order=5, delay=None):
    b, a = butter_bandpass(lowcut, highcut, rate, order=order)
    y = lfilter(b, a, data)
    return y

def getBPF(data,rate,lowcut=500.0,highcut=1250.0,order=5):
    """Given some data and rate, returns results of BPF."""
    data=data*np.hamming(len(data))
    start = time.time()
    if type(lowcut) == float:
        y = butter_bandpass_filter(data, lowcut, highcut, rate, order=order)
        delay = time.time()-start
        return y, delay

    else:
        for i in range(len(lowcut)):
            if i == 0:
                y = butter_bandpass_filter(data, lowcut[i], highcut[i], rate, order=order[i])
            else:
                y += butter_bandpass_filter(data, lowcut[i], highcut[i], rate, order=order[i])

        delay = time.time()-start
        return y, delay

########################################################

class AUDIO():
    def __init__(self, device=None, rate=None, updatesPerSecond=100, low_cut=500.0, high_cut=1250.0, order=5):
        """set the initial value."""
        self.p = pyaudio.PyAudio()
        self.chunk =4096 # gets replaced automatically
        self.updatesPerSecond = updatesPerSecond
        self.chunksRead = 0
        self.device = device
        self.rate = rate
        self.cnt = 0
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.order = order
        self.q = []

        self.delay = 0

    ### SYSTEM TESTS
    def valid_low_rate(self,device):
        """set the rate to the lowest supported audio rate."""
        for testrate in [44100]:
            if self.valid_test(device,testrate):
                return testrate
        print("SOMETHING'S WRONG! I can't figure out how to use DEV",device)
        return None

    def valid_test(self,device,rate=44100):
        """given a device ID and a rate, return TRUE/False if it's valid."""
        try:
            self.info=self.p.get_device_info_by_index(device)
            if not self.info["maxInputChannels"]>0:
                return False
            stream=self.p.open(format=pyaudio.paInt16,channels=1,
               input_device_index=device,frames_per_buffer=self.chunk,
               rate=int(self.info["defaultSampleRate"]),input=True)
            stream.close()
            return True
        except:
            return False

    def valid_input_devices(self):
        """
        See which devices can be opened for microphone input.
        call this when no PyAudio object is loaded.
        """
        mics=[]
        for device in range(self.p.get_device_count()):
            if self.valid_test(device):
                mics.append(device)
        if len(mics)==0:
            print("no microphone devices found!")
        else:
            print("found %d microphone devices: %s"%(len(mics),mics))
        return mics

    def initiate(self):
        """run this after changing settings (like rate) before recording"""
        if self.device is None:
            self.device=self.valid_input_devices()[0] #pick the first one
        if self.rate is None:
            self.rate=self.valid_low_rate(self.device)
        self.chunk = int(self.rate/self.updatesPerSecond) # hold one tenth of a second in memory
        if not self.valid_test(self.device,self.rate):
            print("guessing a valid microphone device/rate...")
            self.device=self.valid_input_devices()[0] #pick the first one
            self.rate=self.valid_low_rate(self.device)
        self.datax=np.arange(self.chunk)/float(self.rate)
        msg='recording from "%s" '%self.info["name"]
        msg+='(device %d) '%self.device
        msg+='at %d Hz'%self.rate
        print(msg)

    def close(self):
        """gently detach from things."""
        print(" -- sending stream termination command...")
        self.keepRecording=False #the threads should self-close
        while(self.t.isAlive()): #wait for all threads to close
            time.sleep(.1)
        self.stream.stop_stream()
        self.p.terminate()

    ### STREAM HANDLING

    def stream_readchunk(self):
        """reads some audio and re-launches itself"""
        try:
            self.data = np.fromstring(self.stream.read(self.chunk),dtype=np.int16)
            self.bpf, self.delay = getBPF(self.data,self.rate,self.low_cut,self.high_cut,self.order)
            delay_array = np.zeros((int(self.delay/(1/44100)/2)))
            self.bpf = np.concatenate((self.bpf[len(delay_array):], delay_array))
            # print(self.bpf[-(len(delay_array)+10):])
            self.stream.write((-self.bpf).astype(np.int16), self.chunk)

            self.fftx, self.fft = getFFT(self.data ,self.rate)
            self.nc_fftx, self.nc_fft = getFFT(self.data-self.bpf ,self.rate)

            self.update_q(self.data)


        except Exception as E:
            print(" -- exception! terminating...")
            print(E,"\n"*5)
            self.keepRecording=False
        if self.keepRecording:
            self.stream_thread_new()
        else:
            self.stream.close()
            self.p.terminate()
            print(" -- stream STOPPED")
        self.chunksRead+=1

    def stream_thread_new(self):
        self.t=threading.Thread(target=self.stream_readchunk)
        self.t.start()

    def stream_start(self):
        """adds data to self.data until termination signal"""
        self.initiate()
        print(" -- starting stream")
        self.keepRecording=True # set this to False later to terminate stream
        self.data=None # will fill up with threaded recording data
        self.fft=None
        self.nc_fft=None
        self.bpf=None
        self.dataFiltered=None #same
        self.stream=self.p.open(format=pyaudio.paInt16,channels=1,
                      rate=self.rate,input=True,output=True,frames_per_buffer=self.chunk)
        self.stream_thread_new()

    def update_q(self, input):
        if len(self.q) >= 200:
            self.q.pop()
            self.q.insert(0, input)
        else:
            self.q.insert(0, input)

    def pop_all_q(self):
        np_arr = np.array(self.q).reshape(-1)
        return np_arr

if __name__ == '__main__':
    ear=AUDIO(updatesPerSecond=10) # optinoally set sample rate here
    ear.stream_start() #goes forever
    lastRead=ear.chunksRead
    while True:
        while lastRead==ear.chunksRead:
            time.sleep(.001)
        print(ear.chunksRead,len(ear.data))
        lastRead=ear.chunksRead
    print("DONE")
