import socket
import numpy as np

HOST = '169.254.105.151'
PORT = 4000
class Client(object):
    
    def __init__(self, host, port):
        self.host = host
        self.port = port
        
    def connect_server(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.s.connect((self.host, self.port))
        
    
    def run_classification_model(self, sound_data ):
    
        print(sound_data)
        stringData = sound_data.tostring()
        self.s.send(np.array([len(stringData)]).tostring())
        self.s.send(stringData)
        
        print('send complete')
        data = self.s.recv(1024)
        print(data)
        return data
    
    def run(self):    
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.hostHOST, self.portPORT))
        
            while True:
                
                line = np.random.rand(441000) # add
                print(line)
                stringData = line.tostring()
                s.send(np.array([len(stringData)]).tostring())
                s.send(stringData)
                print('send complete')
                #if len(line): break          ## 빈 데이터를 먼저 보낸 후 루프를 탈출
                
                data = s.recv(1024)
                
                print(data)
        
    
if __name__ == '__main__':
    client = Client(HOST, PORT)
    client.run()

