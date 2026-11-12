"""Run a Python entrypoint with Internet socket connections blocked."""
import os, runpy, socket, sys
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
original_connect=socket.socket.connect
def offline_connect(self,address):
    if self.family in (socket.AF_INET,socket.AF_INET6):
        raise RuntimeError('Offline validation blocked network connection: '+str(address))
    return original_connect(self,address)
socket.socket.connect=offline_connect
entry=sys.argv[1]
sys.argv=sys.argv[1:]
runpy.run_path(entry,run_name='__main__')
