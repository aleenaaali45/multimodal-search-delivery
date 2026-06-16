import re 
f=open('src/audio_ingest.py','r',encoding='utf-8') 
t=f.read() 
f.close() 
t2=t.replace('    return transcript', '    global _whisper_model\n    _whisper_model = None\n    import gc; gc.collect()\n    return transcript', 1) 
f=open('src/audio_ingest.py','w',encoding='utf-8') 
f.write(t2) 
f.close() 
print('Patched OK') 
