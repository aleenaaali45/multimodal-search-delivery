f=open('src/audio_ingest.py','r',encoding='utf-8') 
t=f.read() 
f.close() 
old_fn="    model = get_whisper_model()\n    print(f'[audio] Transcribing: {path.name} (this may take 10-60s on CPU)')\n    result = model.transcribe(str(path), fp16=False, verbose=False)\n    transcript = result[\"text\"].strip()\n    print(f'[audio] Transcript ({len(transcript)} chars): {transcript[:120]}{\"...\" if len(transcript) > 120 else \"\"}')\n    global _whisper_model\n    _whisper_model = None\n    import gc; gc.collect()\n    return transcript" 
new_fn="    import subprocess, sys\n    print(f'[audio] Transcribing: {path.name} (this may take 10-60s on CPU)')\n    result = subprocess.run([sys.executable, 'whisper_transcribe.py', str(path)], capture_output=True, text=True)\n    transcript = result.stdout.strip()\n    print(f'[audio] Transcript ({len(transcript)} chars): {transcript[:120]}{\"...\" if len(transcript) > 120 else \"\"}')\n    return transcript" 
t2=t.replace(old_fn, new_fn) 
f=open('src/audio_ingest.py','w',encoding='utf-8') 
f.write(t2) 
f.close() 
print('done', 'changed' if old_fn in t else 'NOT FOUND') 
