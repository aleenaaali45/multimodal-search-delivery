f = open('src/audio_ingest.py', 'r', encoding='utf-8')
t = f.read()
f.close()

old = '    model = get_whisper_model()'
new = '    import subprocess, sys'

t2 = t.replace(old, new, 1)

old2 = '    result = model.transcribe(str(path), fp16=False, verbose=False)\n    transcript = result["text"].strip()'
new2 = '    res = subprocess.run([sys.executable, "whisper_transcribe.py", str(path)], capture_output=True, text=True)\n    transcript = res.stdout.strip()'

t2 = t2.replace(old2, new2, 1)

old3 = '    global _whisper_model\n    _whisper_model = None\n    import gc; gc.collect()\n'
t2 = t2.replace(old3, '', 1)

f = open('src/audio_ingest.py', 'w', encoding='utf-8')
f.write(t2)
f.close()
print('Done')