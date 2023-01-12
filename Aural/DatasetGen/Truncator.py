from pydub import AudioSegment

base_path = "Aural/Research-Tests/Stochastic/"

for i in range(6):
    sound_file = AudioSegment.from_file(f"{base_path}{i + 1}.wav")
    try:
        sound_file = sound_file[:5000]
    except:
        sound_file = sound_file
    
    sound_file.export(f"{base_path}{i+1}.wav", format="wav")