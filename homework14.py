import gtts
import speech_recognition as sr
import librosa
import soundfile as sf


def synthesize(text, lang, filename):
    '''
    Use gtts.gTTS(text=text, lang=lang) to synthesize speech,
    then write it to filename.

    @params:
    text (str) - the text you want to synthesize
    lang (str) - the language in which you want to synthesize it
    filename (str) - the filename in which it should be saved
    '''

    tts = gtts.gTTS(text=text, lang=lang)
    tts.save(filename)


def make_a_corpus(texts, languages, filenames):
    '''
    Create many speech files, and check their content using SpeechRecognition.
    The output files are created as MP3, then converted to WAV, then recognized.

    @param:
    texts - list of texts
    languages - list of language codes
    filenames - list of filenames WITHOUT extension

    @return:
    recognized_texts - list of recognized strings
    '''

    recognizer = sr.Recognizer()
    recognized_texts = []

    for text, lang, root in zip(texts, languages, filenames):

        mp3_file = root + ".mp3"
        wav_file = root + ".wav"

        # Generate MP3
        synthesize(text, lang, mp3_file)

        # Convert MP3 -> WAV
        audio, sample_rate = librosa.load(mp3_file, sr=None)
        sf.write(wav_file, audio, sample_rate)

        # Read WAV file
        with sr.AudioFile(wav_file) as source:
            audio_data = recognizer.record(source)

        # Recognize speech
        try:
            recognized = recognizer.recognize_google(
                audio_data,
                language=lang
            )
        except sr.UnknownValueError:
            recognized = ""
        except sr.RequestError:
            recognized = ""

        recognized_texts.append(recognized)

    return recognized_texts


# ==========================
# Example Usage
# ==========================

texts = [
    "Hello, how are you?",
    "Good morning.",
    "Thank you very much."
]

languages = [
    "en",
    "en",
    "en"
]

filenames = [
    "speech1",
    "speech2",
    "speech3"
]

results = make_a_corpus(texts, languages, filenames)

print("Recognized Texts:")
for r in results:
    print(r)
