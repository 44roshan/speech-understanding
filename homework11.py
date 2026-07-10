import speech_recognition as sr
from deep_translator import GoogleTranslator


def transcribe_and_translate(filename):
    '''
    Transcribe Japanese speech from a WAV file and translate it to English.

    @params:
    filename (str) - Path to the WAV file

    @returns:
    japanese_text (str)
    english_text (str)
    '''

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)

        # Speech Recognition (Japanese)
        japanese_text = recognizer.recognize_google(audio, language="ja-JP")

        # Translate to English
        english_text = GoogleTranslator(
            source='ja', target='en').translate(japanese_text)

        return japanese_text, english_text

    except sr.UnknownValueError:
        return "", "Speech could not be recognized."

    except sr.RequestError as e:
        return "", f"Speech Recognition Error: {e}"

    except Exception as e:
        return "", f"Error: {e}"


# Example usage
if __name__ == "__main__":
    filename = "speech.wav"      # Replace with your WAV file

    japanese, english = transcribe_and_translate(filename)

    print("Japanese Text:")
    print(japanese)

    print("\nEnglish Translation:")
    print(english)
