import os
import google.generativeai as genai

# Point to your service account JSON (ADC)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\AARADHYA\Desktop\IIT MANDI PROJECT\gen-lang-client-0327394450-bb75c29f5afb.json"

def main():
    genai.configure()
    model = genai.GenerativeModel('models/gemini-flash-latest')
    prompt = "Provide a short, 3-step treatment plan for tomato late blight, focusing on organic methods."
    try:
        resp = model.generate_content(prompt)
        # Print text if available
        text = getattr(resp, 'text', None)
        if text:
            print('Model response:')
            print(text)
        else:
            print('Raw response object:')
            print(resp)
    except Exception as e:
        print('API Error:', e)

if __name__ == '__main__':
    main()
