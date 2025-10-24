import google.generativeai as genai

# Uses GOOGLE_APPLICATION_CREDENTIALS or ADC
# Make sure you've activated the 'plantbot' env where google-generativeai is installed

def main():
    genai.configure()
    models = genai.list_models()
    print('Found models:')
    for m in models:
        try:
            # some model objects may be dict-like
            name = m.name if hasattr(m, 'name') else m.get('name')
        except Exception:
            name = str(m)
        print('-', name)

if __name__ == '__main__':
    main()
