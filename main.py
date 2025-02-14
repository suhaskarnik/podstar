from audio import diarize

def main():
    diarize("./testdata/sample.wav", "outputs/output.rttm")

if __name__ == "__main__":
    main()
