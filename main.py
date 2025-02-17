from audio import diarize_episode
from audio.transcribe import transcribe_episode
from utils import parse_rttm
import click

@click.group()
def cli():
    pass
    # testfile = "./testdata/sample.wav"
    # transcribe(testfile, "outputs/transcript.json")
    
@cli.command()
@click.argument('input_file')
@click.argument('output_file')
def diarize(input_file: str, output_file: str):
    diarize_episode(input_file, output_file)

@cli.command()
@click.argument('input_file')
@click.argument('output_file')
def transcribe(input_file: str, output_file: str):
    transcribe_episode(input_file, output_file)



if __name__ == "__main__":
    cli()
