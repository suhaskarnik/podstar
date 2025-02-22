from utils import hf_pipeline_to_srt, json_to_srt
import click

@click.group()
def cli():
    pass
    
@cli.command()
@click.argument('input_file')
@click.argument('output_file')
def diarize(input_file: str, output_file: str):
    from audio import diarize_episode
    diarize_episode(input_file, output_file)

@cli.command()
@click.argument('input_file')
@click.argument('output_file')
def transcribe(input_file: str, output_file: str):
    from audio.transcribe import transcribe_episode
    transcribe_episode(input_file, output_file)


@cli.command()
@click.argument('input_file')
@click.argument('output_file')
def convert(input_file: str, output_file: str):
    json_to_srt(input_file, output_file)


if __name__ == "__main__":
    cli()
