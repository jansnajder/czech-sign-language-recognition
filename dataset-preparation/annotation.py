import os
import csv
import subprocess
from moviepy.editor import VideoFileClip


def export_audio(input_path: str, output_folder: str) -> str:
    '''Export audio from video from input path.

    :param input_path: video to export audio from
    :param output_folder: folder to output the audio to
    '''
    input_name = os.path.splitext(os.path.basename(input_path))[0]
    audio_path = os.path.join(output_folder, input_name + '.mp3')

    input_clip = VideoFileClip(input_path)
    audio_clip = input_clip.audio

    audio_clip.write_audiofile(audio_path)

    input_clip.close()
    audio_clip.close()

    return audio_path


def asr_process(input_path: str, output_folder: str) -> str:
    '''Proces the audio file by the ASR.

    :param input_path: path to audio file
    :param output_folder: folder to output the results to
    '''
    input_name = os.path.splitext(os.path.basename(input_path))[0]
    asr_path = os.path.join(output_folder, input_name)
    subprocess.run(['uwebasr.bat', 'cs', asr_path, input_path])
    return asr_path + '.txt'


def clean_up(paths):
    '''Clean up unnecessary artifacts.'''
    for path in paths:
        try:
            os.remove(path)
        except OSError as ex:
            print(f'Failed to remove {path}. Reason {e}')


if __name__ == '__main__':
    input_folder = r''
    audio_folder = 'audio'
    asr_folder = 'uwebasr-output'

    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(asr_folder, exist_ok=True)

    input_names = os.listdir(input_folder)
    failed = []

    with open('output.csv', 'a', encoding="utf-8") as csv_handle:
        output_writer = csv.writer(csv_handle)

        for input_name in input_names:
            try:
                input_path = os.path.join(input_folder, input_name)
                audio_path = export_audio(input_path, audio_folder)
                asr_path = asr_process(audio_path, asr_folder)

                with open(asr_path, 'r', encoding="utf-8") as asr_handle:
                    asr_result = asr_handle.read().replace('\n', ' ').strip()
            except Exception as ex:
                failed.append(input_name + ' - ' + str(ex))
            else:
                name = os.path.basename(input_path)
                csv_row = [name, asr_result]
                output_writer.writerow(csv_row)
            finally:
                clean_up([audio_path, asr_path, asr_path.rstrip('txt') + 'json'])

    if failed:
        with open('failed.txt', 'w') as failed_log:
            failed_log.writelines(failed)

    os.rmdir(audio_folder)
    os.rmdir(asr_folder)
