@echo off
setlocal enabledelayedexpansion

set LANG=%1
set URL=https://uwebasr.zcu.cz/api/v2/lindat/malach/%LANG%
set CONV_URL=https://uwebasr.zcu.cz/utils/v2/convert-speechcloud-json

shift
set OUTPUT_PATH=%1
shift
set INPUT_FILE=%1

set JSON_FILE=!OUTPUT_PATH!.json
set TXT_FILE=!OUTPUT_PATH!.txt

echo === Recognizing to raw JSON: !JSON_FILE!
ffmpeg -hide_banner -loglevel error -i "%INPUT_FILE%" -ar 16000 -ac 1 -q:a 1 -f mp3 - | curl --http1.1 --data-binary @- "%URL%?format=speechcloud_json" > "!JSON_FILE!"
echo === Converting to plaintext: !TXT_FILE!
curl --data-binary "@!JSON_FILE!" "%CONV_URL%?format=plaintext" > "!TXT_FILE!"