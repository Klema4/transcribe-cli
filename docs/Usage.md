# Usage

## Basic

```bash
lwt transcribe meeting.mp4
lwt t meeting.mp4
lwt transcribe call.wav -o transcript.txt
```

## Language & prompt

```bash
lwt transcribe meeting.wav --language cs --prompt "Technical API meeting"
```

## Subtitles

```bash
lwt transcribe video.mp4 --format srt -o subtitles.srt
lwt transcribe video.mp4 --format vtt -o subtitles.vtt
```

## JSON

```bash
lwt transcribe meeting.mp4 --format json -o output.json
```

## Speaker diarization

> Requires [[HuggingFace Diarization]] setup first.

```bash
lwt install diarization
lwt config set diarization.hf_token hf_YOUR_TOKEN
lwt transcribe panel.mp3 --diarize --speaker-names "Alice,Bob,Carol"
```

Output:
```
[00:01:23] Alice: Good morning, let's start the meeting.
[00:01:45] Bob: Sounds good, I have the presentation ready.
```

## Summarize & translate

Requires [[Ollama]].

```bash
lwt ollama pull llama3.2
lwt transcribe standup.mp4 --summarize
lwt transcribe meeting.en.mp3 --translate-to Czech
```

## Everything combined

```bash
lwt transcribe meeting.mp4 \
  --language en \
  --diarize \
  --speaker-names "Alice,Bob" \
  --summarize \
  -o meeting-notes.txt
```

See [[Commands]] for full reference.
