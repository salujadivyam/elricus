# Privacy Policy for Elricus

Last updated: 9 November 2026

Elricus is a Telegram bot built by Divyam Saluja. This policy explains what data the bot handles.

## Data collected
- Telegram user ID and display name
- Text messages you send
- Voice message transcripts
- Bot replies
- Timestamp of each message

## Voice and images
- Voice notes are downloaded temporarily, transcribed locally with Whisper, then deleted. Only the transcript is kept.
- Photos are downloaded temporarily and processed locally with CLIP. Photos are deleted after you use "prompt this". Photos are not sent to third parties.
- Images from /imagine are generated locally and deleted after sending.

## How data is used
- To generate replies and keep conversation context
- To log conversations for debugging and improvement
- Data is not sold or used for advertising

## Third parties
- OpenAI receives your text and transcripts to generate replies. See OpenAI's privacy policy.
- Google receives text sent with /speak to generate audio (gTTS).
- Telegram handles all message delivery.

## Storage and retention
- Conversation memory is held in RAM and cleared on restart or with /wipe.
- Chat logs are stored in a local file (chat_logs.jsonl) on the host machine. /wipe does not delete logs.
- To request deletion of your logged data, contact the developer.

## Security
API keys are stored in environment variables. No system is fully secure, so do not send sensitive personal information.

## Children
The bot is not intended for users under 13.

## Changes
This policy may be updated. The date above shows the latest version.

## Contact
Divyam Saluja, [your email]
