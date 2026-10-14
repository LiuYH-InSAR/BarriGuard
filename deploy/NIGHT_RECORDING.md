# Night recording

Night-only switch: `night_lamp.overnight.record_local` (default false).
Production configs enable it with 60-minute cadence, 300-second capture,
3-day video retention and 7-day result retention. Existing 20:00–03:00
night schedule and 07:00 summary remain in use; daytime water detection
is unchanged.

Frames are saved on the existing 160-ms sampling grid (6.25 fps), not at
the camera's full frame rate. Each MP4 has a JSONL file with actual
monotonic capture timestamps. Replay uses those timestamps, including
gaps. The MP4 timeline alone can be shorter when the stream is interrupted.
Recording and replay failures are logged. Empty windows retain the old
failed/silent-hour reporting behavior.

After capture closes, a per-camera background task replays it into the
original NightSession. Recognition thresholds, temporal checks, hourly
union and morning rendering are unchanged. The discarded shared-processing
rewrite is not included. This does not promise equivalence with the local
experimental masked-background pipeline.

Host paths under the existing data mount:

- `data/night_offline/videos/<camera>/`: MP4 and timestamp JSONL; 3 days.
- `data/night_offline/results/<camera>/`: hourly and morning JPEG + JSON; 7 days.

Dedicated cleaners run hourly while the night adapter exists and also
on startup of each night. Deletion uses file modification time. The generic
48-hour cleaner excludes `night_offline`; other retention is unchanged.
Hourly reports stay local. Only each camera's morning report uses the
existing callback. 07:00 summarizes regions seen flashing in any recorded
hour, not proof of uninterrupted flashing all night.

Restart retains completed-hour state using the existing night persistence.
An interrupted in-progress recording is retained for debugging but is not
automatically replayed after restart. A restarted hour uses a new filename.

Deploy Python changes by rebuilding the image, then recreating with both
`docker-compose.yml` and `docker-compose.blocks.yml`. Turning `record_local`
off restores the prior live adapter and its original hourly push policy.
