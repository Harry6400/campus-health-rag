RFC 3161 timestamp proof. Verify from the project root with:
  python tools/timestamp_evidence.py verify evidence/timestamps/sha256-fb4da754d3980893-cd4c8263

The signed response timestamps manifest.json. The manifest binds each listed file by SHA-256 and byte length. Folder names and filesystem modification times are not used as time evidence.
