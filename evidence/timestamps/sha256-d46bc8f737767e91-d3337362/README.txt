RFC 3161 timestamp proof. Verify from the project root with:
  python tools/timestamp_evidence.py verify evidence/timestamps/sha256-d46bc8f737767e91-d3337362

The signed response timestamps manifest.json. The manifest binds each listed file by SHA-256 and byte length. Folder names and filesystem modification times are not used as time evidence.
