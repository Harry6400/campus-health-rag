RFC 3161 timestamp proof. Verify from the project root with:
  python tools/timestamp_evidence.py verify evidence/timestamps/sha256-f71d835273b1923e-c131f8a8

The signed response timestamps manifest.json. The manifest binds each listed file by SHA-256 and byte length. Folder names and filesystem modification times are not used as time evidence.
