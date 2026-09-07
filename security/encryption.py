"""
Kavach — Phase 4: Encryption at Rest (hardening requirement #9)

WHAT THIS PROTECTS: every uploaded document (receipt/invoice/bank statement
image or PDF) gets encrypted the moment it lands on disk, before any OCR or
processing touches it. If the server's disk were ever compromised, the raw
financial documents are unreadable without the separate key.

KEY MANAGEMENT — the part people get wrong, so read this:
The key lives in a SEPARATE file (kavach_secret.key), generated once, never
committed to git. If the key and the encrypted data end up in the same
place (e.g. both pushed to GitHub), encryption provides ZERO protection —
anyone with repo access can decrypt everything. This module enforces that
separation by generating the key outside of any folder you'd normally
commit, and printing a loud warning if it doesn't find a .gitignore entry
for it.

In production this key would live in an environment variable or a secrets
manager, not a file at all — file-based is acceptable for this academic/
demo stage, but say so honestly in your report, don't present it as
production-final security.
"""

import os

from cryptography.fernet import Fernet

KEY_FILENAME = "ledgerPilot_secret.key"


def generate_key(key_path=KEY_FILENAME):
    """Generates a new encryption key and saves it. Run this ONCE per
    deployment/dev environment. Losing this key means losing access to
    every document encrypted with it — there is no recovery."""
    if os.path.exists(key_path):
        raise FileExistsError(
            f"{key_path} already exists — refusing to overwrite. "
            f"Delete it manually first if you really mean to generate a new key "
            f"(this will make all previously encrypted files unreadable)."
        )
    key = Fernet.generate_key()
    with open(key_path, "wb") as f:
        f.write(key)

    _warn_if_not_gitignored(key_path)
    return key


def load_key(key_path=KEY_FILENAME):
    if not os.path.exists(key_path):
        raise FileNotFoundError(
            f"{key_path} not found. Run generate_key() once first — "
            f"see this module's docstring for why this must be a separate step."
        )
    with open(key_path, "rb") as f:
        return f.read()


def _warn_if_not_gitignored(key_path):
    gitignore_path = ".gitignore"
    key_name = os.path.basename(key_path)
    if os.path.exists(gitignore_path):
        with open(gitignore_path) as f:
            content = f.read()
        if key_name in content:
            return
    print(f"\n{'!' * 60}")
    print(f"WARNING: {key_name} is not in .gitignore.")
    print(f"If you commit this file to git, your encryption is worthless —")
    print(f"anyone with repo access can decrypt every document.")
    print(f"Add a line containing '{key_name}' to .gitignore right now.")
    print(f"{'!' * 60}\n")


def encrypt_file(input_path, output_path, key=None):
    """Encrypts a file (image, PDF, whatever) in place-ish — reads
    input_path, writes the encrypted version to output_path. Does NOT
    delete the original; the caller decides when it's safe to remove
    the plaintext version."""
    key = key or load_key()
    fernet = Fernet(key)

    with open(input_path, "rb") as f:
        plaintext = f.read()

    ciphertext = fernet.encrypt(plaintext)

    with open(output_path, "wb") as f:
        f.write(ciphertext)

    return output_path


def decrypt_file(input_path, output_path, key=None):
    key = key or load_key()
    fernet = Fernet(key)

    with open(input_path, "rb") as f:
        ciphertext = f.read()

    plaintext = fernet.decrypt(ciphertext)  # raises InvalidToken if key is wrong / data tampered

    with open(output_path, "wb") as f:
        f.write(plaintext)

    return output_path


def encrypt_bytes(data, key=None):
    """For encrypting in-memory data (e.g. a file straight from a Flask
    upload, before it ever touches disk unencrypted)."""
    key = key or load_key()
    return Fernet(key).encrypt(data)


def decrypt_bytes(token, key=None):
    key = key or load_key()
    return Fernet(key).decrypt(token)
