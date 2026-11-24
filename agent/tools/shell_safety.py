"""Decide whether a shell command is destructive and needs the user's approval.

This is a deliberately cautious allow-by-asking filter, not a security sandbox:
anything that matches is shown to the user, who must type "y" before it runs.
False alarms (e.g. the word "rm" inside an echo) only cost one question.
"""
import re

_PATTERNS = [
    (r"\brm\b", "deletes files (rm)"),
    (r"\brmdir\b", "deletes folders (rmdir)"),
    (r"\bunlink\b", "deletes files (unlink)"),
    (r"\bshred\b", "destroys file contents (shred)"),
    (r"\btruncate\b", "cuts files (truncate)"),
    (r"\bmv\b", "moves or overwrites files (mv)"),
    (r"\bdd\b", "raw disk copy (dd)"),
    (r"\bmkfs", "formats a disk (mkfs)"),
    (r"\b(fdisk|parted|wipefs)\b", "changes disk partitions"),
    (r"\bsudo\b", "runs with administrator rights (sudo)"),
    (r"\bsu\b", "switches user (su)"),
    (r"\b(chmod|chown|chgrp)\b", "changes permissions or ownership"),
    (r"\b(kill|killall|pkill)\b", "stops processes"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "shuts down or restarts the machine"),
    (r"\bgit\s+(reset\s+--hard|clean|checkout\s+--|restore)\b", "discards git changes"),
    (r"\bgit\s+push\b.*(--force|-f\b)", "force-pushes to git"),
    (r"\bfind\b.*(-delete\b|-exec\s+rm\b)", "deletes files found by find"),
    (r"\bsed\b.*\s-i", "edits files in place (sed -i)"),
    (r"\b(pip3?|npm|apt|apt-get|brew|yum|dnf)\s+(uninstall|remove|purge)\b", "uninstalls software"),
    (r"\bdocker\s+(rm|rmi|system\s+prune|volume\s+rm)\b", "deletes docker data"),
    (r"\bdrop\s+(table|database)\b", "drops a database table"),
    (r"\|\s*(sudo\s+)?(ba|z)?sh\b", "pipes data into a shell"),
    (r":\(\)\s*\{", "looks like a fork bomb"),
]
_COMPILED = [(re.compile(p, re.I), why) for p, why in _PATTERNS]
_REDIRECT = re.compile(r"(?<![>&\d])\d?>(?![>&])\s*([^\s;&|]+)")


def destructive_reason(command):
    """Return a short reason if the command is risky, else None."""
    text = str(command)
    for pattern, why in _COMPILED:
        if pattern.search(text):
            return why
    for match in _REDIRECT.finditer(text):
        if match.group(1) not in ("/dev/null", "/dev/stderr", "/dev/stdout"):
            return f"overwrites a file with > ({match.group(1)})"
    return None
