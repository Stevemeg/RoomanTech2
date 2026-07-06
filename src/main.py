"""CLI entry point.

Usage:
    python -m src.main --resume <path> --jd <path>
    python -m src.main --resumes-dir data/resumes --jd <path> --top-k 10
"""

import argparse
from dataclasses import asdict
from pathlib import Path

from src.features import extract_features
from src.parser import parse_resume
from src.ranking import rank_candidates
from src.utils.exceptions import ResumeRankerError
from src.utils.io import read_text, save_csv, save_json
from src.utils.logger import get_logger

log = get_logger(__name__)


def process_one(resume_path: str, jd_text: str) -> dict:
    log.info(f"Parsing {resume_path}")
    text = parse_resume(resume_path)
    profile = extract_features(text)
    return {"file": resume_path, "profile": asdict(profile)}


def _load_profiles(files: list[str]) -> tuple[list[str], list]:
    """Parse every file, skipping (and logging) any that fail — one bad
    resume must never abort the whole batch."""
    ok_files, profiles = [], []
    for f in files:
        try:
            profile = extract_features(parse_resume(f))
        except ResumeRankerError as exc:
            log.warning(f"Skipping {f}: {exc}")
            continue
        ok_files.append(f)
        profiles.append(profile)
    _warn_on_duplicates(ok_files, profiles)
    return ok_files, profiles


def _warn_on_duplicates(files: list[str], profiles: list) -> list[str]:
    """Flag likely-duplicate candidates in a batch (same email, or
    byte-identical resume text) — a candidate submitted twice under two
    filenames would otherwise silently occupy two ranking slots.

    Returns the human-readable warning messages (also logged), so this is
    directly assertable in tests without depending on log-capture plumbing.
    """
    seen_emails: dict[str, str] = {}
    seen_text: dict[str, str] = {}
    messages: list[str] = []

    for file, profile in zip(files, profiles):
        email = (profile.contact or {}).get("email")
        if email:
            if email in seen_emails:
                msg = (
                    f"Duplicate candidate email '{email}': {seen_emails[email]} and {file} "
                    "will both be ranked — check whether this is the same person."
                )
                log.warning(msg)
                messages.append(msg)
            else:
                seen_emails[email] = file

        text_key = profile.raw_text.strip()
        if text_key in seen_text:
            msg = f"Duplicate resume content: {seen_text[text_key]} and {file}"
            log.warning(msg)
            messages.append(msg)
        else:
            seen_text[text_key] = file

    return messages


def _flatten_result(rank: int, file: str, score) -> dict:
    row = {"rank": rank, "file": file}
    row.update(asdict(score))
    return row


def _write_output(data, path: str) -> bool:
    """Write JSON output, converting permission/OS errors into a clear,
    single-line message instead of a raw traceback — a locked-down output
    directory shouldn't look like a pipeline bug."""
    try:
        save_json(data, path)
    except OSError as exc:
        log.error(f"Could not write {path}: {exc}")
        return False
    log.info(f"Wrote {path}")
    return True


def _write_csv_output(rows: list[dict], path: str) -> bool:
    try:
        save_csv(rows, path)
    except OSError as exc:
        log.error(f"Could not write {path}: {exc}")
        return False
    log.info(f"Wrote CSV results to {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume parser & candidate ranker")
    parser.add_argument("--resume", help="Path to a single resume")
    parser.add_argument("--resumes-dir", help="Directory of resumes for batch ranking")
    parser.add_argument("--jd", required=True, help="Path to job description text file")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", default="data/processed/results.json")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Also write ranked results to this CSV path (batch mode only)",
    )
    args = parser.parse_args()

    try:
        jd_text = read_text(args.jd)
    except OSError as exc:
        parser.error(f"Could not read JD file '{args.jd}': {exc}")
        return

    if not jd_text.strip():
        log.warning(f"JD file '{args.jd}' is empty — scoring will fall back to neutral defaults")

    if args.resume:
        result = process_one(args.resume, jd_text)
        _write_output(result, args.output)
        return

    if args.resumes_dir:
        resumes_dir = Path(args.resumes_dir)
        if not resumes_dir.is_dir():
            parser.error(f"--resumes-dir '{args.resumes_dir}' does not exist or isn't a directory")
            return

        candidate_files = [str(p) for p in resumes_dir.glob("*") if p.is_file()]
        log.info(f"Found {len(candidate_files)} files in {args.resumes_dir}")

        files, profiles = _load_profiles(candidate_files)
        log.info(f"Successfully parsed {len(files)}/{len(candidate_files)} resumes")

        ranked = rank_candidates(profiles, jd_text, top_k=args.top_k, keys=files)
        out = [
            {"rank": i + 1, "file": file, "score": asdict(score)}
            for i, (file, _profile, score) in enumerate(ranked)
        ]
        _write_output(out, args.output)

        if args.output_csv:
            csv_rows = [
                _flatten_result(i + 1, file, score)
                for i, (file, _profile, score) in enumerate(ranked)
            ]
            _write_csv_output(csv_rows, args.output_csv)
        return

    parser.error("Provide either --resume or --resumes-dir")


if __name__ == "__main__":
    main()
