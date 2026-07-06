"""FastAPI REST server.

Run:
    uvicorn src.api:app --reload --port 8000
"""

import re
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.features import extract_features
from src.parser import parse_resume
from src.ranking import rank_candidates, score_candidate
from src.utils.config import get as cfg_get
from src.utils.exceptions import FileTooLargeError, ResumeRankerError
from src.utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="Resume Ranker API",
    description="Parse resumes, extract features, score against job descriptions.",
    version="0.1.0",
)

# The client-supplied filename is untrusted. It's only ever used to sniff a
# file extension for the temp file (never as a path), and even that is
# checked against this strict allowlist first — a crafted filename like
# "../../etc/passwd" or one embedding path separators/null bytes in its
# "suffix" cannot influence where the temp file is written.
_SAFE_SUFFIX_RE = re.compile(r"^\.[A-Za-z0-9]{1,10}$")

# Read in chunks and enforce the size cap *while* streaming, not after
# buffering an arbitrarily large upload to disk first — belt-and-suspenders
# alongside `src.parser`'s own post-write size check.
_UPLOAD_CHUNK_BYTES = 1024 * 1024


class ScoreResponse(BaseModel):
    profile: dict
    score: dict


class RankedCandidate(BaseModel):
    rank: int
    file: str
    score: dict


async def _save_upload(file: UploadFile) -> str:
    raw_suffix = Path(file.filename or "").suffix
    suffix = raw_suffix if _SAFE_SUFFIX_RE.match(raw_suffix) else ""

    max_bytes = int(cfg_get("parser.max_file_size_mb", 10) * 1024 * 1024)
    written = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
            written += len(chunk)
            if written > max_bytes:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise FileTooLargeError(
                    f"Upload exceeds the {cfg_get('parser.max_file_size_mb', 10)} MB limit"
                )
            tmp.write(chunk)
        return tmp.name


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/parse", summary="Parse a single resume into structured features")
async def parse_endpoint(file: UploadFile = File(...)) -> dict:
    try:
        tmp_path = await _save_upload(file)
        text = parse_resume(tmp_path)
    except ResumeRankerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    profile = extract_features(text)
    return asdict(profile)


@app.post("/score", response_model=ScoreResponse, summary="Score a resume against a JD")
async def score_endpoint(
    file: UploadFile = File(...),
    jd_text: str = Form(...),
) -> ScoreResponse:
    try:
        tmp_path = await _save_upload(file)
        profile = extract_features(parse_resume(tmp_path))
    except ResumeRankerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    score = score_candidate(profile, jd_text)
    return ScoreResponse(profile=asdict(profile), score=asdict(score))


@app.post(
    "/rank",
    response_model=list[RankedCandidate],
    summary="Rank a batch of resumes against a single JD",
)
async def rank_endpoint(
    files: list[UploadFile] = File(...),
    jd_text: str = Form(...),
    top_k: int | None = Form(default=None),
) -> list[RankedCandidate]:
    """Batch counterpart to /score. Resumes that fail to parse are skipped
    (with a warning logged) rather than failing the whole request — the
    same "never crash a batch over one bad file" behaviour as the CLI."""
    names, profiles = [], []
    for file in files:
        try:
            tmp_path = await _save_upload(file)
            profiles.append(extract_features(parse_resume(tmp_path)))
        except ResumeRankerError as exc:
            log.warning(f"Skipping {file.filename}: {exc}")
            continue
        names.append(file.filename or tmp_path)

    if not profiles:
        raise HTTPException(status_code=400, detail="No resumes could be parsed")

    ranked = rank_candidates(profiles, jd_text, top_k=top_k, keys=names)
    return [
        RankedCandidate(rank=i + 1, file=name, score=asdict(score))
        for i, (name, _profile, score) in enumerate(ranked)
    ]
