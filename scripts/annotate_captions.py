"""One caption per AsyncOpenAI call, with bounded concurrency and durable resume records."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAIError

from mm_sae.data.caption_annotations import CaptionAnnotation, validate_and_locate
from mm_sae.io import atomic_json, file_lock, sha256
from mm_sae.progress import ProgressReporter, progress_task, stage_progress


def stable_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def source_key(row: dict) -> str:
    fields = {k: row[k] for k in ["split", "caption_id", "image_id", "original", "objects"]}
    return hashlib.sha256(stable_json(fields).encode()).hexdigest()


def request_options(args) -> dict[str, Any]:
    options: dict[str, Any] = {
        "model": args.model,
        "n": 1,
        "max_completion_tokens": args.max_completion_tokens,
        "store": False,
    }
    if args.reasoning_effort is not None:
        options["reasoning_effort"] = args.reasoning_effort
        if args.reasoning_effort == "none":
            options.update(temperature=0, top_p=1)
    else:
        options.update(temperature=0, top_p=1, seed=0, frequency_penalty=0, presence_penalty=0)
    return options


async def annotate(args):
    prompt = (args.spec / "prompt.txt").read_text()
    inputs = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if len({(r["split"], r["caption_id"]) for r in inputs}) != len(inputs):
        raise ValueError("Duplicate caption identifiers in input")
    key = dotenv_values(args.env_file).get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is missing from the specified environment file")
    options = request_options(args)
    settings = {
        **options,
        "concurrency": args.concurrency,
        "captions_per_request": 1,
        "max_attempts": args.max_attempts,
    }
    signature = {
        "input_sha256": sha256(args.input),
        "prompt_sha256": sha256(args.spec / "prompt.txt"),
        "schema": CaptionAnnotation.model_json_schema(),
        "settings": settings,
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    signature_file = args.work_dir / "request-definition.json"
    if signature_file.exists() and json.loads(signature_file.read_text()) != signature:
        raise ValueError("Work directory contains a different request definition")
    atomic_json(signature_file, signature)
    database = sqlite3.connect(args.work_dir / "responses.sqlite3")
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, result TEXT, raw TEXT)")
    database.execute("CREATE TABLE IF NOT EXISTS rejected (key TEXT, parsed TEXT, reason TEXT, raw TEXT)")
    cached = {row[0] for row in database.execute("SELECT key FROM responses")}
    pending = [row for row in inputs if source_key(row) not in cached]
    successes = len(inputs) - len(pending)
    failures: list[dict] = []
    totals = {"requests": 0, "retries": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    fingerprints: set[str] = set()
    returned_models: set[str] = set()
    started = time.monotonic()
    fatal = asyncio.Event()
    semaphore = asyncio.Semaphore(args.concurrency)
    queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=args.concurrency * 2)
    client = AsyncOpenAI(api_key=key, max_retries=0, timeout=120)

    with ProgressReporter(args.work_dir, ["annotate"], [], interval=10) as reporter:
        with (
            stage_progress("annotate"),
            progress_task(
                "Annotate captions", total=len(inputs), unit="captions", initial=successes
            ) as meter,
        ):

            async def process(row):
                nonlocal successes
                last_error = {"type": "cancelled_after_fatal_error"}
                for attempt in range(args.max_attempts):
                    if fatal.is_set():
                        break
                    delay = 0.0
                    try:
                        async with semaphore:
                            totals["requests"] += 1
                            completion = await client.chat.completions.parse(
                                **options,
                                messages=[
                                    {"role": "system", "content": prompt},
                                    {
                                        "role": "user",
                                        "content": stable_json(
                                            {"caption": row["original"], "objects": row["objects"]}
                                        ),
                                    },
                                ],
                                response_format=CaptionAnnotation,
                            )
                        if completion.usage is not None:
                            totals["prompt_tokens"] += completion.usage.prompt_tokens
                            totals["completion_tokens"] += completion.usage.completion_tokens
                            details = completion.usage.prompt_tokens_details
                            totals["cached_tokens"] += (details.cached_tokens or 0) if details else 0
                        if completion.system_fingerprint:
                            fingerprints.add(completion.system_fingerprint)
                        returned_models.add(completion.model)
                        message = completion.choices[0].message
                        if message.refusal or message.parsed is None:
                            last_error = {"type": "refusal_or_missing_parsed_response"}
                            break
                        raw = completion.model_dump_json(
                            exclude={"choices": {"__all__": {"message": {"parsed"}}}}
                        )
                        try:
                            result = validate_and_locate(row, message.parsed)
                        except ValueError as exc:
                            # Invalid source spans will not improve through transport retries.
                            last_error = {"type": "source_validation", "reason": str(exc)}
                            database.execute(
                                "INSERT INTO rejected VALUES (?, ?, ?, ?)",
                                (source_key(row), message.parsed.model_dump_json(), str(exc), raw),
                            )
                            database.commit()
                            break
                        database.execute(
                            "INSERT OR REPLACE INTO responses VALUES (?, ?, ?)",
                            (source_key(row), stable_json(result), raw),
                        )
                        database.commit()
                        successes += 1
                        return
                    except APIStatusError as exc:
                        code = str(exc.code or "unknown")
                        last_error = {
                            "type": type(exc).__name__,
                            "status_code": exc.status_code,
                            "code": code,
                        }
                        if exc.status_code in {401, 403} or code in {
                            "insufficient_quota",
                            "billing_hard_limit_reached",
                        }:
                            fatal.set()
                            break
                        if exc.status_code not in {408, 409, 429} and exc.status_code < 500:
                            break
                        try:
                            delay = float(exc.response.headers.get("retry-after", "0"))
                        except ValueError:
                            delay = 0
                    except APIConnectionError as exc:
                        # Never print raw API errors, which may contain credential fragments.
                        last_error = {"type": type(exc).__name__}
                    except (OpenAIError, ValueError) as exc:
                        last_error = {"type": type(exc).__name__}
                        break
                    if attempt + 1 < args.max_attempts:
                        totals["retries"] += 1
                        await asyncio.sleep(max(delay, min(2**attempt, 30) + random.random()))
                failures.append({"split": row["split"], "caption_id": row["caption_id"], **last_error})

            async def worker():
                while True:
                    row = await queue.get()
                    if row is None:
                        queue.task_done()
                        return
                    try:
                        await process(row)
                    finally:
                        meter.update(
                            successes + len(failures), successes=successes, failures=len(failures), **totals
                        )
                        reporter.write()
                        queue.task_done()

            try:
                async with asyncio.TaskGroup() as tasks:
                    worker_count = min(args.concurrency, max(1, len(pending)))
                    for _ in range(worker_count):
                        tasks.create_task(worker())
                    for row in pending:
                        await queue.put(row)
                    for _ in range(worker_count):
                        await queue.put(None)
            finally:
                await client.close()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(".jsonl.tmp")
            with temporary.open("w") as out:
                for row in sorted(inputs, key=lambda r: (r["split"], r["caption_id"])):
                    result = database.execute(
                        "SELECT result FROM responses WHERE key=?", (source_key(row),)
                    ).fetchone()
                    if result is not None:
                        out.write(result[0] + "\n")
            os.replace(temporary, args.output)
            database.close()
            summary = {
                "finished_at": datetime.now(UTC).isoformat(),
                "annotation_source": "openai_structured_outputs",
                "input_captions": len(inputs),
                "successful_captions": successes,
                "failed_captions": len(failures),
                "failures": failures,
                "settings": settings,
                "usage_this_invocation": totals,
                "returned_models": sorted(returned_models),
                "system_fingerprints": sorted(fingerprints),
                "elapsed_seconds": time.monotonic() - started,
                "output_sha256": sha256(args.output),
                "complete": not failures,
            }
            atomic_json(args.work_dir / "summary.json", summary)
            with (args.work_dir / "invocations.jsonl").open("a") as history:
                history.write(stable_json(summary) + "\n")
            print(json.dumps(summary, indent=2), flush=True)
            if failures:
                raise RuntimeError(f"{len(failures)} captions failed; see the recorded failure list")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path("annotations/coco_captions"))
    parser.add_argument("--model", default="gpt-4o-2024-11-20")
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high"])
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_attempts < 1:
        parser.error("concurrency and max-attempts must be positive")
    if args.max_completion_tokens < 1:
        parser.error("max-completion-tokens must be positive")
    if args.model.startswith("gpt-5") and args.reasoning_effort is None:
        parser.error("GPT-5 requires an explicit --reasoning-effort")
    if args.reasoning_effort == "none" and (args.model == "gpt-5" or args.model.startswith("gpt-5-")):
        parser.error("GPT-5 does not support none; use a model that explicitly supports non-reasoning")
    with file_lock(args.work_dir / ".lock", blocking=False):
        asyncio.run(annotate(args))


if __name__ == "__main__":
    main()
