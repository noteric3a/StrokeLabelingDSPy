import asyncio
import json
import urllib.error
import urllib.request
from typing import Any, Dict

from Code.config import MODEL_NAME, OLLAMA_URL, REQUEST_TIMEOUT_SECONDS, NUM_PREDICT
import Code.config as cfg

# Optional config value. If NUM_CTX is missing, default to 8192.
NUM_CTX = getattr(cfg, "NUM_CTX", 8192)


def ollama_generate_sync(
    prompt: str,
    schema: Dict[str, Any],
    case_id: str,
    tag: str,
    model: str = MODEL_NAME,
    temperature: float = 0,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "format": schema,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
        },
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            response_text = response.read().decode("utf-8")
            result = json.loads(response_text)

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama HTTP error for case {case_id} [{tag}]: "
            f"status={e.code}, body={body[:1000]}"
        ) from e

    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama request failed for case {case_id} [{tag}]: {e}"
        ) from e

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Ollama returned non-JSON wrapper for case {case_id} [{tag}]: {e}"
        ) from e

    # Log the full Ollama wrapper response.
    with open(cfg.OLLAMA_WRAPPER_LOG, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n=== Case {case_id} | {tag} ===\n")
        log_file.write(json.dumps(result, indent=2))
        log_file.write("\n")

    output = result.get("response", "")
    if isinstance(output, str):
        output = output.strip()
    else:
        output = str(output).strip()

    done_reason = result.get("done_reason")
    prompt_eval_count = result.get("prompt_eval_count")
    eval_count = result.get("eval_count")

    # This is the exact failure you saw: response was only "{",
    # done_reason was "length", and eval_count was 1.
    if done_reason == "length":
        raise RuntimeError(
            f"Ollama stopped due to length for case {case_id} [{tag}]. "
            f"prompt_eval_count={prompt_eval_count}, eval_count={eval_count}, "
            f"num_ctx={NUM_CTX}, num_predict={NUM_PREDICT}, "
            f"partial_response={output[:200]!r}. "
            f"Fix by increasing NUM_CTX or shortening the prompt."
        )

    if not output:
        raise RuntimeError(
            f"Ollama returned no output for case {case_id} [{tag}]. "
            f"done_reason={done_reason}, prompt_eval_count={prompt_eval_count}, "
            f"eval_count={eval_count}, num_ctx={NUM_CTX}."
        )

    return output


async def ollama_generate_async(
    prompt: str,
    schema: Dict[str, Any],
    case_id: str,
    tag: str,
    semaphore: asyncio.Semaphore,
    model: str = MODEL_NAME,
    temperature: float = 0,
) -> Dict[str, Any]:
    async with semaphore:
        print(f"Running {tag} for case {case_id}")
        raw = await asyncio.to_thread(
            ollama_generate_sync,
            prompt,
            schema,
            case_id,
            tag,
            model,
            temperature,
        )

    try:
        return json.loads(raw)

    except json.JSONDecodeError as e:
        with open(cfg.BAD_JSON_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n=== Case {case_id} | {tag} ===\n")
            f.write(raw)
            f.write("\n")

        raise RuntimeError(
            f"Invalid JSON for case {case_id} [{tag}]: {e}. "
            f"Raw output starts with: {raw[:300]!r}"
        ) from e