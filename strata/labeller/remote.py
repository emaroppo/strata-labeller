"""Asking another host to run the round.

The wire is deliberately narrow: a dataset id, a model name, params. The
caller freezes the dataset — collections and val ratio are the project's
business — and the host materialises it, picks a parent from the runs it
holds, trains, and records. Nothing about a project travels.

Standard library rather than a client library, because this is four JSON
calls and adding a dependency to the tool people install locally to save
twenty lines is a poor trade.

**A round is submitted, not awaited.** The submission is one short request;
after it returns, the round is the host's problem. Closing the laptop,
losing wifi, walking out — none of it reaches the training, and reconnecting
means asking after a job id rather than starting again.

Polling therefore has to be harder to kill than the thing it is watching. A
network error while polling is not a failed round, it is a failed question
about a round, so it is retried rather than raised. The only fatal answers
are the host saying the job failed, or saying it never heard of it.
"""

import json
import time
import urllib.error
import urllib.request

#: Generous, because the request is held open for a whole training run and
#: a timeout here reads as a failed round rather than as a slow one.
DEFAULT_TIMEOUT = 4 * 3600


class RemoteError(Exception):
    """Something went wrong talking to the modelling host."""


class Refused(RemoteError):
    """The host answered, and its answer was no.

    Separate from a host that could not be reached, because the two want
    opposite handling: a refusal is final and retrying only produces the
    same no more often, while an unreachable host says nothing at all about
    the round it is running.
    """


class Unreachable(RemoteError):
    """The question did not arrive. Says nothing about the round."""


class Trainer:
    """A modelling host, reached over HTTP."""

    def __init__(self, url: str, token: str, timeout: int = DEFAULT_TIMEOUT):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _call(self, path: str, payload: dict | None = None, timeout: int | None = None) -> dict:
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            # The service puts its reason in the body, and that reason is the
            # whole value of the error — an unservable model says what to do
            # about it, and a bare 400 says nothing.
            detail = _detail(e)
            raise Refused(f"{self.url}{path} refused it: {detail}") from None
        except urllib.error.URLError as e:
            raise Unreachable(f"Could not reach {self.url}: {e.reason}") from None

    def models(self) -> dict[str, str]:
        """What that host can serve, which is not what this one can."""
        return self._call("/models", timeout=30).get("models", {})

    def submit(self, dataset_id: int, model: str, params: dict, fresh: bool = False) -> dict:
        """Ask for a round. Returns the job; the round runs after this returns."""
        return self._call(
            "/round",
            {"dataset_id": dataset_id, "model": model, "params": params, "fresh": fresh},
            timeout=60,
        )

    def job(self, job_id: str) -> dict:
        """How a round is getting on."""
        return self._call(f"/jobs/{job_id}", timeout=30)

    def follow(self, job_id: str, on_state=None, interval: float = 3.0, sleep=None) -> dict:
        """Poll until the round finishes, tolerating a network that does not.

        A question that fails to arrive says nothing about the round, so it
        is retried. What ends this loop is the host answering — that the job
        is done, that it failed, or that it has never heard of it.
        """
        sleep = sleep or time.sleep
        unreachable = 0
        while True:
            try:
                job = self.job(job_id)
                unreachable = 0
            except Unreachable:
                unreachable += 1
                if on_state is not None:
                    on_state({"state": "unreachable", "attempts": unreachable})
                sleep(interval)
                continue

            if on_state is not None:
                on_state(job)
            if job.get("state") in ("done", "failed"):
                return job
            sleep(interval)


def _detail(error: urllib.error.HTTPError) -> str:
    try:
        return json.loads(error.read()).get("detail", str(error))
    except Exception:
        return str(error)
