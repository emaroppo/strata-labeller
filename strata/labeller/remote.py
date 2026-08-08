"""Asking another host to run the round.

The wire is deliberately narrow: a dataset id, a model name, params. The
caller freezes the dataset — collections and val ratio are the project's
business — and the host materialises it, picks a parent from the runs it
holds, trains, and records. Nothing about a project travels.

Standard library rather than a client library, because this is four JSON
calls and adding a dependency to the tool people install locally to save
twenty lines is a poor trade.

**A round is a long request.** Training is minutes, and this holds the
connection open for all of it. That is the simple thing that works on a
LAN; it is not the right thing across a network that drops idle
connections, or from a laptop that sleeps. Submitting a job and polling is
the shape that survives both, and it is worth doing the moment this becomes
annoying rather than before.
"""

import json
import urllib.error
import urllib.request

#: Generous, because the request is held open for a whole training run and
#: a timeout here reads as a failed round rather than as a slow one.
DEFAULT_TIMEOUT = 4 * 3600


class RemoteError(Exception):
    """A round the other host would not or could not run."""


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
            raise RemoteError(f"{self.url}{path} refused it: {detail}") from None
        except urllib.error.URLError as e:
            raise RemoteError(f"Could not reach {self.url}: {e.reason}") from None

    def models(self) -> dict[str, str]:
        """What that host can serve, which is not what this one can."""
        return self._call("/models", timeout=30).get("models", {})

    def round(self, dataset_id: int, model: str, params: dict, fresh: bool = False) -> dict:
        """Run one round there. Returns the run and its metrics."""
        return self._call(
            "/round",
            {"dataset_id": dataset_id, "model": model, "params": params, "fresh": fresh},
        )


def _detail(error: urllib.error.HTTPError) -> str:
    try:
        return json.loads(error.read()).get("detail", str(error))
    except Exception:
        return str(error)
