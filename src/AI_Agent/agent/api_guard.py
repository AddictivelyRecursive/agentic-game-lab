"""Stop experiments when the provider is unavailable; never count parser errors."""
import requests


class APIUnavailableError(RuntimeError):
    """Fatal batch interruption after provider failures."""


class GuardedClient:
    def __init__(self, client, max_consecutive_failures=3):
        self.client = client
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures = 0

    def generate(self, **kwargs):
        try:
            result = self.client.generate(**kwargs)
        except requests.RequestException as exc:
            self.consecutive_failures += 1
            response = getattr(exc, "response", None)
            status = response.status_code if response is not None else None
            if status in (401, 402, 403) or self.consecutive_failures >= self.max_consecutive_failures:
                raise APIUnavailableError(
                    f"Batch stopped: provider HTTP {status or 'unavailable'}; "
                    f"{self.consecutive_failures} consecutive API failure(s). "
                    "Check provider access, credits, and connectivity before rerunning."
                ) from exc
            raise
        self.consecutive_failures = 0
        return result
