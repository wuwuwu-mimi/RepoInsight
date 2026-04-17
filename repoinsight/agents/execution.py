from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass(slots=True)
class ExecutionOutcome:
    """表示一次可重试任务执行后的结果。"""

    value: Any = None
    attempt_count: int = 0
    duration_ms: int = 0
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        """返回任务是否执行成功。"""
        return self.error is None

    @property
    def used_retry(self) -> bool:
        """返回任务是否实际使用过重试。"""
        return self.attempt_count > 1



def execute_with_retry(
    func,
    *args,
    retries: int = 1,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
    **kwargs,
) -> ExecutionOutcome:
    """执行一个任务，并在失败时按给定次数重试。"""
    start_time = perf_counter()
    max_attempts = max(1, retries)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            value = func(*args, **kwargs)
            duration_ms = int((perf_counter() - start_time) * 1000)
            return ExecutionOutcome(
                value=value,
                attempt_count=attempt,
                duration_ms=duration_ms,
                error=None,
            )
        except retry_exceptions as exc:  # type: ignore[misc]
            if isinstance(exc, Exception):
                last_error = exc
            else:
                raise
            if attempt >= max_attempts:
                break

    duration_ms = int((perf_counter() - start_time) * 1000)
    return ExecutionOutcome(
        value=None,
        attempt_count=max_attempts,
        duration_ms=duration_ms,
        error=last_error,
    )



def run_parallel_tasks(tasks: dict[str, Any], *, max_workers: int | None = None) -> dict[str, Any]:
    """并行执行一组互不依赖的任务，并按名称返回结果。"""
    if not tasks:
        return {}

    if len(tasks) == 1:
        name, task = next(iter(tasks.items()))
        return {name: task()}

    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers or len(tasks)) as executor:
        future_to_name = {
            executor.submit(task): name
            for name, task in tasks.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            results[name] = future.result()
    return results
