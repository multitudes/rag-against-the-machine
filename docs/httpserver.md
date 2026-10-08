# Architecture: HTTP Server & Context Injection

This document details how the local HTTP server initializes, handles concurrent connections, manages state injection (such as `index_dir`), and inspects runtime socket metadata using standard library utilities.

---

## Overview

The RAG CLI server uses `http.server.ThreadingHTTPServer` to expose endpoints (`/health`, `/search`, `/answer`) for local consumption without requiring external web frameworks.

`ThreadingHTTPServer` spawns a separate OS thread for each incoming connection, ensuring long-running inference or retrieval calls do not block health checks or concurrent queries.

---

## Constraints of `ThreadingHTTPServer`

`ThreadingHTTPServer` inherits from Python's `socketserver.BaseServer`. It expects its second argument—the request handler—to be a callable factory that accepts **exactly three positional arguments**:

```python
HandlerCallable(request, client_address, server)

```

### The Parameter Binding Problem

Our custom handler (`_RagHandler`) requires runtime configuration—specifically `index_dir`—to execute searches against the on-disk index. Because `ThreadingHTTPServer` controls instantiation during client connections, custom parameters cannot be passed through the standard server constructor without breaking the expected signature.

---

## Context Injection with `functools.partial`

To inject `index_dir` cleanly without global variables or unreadable class-factory metaprogramming (`type(...)`), we use `functools.partial` for partial argument application.

```python
from functools import partial
from http.server import ThreadingHTTPServer

server = ThreadingHTTPServer(
    (host, port),
    partial(_RagHandler, index_dir),
)

```

### How `partial` Operates

1. `partial(_RagHandler, index_dir)` returns a callable wrapper where `index_dir` is pre-bound as the **first positional argument**.
2. When a connection arrives, `ThreadingHTTPServer` executes its factory call:

```python
factory(request, client_address, server)
```



3. `functools.partial` intercepts the execution and prepends `index_dir`, expanding the invocation to:

```python
_RagHandler(index_dir, request, client_address, server)

```

---

## Socket Address Resolution (`server_address`)

Immediately following server instantiation, we inspect the underlying socket to log active host and port parameters:

```python
bound_host, bound_port = server.server_address[:2]

```

This extraction serves two operational purposes:

* **Ephemeral Port Support (`port=0`):** Passing `0` as the port argument instructs the OS to assign any unallocated high-order TCP port. While the input parameter remains `0`, `server.server_address[1]` holds the true OS-assigned port number. This pattern prevents port collisions during automated test suite execution.
* **IPv4/IPv6 Cross-Compatibility:** Slicing the tuple with `[:2]` safely extracts `(host, port)` regardless of network family. IPv4 address structures return 2-tuples (`('127.0.0.1', 8000)`), whereas IPv6 socket structures return 4-tuples (`('::1', 8000, 0, 0)`). Slicing guarantees predictable unpacking across both stack types.

---

## Request Lifecycle & Data Flow

```text
[ Client Connection ]
         │
         ▼
[ ThreadingHTTPServer ] ── Spawns worker thread
         │
         ▼
[ factory(request, client_address, server) ]
         │
         ▼
[ functools.partial ] ── Prepends `index_dir`
         │
         ▼
[ _RagHandler.__init__(index_dir, request, client_address, server) ]
         │
         ├─► Binds self.index_dir = index_dir
         └─► Calls super().__init__(request, client_address, server)
                 │
                 ▼
     [ do_GET() / do_POST() ] ── Accesses self.index_dir for search/answer

```

### Execution Steps

1. **Initialization:** `run_server()` pre-binds `index_dir` into `_RagHandler` via `partial()`, binds the server socket, and inspects `server.server_address` to log accurate socket metadata.
2. **Dispatch:** On an incoming request, `ThreadingHTTPServer` instantiates the pre-bound `_RagHandler`.
3. **Setup:** `_RagHandler.__init__` sets `self.index_dir` on the request instance **before** invoking `super().__init__()`. This order is required because `BaseHTTPRequestHandler.__init__` triggers internal request parsing and method routing (`do_GET`/`do_POST`) immediately.
4. **Execution:** Route handlers read `self.index_dir` to load indices, evaluate cache hits, and execute query pipelines.

---

## Implementation Reference

```python
class _RagHandler(BaseHTTPRequestHandler):
    """Route GET/POST to health, search, and answer endpoints."""

    def __init__(
        self,
        index_dir: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Bind request to index directory before base request processing starts.
        """
        self.index_dir = index_dir
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        # self.index_dir is available for routing logic
        ...

def run_server(host: str, port: int, index_dir: str) -> None:
    """Serve /health, /search, and /answer endpoints until interrupted."""
    server = ThreadingHTTPServer(
        (host, port),
        partial(_RagHandler, index_dir),
    )
    bound_host, bound_port = server.server_address[:2]
    logger.info("RAG API listening on http://%s:%s", bound_host, bound_port)
    try:
        server.serve_forever()
    finally:
        server.server_close()

```