from __future__ import annotations

import uvicorn

from code_quality_env.server.app import app


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
