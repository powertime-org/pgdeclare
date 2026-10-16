# Contributing

Thanks for helping. Issues and pull requests go to
[github.com/powertime-org/pgdeclare](https://github.com/powertime-org/pgdeclare).

- **Bugs:** include the PostgreSQL version, the smallest schema files that
  reproduce it, and the output of `pgdeclare diff` (or the traceback).
- **Features:** open an issue first if the change is large, so we can agree on
  the shape before you write it.
- **Security issues:** do not open a public issue; see [SECURITY.md](SECURITY.md).

## Development

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # create .venv with the dev tools
uv run pytest -n auto            # unit tests; integration tests skip
uv run ruff check . && uv run ruff format --check .
uv run basedpyright              # must report zero errors
```

### Integration tests

Integration tests need a PostgreSQL server and a role that can create
databases, schemas and roles. Point `TEST_DATABASE_URL` at it:

```bash
docker run --rm -d --name pgdeclare-pg -p 5432:5432 \
    -e POSTGRES_PASSWORD=postgres postgres:17
export TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
uv run pytest -n auto
docker stop pgdeclare-pg
```

Under pytest-xdist each worker gets a database of its own, so parallel runs do
not contend for the migration lock. CI runs the suite against PostgreSQL 14
to 18; if your change depends on a version, gate the test with
`require_server_version` from `tests/conftest.py`.

## Style

- Format with `ruff format`; lint with `ruff check`; type-check with
  basedpyright (standard mode).
- Every behaviour change comes with a test. Prefer an integration test that
  applies the change to a real database and checks that a second diff is empty.
- Comments explain why, not what. Keep the package free of any one
  application's names; everything project-specific belongs in
  `PgDeclareConfig`.
- Add a line to `CHANGELOG.md` under an `Unreleased` heading.
