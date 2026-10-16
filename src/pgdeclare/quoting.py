"""Quoting for names and values the engine writes into SQL.

Names come from the schema files and from the live catalog, and a role with
far fewer privileges than the migrator can create objects whose names carry a
double quote or a semicolon. Every name is therefore quoted with its embedded
quotes doubled, and every literal with its single quotes doubled. For ordinary
names the output is byte-identical to plain `"name"` and `'value'`.
"""

import re

_BARE_ROLE = re.compile(r"[a-z_][a-z0-9_$]*")


def quote_ident(name: str) -> str:
    """`name` as a double-quoted SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    """`value` as a single-quoted SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def role_ident(name: str) -> str:
    """A role name as written after TO / FROM.

    Plain lower-case names (and the PUBLIC pseudo-role) stay bare, as the
    engine has always written them; anything else is quoted.
    """
    return name if _BARE_ROLE.fullmatch(name) else quote_ident(name)


def dollar_tag(body: str) -> str:
    """A dollar-quote tag that does not occur in `body`: `$$` whenever possible.

    A name may legally contain `$$`, which would close a plain `$$` block
    early, so a longer tag is used when the body contains one.
    """
    tag = "$$"
    n = 0
    while tag in body:
        n += 1
        tag = f"$pgdeclare{n}$"
    return tag


def do_block(body: str) -> str:
    """`DO $$ <body> $$;` with a tag that cannot occur inside `body`."""
    tag = dollar_tag(body)
    return f"DO {tag} {body} {tag};"
