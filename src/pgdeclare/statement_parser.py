"""
SQL statement parser using pglast.

Parses SQL files into individual statements with metadata for retry-until-stable
execution and duplicate detection.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from pglast import split as pglast_split, parse_sql
from pglast import ast

from .config import logger


@dataclass
class ParsedStatement:
    """A single SQL statement with metadata for execution and duplicate detection."""

    raw_sql: str  # Original SQL text
    statement_type: str  # CREATE FUNCTION, CREATE TABLE, etc.
    object_type: Optional[str] = None  # TABLE, FUNCTION, TRIGGER, etc.
    object_name: Optional[str] = None  # Schema-qualified name
    signature: Optional[str] = None  # For functions: name(arg_types)
    source_file: Optional[str] = None  # Which file this came from
    line_number: Optional[int] = None  # Line in source file
    duplicate_key: Optional[str] = None  # Unique key for duplicate detection
    is_idempotent: bool = False  # CREATE OR REPLACE, IF NOT EXISTS, etc.

    def __post_init__(self):
        """Compute duplicate_key if not provided."""
        if self.duplicate_key is None and self.object_type and self.object_name:
            if self.signature:
                # Functions use signature for uniqueness
                self.duplicate_key = f"{self.object_type}:{self.signature}"
            else:
                self.duplicate_key = f"{self.object_type}:{self.object_name}"


# Statement types that should be tracked for duplicate detection
TRACKED_OBJECT_TYPES = {
    "TABLE",
    "FUNCTION",
    "VIEW",
    "INDEX",
    "TRIGGER",
    "POLICY",
    "EXTENSION",
    "SCHEMA",
    "TYPE",
    "SEQUENCE",
}

# Statement types that are additive/anonymous (not tracked for duplicates)
UNTRACKED_STATEMENT_TYPES = {
    "GRANT",
    "REVOKE",
    "ALTER",
    "DO",
    "DROP",
    "SET",
    "COMMENT",
    "INSERT",
    "UPDATE",
    "DELETE",
}


def parse_sql_file(
    file_path: Path, base_path: Optional[Path] = None
) -> list[ParsedStatement]:
    """
    Parse a SQL file into individual statements.

    Args:
        file_path: Path to the SQL file
        base_path: Optional base path for relative file reporting

    Returns:
        List of ParsedStatement objects
    """
    content = file_path.read_text()
    relative_path = str(file_path.relative_to(base_path) if base_path else file_path)
    return parse_sql_content(content, source_file=relative_path)


def parse_sql_content(
    content: str,
    source_file: Optional[str] = None,
) -> list[ParsedStatement]:
    """
    Parse SQL content into individual statements.

    Uses pglast for accurate statement splitting that handles:
    - Dollar-quoted strings ($$...$$)
    - Nested function bodies
    - Comments
    - Multi-statement strings

    Args:
        content: SQL content to parse
        source_file: Optional source file path for error reporting

    Returns:
        List of ParsedStatement objects
    """
    statements: list[ParsedStatement] = []
    line_offsets = _compute_line_offsets(content)

    try:
        # Use pglast.split() for reliable statement extraction
        # This correctly handles dollar-quoting, comments, and nested structures
        split_statements = pglast_split(content)

        # Track position in content to compute line numbers
        search_pos = 0

        for raw_sql in split_statements:
            raw_sql = raw_sql.strip()

            # Skip empty statements
            if not raw_sql or raw_sql == ";":
                continue

            # Add semicolon if missing
            if not raw_sql.rstrip().endswith(";"):
                raw_sql = raw_sql.rstrip() + ";"

            # Find position in original content for line number calculation
            # Look for a distinctive part of the statement (skip leading comments)
            sql_for_search = _strip_leading_comments(raw_sql)
            # Use first non-empty line for search
            search_text = (
                sql_for_search.split("\n")[0].strip()[:50]
                if sql_for_search
                else raw_sql[:50]
            )

            stmt_location = content.find(search_text, search_pos)
            if stmt_location == -1:
                stmt_location = search_pos
            else:
                # Advance past the body we just matched, NOT past len(raw_sql).
                # pglast < 8 keeps a statement's leading comments attached to
                # raw_sql while stmt_location points at the first non-comment
                # character, so advancing by len(raw_sql) overshoots by the
                # comment length and desynchronises every later statement's
                # line number. sql_for_search is the text actually starting at
                # stmt_location, so it is the correct stride under both.
                search_pos = stmt_location + len(sql_for_search.strip())

            # Compute line number
            line_number = _offset_to_line(stmt_location, line_offsets)

            # Parse the statement metadata
            parsed_stmt = _parse_statement_metadata(
                raw_sql, source_file, line_number, None
            )
            statements.append(parsed_stmt)

    except Exception as e:
        logger.warning(
            f"pglast parsing failed for {source_file}: {e}, falling back to simple splitting"
        )
        return _parse_sql_simple(content, source_file)

    return statements


def _compute_line_offsets(content: str) -> list[int]:
    """Compute character offset for start of each line."""
    offsets = [0]
    for i, char in enumerate(content):
        if char == "\n":
            offsets.append(i + 1)
    return offsets


def _offset_to_line(offset: int, line_offsets: list[int]) -> int:
    """Convert character offset to line number (1-based)."""
    for i, line_start in enumerate(line_offsets):
        if i + 1 < len(line_offsets) and line_offsets[i + 1] > offset:
            return i + 1
        elif i + 1 == len(line_offsets) and line_start <= offset:
            return i + 1
    return 1


def _strip_leading_comments(sql: str) -> str:
    """
    Strip leading comments and whitespace from SQL.

    Returns the SQL starting from the first non-comment line.
    """
    lines = sql.split("\n")
    result_lines = []
    found_sql = False

    for line in lines:
        stripped = line.strip()
        # Skip leading empty lines and comment-only lines
        if not found_sql:
            if not stripped or stripped.startswith("--"):
                continue
            found_sql = True
        result_lines.append(line)

    return "\n".join(result_lines)


def _parse_statement_metadata(
    raw_sql: str,
    source_file: Optional[str],
    line_number: int,
    stmt_node=None,
) -> ParsedStatement:
    """
    Extract metadata from a SQL statement.

    Uses pglast AST if available, falls back to regex parsing.
    """
    # Strip leading comments to find the actual SQL statement type
    # CRITICAL: Use this for regex matching to avoid matching patterns in comments
    sql_for_parsing = _strip_leading_comments(raw_sql)

    # Determine statement type from stripped SQL
    stmt_upper = sql_for_parsing.upper().lstrip()

    # Check for idempotent patterns
    is_idempotent = (
        "OR REPLACE" in stmt_upper
        or "IF NOT EXISTS" in stmt_upper
        or "IF EXISTS" in stmt_upper
    )

    # Parse based on statement type
    # Pass sql_for_parsing for regex matching, raw_sql is used for storage in ParsedStatement
    if stmt_upper.startswith("CREATE TABLE"):
        return _parse_create_table(
            raw_sql, sql_for_parsing, source_file, line_number, is_idempotent
        )
    elif stmt_upper.startswith("CREATE OR REPLACE FUNCTION") or stmt_upper.startswith(
        "CREATE FUNCTION"
    ):
        return _parse_create_function(
            raw_sql, sql_for_parsing, source_file, line_number, is_idempotent
        )
    elif stmt_upper.startswith("CREATE OR REPLACE VIEW") or stmt_upper.startswith(
        "CREATE VIEW"
    ):
        return _parse_create_view(
            raw_sql, sql_for_parsing, source_file, line_number, is_idempotent
        )
    elif stmt_upper.startswith("CREATE INDEX"):
        return _parse_create_index(
            raw_sql, sql_for_parsing, source_file, line_number, is_idempotent
        )
    elif stmt_upper.startswith("CREATE TRIGGER") or stmt_upper.startswith(
        "CREATE OR REPLACE TRIGGER"
    ):
        return _parse_create_trigger(
            raw_sql, sql_for_parsing, source_file, line_number, is_idempotent
        )
    elif stmt_upper.startswith("CREATE POLICY"):
        return _parse_create_policy(
            raw_sql, sql_for_parsing, source_file, line_number, is_idempotent
        )
    elif stmt_upper.startswith("CREATE TYPE"):
        return _parse_create_type(
            raw_sql, sql_for_parsing, source_file, line_number, is_idempotent
        )
    elif stmt_upper.startswith("CREATE EXTENSION"):
        return _parse_create_extension(
            raw_sql, sql_for_parsing, source_file, line_number, is_idempotent
        )
    elif stmt_upper.startswith("CREATE SCHEMA"):
        return _parse_create_schema(
            raw_sql, sql_for_parsing, source_file, line_number, is_idempotent
        )
    elif stmt_upper.startswith("CREATE SEQUENCE"):
        return _parse_create_sequence(
            raw_sql, sql_for_parsing, source_file, line_number, is_idempotent
        )
    elif stmt_upper.startswith("CREATE ROLE"):
        return _parse_create_role(
            raw_sql, sql_for_parsing, source_file, line_number, is_idempotent
        )
    else:
        # Generic statement (GRANT, REVOKE, ALTER, DO, etc.)
        stmt_type = _extract_statement_type(stmt_upper)
        return ParsedStatement(
            raw_sql=raw_sql,
            statement_type=stmt_type,
            object_type=None,
            object_name=None,
            source_file=source_file,
            line_number=line_number,
            is_idempotent=is_idempotent,
        )


def _extract_statement_type(stmt_upper: str) -> str:
    """Extract the statement type from uppercase SQL."""
    # Match first two words typically (CREATE TABLE, ALTER TABLE, etc.)
    match = re.match(r"^(\w+(?:\s+\w+)?)", stmt_upper)
    if match:
        return match.group(1)
    return "UNKNOWN"


def _extract_function_arg_types(parameters: Sequence[Any] | None) -> list[str]:
    """
    Extract normalized argument types from pglast function parameters.

    pglast hands back a tuple, or None for a no-argument function; the falsy
    check below already covered both, the annotation just said otherwise.

    Handles IN, INOUT, VARIADIC parameters (excludes OUT and TABLE parameters).
    Normalizes type names to canonical PostgreSQL forms.
    """
    if not parameters:
        return []

    types = []
    for param in parameters:
        if not isinstance(param, ast.FunctionParameter):
            continue

        # Skip output parameters (mode 'o' for OUT, 't' for TABLE)
        # pglast uses character codes: 'i'=IN, 'o'=OUT, 'b'=INOUT, 'v'=VARIADIC, 't'=TABLE
        if hasattr(param, "mode") and param.mode in ("t", "o", ord("t"), ord("o")):
            continue

        if param.argType and param.argType.names:
            # Extract type name parts
            type_parts = []
            for name_node in param.argType.names:
                if hasattr(name_node, "sval"):
                    type_parts.append(name_node.sval)

            # Normalize type names based on schema:
            # - pg_catalog: built-in types, already canonical lowercase from pglast
            # - public: strip prefix ONLY if type name is all lowercase
            #   (case-sensitive types like "MyType" need the public. prefix preserved)
            # - other schemas: keep schema prefix, preserve case
            if not type_parts:
                type_name = "unknown"
            elif type_parts[0] == "pg_catalog":
                # Built-in types: strip pg_catalog, space-join (already canonical lowercase)
                type_parts = type_parts[1:]
                type_name = " ".join(type_parts) if type_parts else "unknown"
            elif type_parts[0] == "public":
                # Public schema: strip only if type name is all lowercase
                type_name_part = type_parts[1] if len(type_parts) > 1 else "unknown"
                if type_name_part == type_name_part.lower():
                    # Lowercase type in public schema - strip public prefix
                    type_name = type_name_part
                else:
                    # Case-sensitive type - keep public prefix to distinguish
                    type_name = ".".join(type_parts)
            else:
                # Custom schema: keep schema prefix, preserve case
                type_name = ".".join(type_parts)

            # Handle array types
            if param.argType.arrayBounds:
                type_name += "[]"

            types.append(type_name)

    return types


def _parse_create_table(
    raw_sql: str,
    sql_for_parsing: str,
    source_file: Optional[str],
    line_number: int,
    is_idempotent: bool,
) -> ParsedStatement:
    """Parse CREATE TABLE statement using pglast AST."""
    object_name = None

    try:
        tree = parse_sql(sql_for_parsing)
        for raw_stmt in tree:
            stmt = raw_stmt.stmt
            if isinstance(stmt, ast.CreateStmt) and stmt.relation is not None:
                schema = stmt.relation.schemaname or "public"
                name = stmt.relation.relname
                object_name = f"{schema}.{name}"
                # Check if_not_exists from AST
                if hasattr(stmt, "if_not_exists") and stmt.if_not_exists:
                    is_idempotent = True
                break
    except Exception:
        pass  # If pglast fails, PostgreSQL would fail too

    return ParsedStatement(
        raw_sql=raw_sql,
        statement_type="CREATE TABLE",
        object_type="TABLE",
        object_name=object_name,
        source_file=source_file,
        line_number=line_number,
        is_idempotent=is_idempotent,
    )


def _parse_create_function(
    raw_sql: str,
    sql_for_parsing: str,
    source_file: Optional[str],
    line_number: int,
    is_idempotent: bool,
) -> ParsedStatement:
    """Parse CREATE FUNCTION statement using pglast AST."""
    object_name = None
    signature = None

    try:
        tree = parse_sql(sql_for_parsing)
        for raw_stmt in tree:
            stmt = raw_stmt.stmt
            if isinstance(stmt, ast.CreateFunctionStmt):
                # Extract function name from funcname list
                funcname_parts = []
                for name_node in stmt.funcname or []:
                    if hasattr(name_node, "sval"):
                        funcname_parts.append(name_node.sval)

                if len(funcname_parts) == 1:
                    object_name = f"public.{funcname_parts[0]}"
                elif len(funcname_parts) >= 2:
                    object_name = f"{funcname_parts[-2]}.{funcname_parts[-1]}"

                # Extract and normalize argument types
                arg_types = _extract_function_arg_types(stmt.parameters)
                signature = f"{object_name}({', '.join(arg_types)})"

                # Check replace from AST
                if hasattr(stmt, "replace") and stmt.replace:
                    is_idempotent = True
                break
    except Exception as e:
        # Log warning - if pglast can't parse it, PostgreSQL likely can't either
        logger.warning(f"pglast failed to parse function: {e}")

    return ParsedStatement(
        raw_sql=raw_sql,
        statement_type="CREATE FUNCTION",
        object_type="FUNCTION",
        object_name=object_name,
        signature=signature,
        source_file=source_file,
        line_number=line_number,
        is_idempotent=is_idempotent,
    )


def _parse_create_view(
    raw_sql: str,
    sql_for_parsing: str,
    source_file: Optional[str],
    line_number: int,
    is_idempotent: bool,
) -> ParsedStatement:
    """Parse CREATE VIEW statement using pglast AST."""
    object_name = None

    try:
        tree = parse_sql(sql_for_parsing)
        for raw_stmt in tree:
            stmt = raw_stmt.stmt
            if isinstance(stmt, ast.ViewStmt) and stmt.view is not None:
                schema = stmt.view.schemaname or "public"
                name = stmt.view.relname
                object_name = f"{schema}.{name}"
                # Check replace from AST
                if hasattr(stmt, "replace") and stmt.replace:
                    is_idempotent = True
                break
    except Exception:
        pass  # If pglast fails, PostgreSQL would fail too

    return ParsedStatement(
        raw_sql=raw_sql,
        statement_type="CREATE VIEW",
        object_type="VIEW",
        object_name=object_name,
        source_file=source_file,
        line_number=line_number,
        is_idempotent=is_idempotent,
    )


def _parse_create_index(
    raw_sql: str,
    sql_for_parsing: str,
    source_file: Optional[str],
    line_number: int,
    is_idempotent: bool,
) -> ParsedStatement:
    """Parse CREATE INDEX statement using pglast AST."""
    object_name = None

    try:
        tree = parse_sql(sql_for_parsing)
        for raw_stmt in tree:
            stmt = raw_stmt.stmt
            if isinstance(stmt, ast.IndexStmt):
                object_name = stmt.idxname
                # Check if_not_exists from AST
                if hasattr(stmt, "if_not_exists") and stmt.if_not_exists:
                    is_idempotent = True
                break
    except Exception:
        pass  # If pglast fails, PostgreSQL would fail too

    return ParsedStatement(
        raw_sql=raw_sql,
        statement_type="CREATE INDEX",
        object_type="INDEX",
        object_name=object_name,
        source_file=source_file,
        line_number=line_number,
        is_idempotent=is_idempotent,
    )


def _parse_create_trigger(
    raw_sql: str,
    sql_for_parsing: str,
    source_file: Optional[str],
    line_number: int,
    is_idempotent: bool,
) -> ParsedStatement:
    """Parse CREATE TRIGGER statement using pglast AST."""
    object_name = None

    try:
        tree = parse_sql(sql_for_parsing)
        for raw_stmt in tree:
            stmt = raw_stmt.stmt
            if isinstance(stmt, ast.CreateTrigStmt) and stmt.relation is not None:
                trigger_name = stmt.trigname
                # Extract table name for uniqueness
                table_schema = stmt.relation.schemaname or "public"
                table_name = stmt.relation.relname
                object_name = f"{table_schema}.{table_name}.{trigger_name}"
                # Check replace from AST
                if hasattr(stmt, "replace") and stmt.replace:
                    is_idempotent = True
                break
    except Exception:
        pass  # If pglast fails, PostgreSQL would fail too

    return ParsedStatement(
        raw_sql=raw_sql,
        statement_type="CREATE TRIGGER",
        object_type="TRIGGER",
        object_name=object_name,
        source_file=source_file,
        line_number=line_number,
        is_idempotent=is_idempotent,
    )


def _parse_create_policy(
    raw_sql: str,
    sql_for_parsing: str,
    source_file: Optional[str],
    line_number: int,
    is_idempotent: bool,
) -> ParsedStatement:
    """Parse CREATE POLICY statement using pglast AST."""
    object_name = None

    try:
        tree = parse_sql(sql_for_parsing)
        for raw_stmt in tree:
            stmt = raw_stmt.stmt
            if isinstance(stmt, ast.CreatePolicyStmt) and stmt.table is not None:
                policy_name = stmt.policy_name
                # Extract table name
                table_schema = stmt.table.schemaname or "public"
                table_name = stmt.table.relname
                object_name = f"{table_schema}.{table_name}.{policy_name}"
                break
    except Exception:
        pass  # If pglast fails, PostgreSQL would fail too

    return ParsedStatement(
        raw_sql=raw_sql,
        statement_type="CREATE POLICY",
        object_type="POLICY",
        object_name=object_name,
        source_file=source_file,
        line_number=line_number,
        is_idempotent=is_idempotent,
    )


def _parse_create_type(
    raw_sql: str,
    sql_for_parsing: str,
    source_file: Optional[str],
    line_number: int,
    is_idempotent: bool,
) -> ParsedStatement:
    """Parse CREATE TYPE statement using pglast AST."""
    object_name = None

    try:
        tree = parse_sql(sql_for_parsing)
        for raw_stmt in tree:
            stmt = raw_stmt.stmt
            # Handle both composite types and enum types
            if isinstance(stmt, ast.CompositeTypeStmt) and stmt.typevar is not None:
                # Composite type: CREATE TYPE name AS (...)
                schema = stmt.typevar.schemaname or "public"
                name = stmt.typevar.relname
                object_name = f"{schema}.{name}"
                break
            elif isinstance(stmt, ast.CreateEnumStmt):
                # Enum type: CREATE TYPE name AS ENUM (...)
                type_parts = []
                for name_node in stmt.typeName or []:
                    if hasattr(name_node, "sval"):
                        type_parts.append(name_node.sval)
                if len(type_parts) == 1:
                    object_name = f"public.{type_parts[0]}"
                elif len(type_parts) >= 2:
                    object_name = f"{type_parts[-2]}.{type_parts[-1]}"
                break
    except Exception:
        pass  # If pglast fails, PostgreSQL would fail too

    return ParsedStatement(
        raw_sql=raw_sql,
        statement_type="CREATE TYPE",
        object_type="TYPE",
        object_name=object_name,
        source_file=source_file,
        line_number=line_number,
        is_idempotent=is_idempotent,
    )


def _parse_create_extension(
    raw_sql: str,
    sql_for_parsing: str,
    source_file: Optional[str],
    line_number: int,
    is_idempotent: bool,
) -> ParsedStatement:
    """Parse CREATE EXTENSION statement using pglast AST."""
    object_name = None

    try:
        tree = parse_sql(sql_for_parsing)
        for raw_stmt in tree:
            stmt = raw_stmt.stmt
            if isinstance(stmt, ast.CreateExtensionStmt):
                object_name = stmt.extname
                # Check if_not_exists from AST
                if hasattr(stmt, "if_not_exists") and stmt.if_not_exists:
                    is_idempotent = True
                break
    except Exception:
        pass  # If pglast fails, PostgreSQL would fail too

    return ParsedStatement(
        raw_sql=raw_sql,
        statement_type="CREATE EXTENSION",
        object_type="EXTENSION",
        object_name=object_name,
        source_file=source_file,
        line_number=line_number,
        is_idempotent=is_idempotent,
    )


def _parse_create_schema(
    raw_sql: str,
    sql_for_parsing: str,
    source_file: Optional[str],
    line_number: int,
    is_idempotent: bool,
) -> ParsedStatement:
    """Parse CREATE SCHEMA statement using pglast AST."""
    object_name = None

    try:
        tree = parse_sql(sql_for_parsing)
        for raw_stmt in tree:
            stmt = raw_stmt.stmt
            if isinstance(stmt, ast.CreateSchemaStmt):
                object_name = stmt.schemaname
                # Check if_not_exists from AST
                if hasattr(stmt, "if_not_exists") and stmt.if_not_exists:
                    is_idempotent = True
                break
    except Exception:
        pass  # If pglast fails, PostgreSQL would fail too

    return ParsedStatement(
        raw_sql=raw_sql,
        statement_type="CREATE SCHEMA",
        object_type="SCHEMA",
        object_name=object_name,
        source_file=source_file,
        line_number=line_number,
        is_idempotent=is_idempotent,
    )


def _parse_create_sequence(
    raw_sql: str,
    sql_for_parsing: str,
    source_file: Optional[str],
    line_number: int,
    is_idempotent: bool,
) -> ParsedStatement:
    """Parse CREATE SEQUENCE statement using pglast AST."""
    object_name = None

    try:
        tree = parse_sql(sql_for_parsing)
        for raw_stmt in tree:
            stmt = raw_stmt.stmt
            if isinstance(stmt, ast.CreateSeqStmt) and stmt.sequence is not None:
                schema = stmt.sequence.schemaname or "public"
                name = stmt.sequence.relname
                object_name = f"{schema}.{name}"
                # Check if_not_exists from AST
                if hasattr(stmt, "if_not_exists") and stmt.if_not_exists:
                    is_idempotent = True
                break
    except Exception:
        pass  # If pglast fails, PostgreSQL would fail too

    return ParsedStatement(
        raw_sql=raw_sql,
        statement_type="CREATE SEQUENCE",
        object_type="SEQUENCE",
        object_name=object_name,
        source_file=source_file,
        line_number=line_number,
        is_idempotent=is_idempotent,
    )


def _parse_create_role(
    raw_sql: str,
    sql_for_parsing: str,
    source_file: Optional[str],
    line_number: int,
    is_idempotent: bool,
) -> ParsedStatement:
    """Parse CREATE ROLE statement using pglast AST."""
    object_name = None

    try:
        tree = parse_sql(sql_for_parsing)
        for raw_stmt in tree:
            stmt = raw_stmt.stmt
            if isinstance(stmt, ast.CreateRoleStmt):
                object_name = stmt.role
                break
    except Exception:
        pass  # If pglast fails, PostgreSQL would fail too

    return ParsedStatement(
        raw_sql=raw_sql,
        statement_type="CREATE ROLE",
        object_type="ROLE",
        object_name=object_name,
        source_file=source_file,
        line_number=line_number,
        is_idempotent=is_idempotent,
    )


def parse_role_from_sql(raw_sql: str) -> dict[str, Any] | None:
    """
    Extract role name and password from CREATE ROLE statement.

    PostgreSQL introspection cannot retrieve password hashes from pg_roles,
    so we parse passwords directly from declarative SQL to inject them
    into the introspected schema for proper diff generation.

    Args:
        raw_sql: Raw SQL text (may contain comments)

    Returns:
        Dictionary with role attributes if CREATE ROLE statement, None otherwise.
        Example: {"name": "app_user", "password": "secret", "login": True}
    """
    # Strip leading comments to find actual SQL
    sql_for_parsing = _strip_leading_comments(raw_sql)
    sql_upper = sql_for_parsing.upper().lstrip()

    if not sql_upper.startswith("CREATE ROLE"):
        return None

    # Extract role name
    name_pattern = r"CREATE\s+ROLE\s+([\"']?[\w]+[\"']?)"
    name_match = re.search(name_pattern, sql_for_parsing, re.IGNORECASE)
    if not name_match:
        return None

    role_name = name_match.group(1).strip("\"'")

    result: dict[str, Any] = {"name": role_name}

    # Extract password from PASSWORD clause
    # Handle: PASSWORD 'password' or PASSWORD 'pass''word' (escaped quotes)
    # The password is in single quotes and may contain escaped single quotes ('')
    password_pattern = r"\bPASSWORD\s+'((?:[^']|'')+)'"
    password_match = re.search(password_pattern, sql_for_parsing, re.IGNORECASE)
    if password_match:
        # Unescape doubled single quotes: '' -> '
        password = password_match.group(1).replace("''", "'")
        result["password"] = password

    # Check for LOGIN attribute
    if re.search(r"\bLOGIN\b", sql_for_parsing, re.IGNORECASE):
        result["login"] = True
    elif re.search(r"\bNOLOGIN\b", sql_for_parsing, re.IGNORECASE):
        result["login"] = False

    return result


def _parse_sql_simple(
    content: str, source_file: Optional[str]
) -> list[ParsedStatement]:
    """
    Simple SQL splitting fallback when pglast is not available.

    Uses regex-based splitting that handles basic cases but may fail on
    complex dollar-quoted strings or nested functions.
    """
    statements: list[ParsedStatement] = []

    # Split by semicolons, but be careful of dollar-quoted strings
    # This is a simplified approach - pglast is preferred
    current_stmt: list[str] = []
    in_dollar_quote = False
    dollar_tag = ""
    stmt_start_line = 1

    lines = content.split("\n")

    for line_idx, line in enumerate(lines, 1):
        # Skip empty lines and comment-only lines when not in a statement
        stripped_line = line.strip()
        if not current_stmt and (not stripped_line or stripped_line.startswith("--")):
            continue

        # Track dollar quoting
        if not in_dollar_quote:
            # Look for dollar quote start - but only on non-comment parts
            non_comment = line.split("--")[0] if "--" in line else line
            dollar_match = re.search(r"\$(\w*)\$", non_comment)
            if dollar_match:
                in_dollar_quote = True
                dollar_tag = dollar_match.group(1)
                # Check if this same line has the closing tag (single-line dollar quote)
                rest_of_line = non_comment[dollar_match.end() :]
                end_pattern = f"\\${re.escape(dollar_tag)}\\$"
                if re.search(end_pattern, rest_of_line):
                    in_dollar_quote = False
                    dollar_tag = ""
        else:
            # Look for matching dollar quote end
            end_pattern = f"\\${re.escape(dollar_tag)}\\$"
            if re.search(end_pattern, line):
                in_dollar_quote = False
                dollar_tag = ""

        # Start new statement if needed
        if not current_stmt:
            stmt_start_line = line_idx

        current_stmt.append(line)

        # Check for statement end (semicolon not in dollar quote)
        if not in_dollar_quote and ";" in line:
            # Check if the semicolon is in a comment
            non_comment = line.split("--")[0] if "--" in line else line
            if ";" in non_comment:
                # Join and process the statement
                stmt_text = "\n".join(current_stmt).strip()

                # Skip empty statements, comment-only statements, and bare semicolons
                if _is_valid_statement(stmt_text):
                    # Parse the statement metadata
                    parsed = _parse_statement_metadata(
                        stmt_text, source_file, stmt_start_line, None
                    )
                    statements.append(parsed)

                current_stmt = []

    # Handle any remaining content
    if current_stmt:
        stmt_text = "\n".join(current_stmt).strip()
        if _is_valid_statement(stmt_text):
            parsed = _parse_statement_metadata(
                stmt_text, source_file, stmt_start_line, None
            )
            statements.append(parsed)

    return statements


def _is_valid_statement(stmt_text: str) -> bool:
    """Check if a statement is valid (not empty, not just comments/whitespace)."""
    if not stmt_text:
        return False

    # Remove all single-line comments
    lines = []
    for line in stmt_text.split("\n"):
        # Remove comments
        if "--" in line:
            line = line.split("--")[0]
        lines.append(line)

    # Join and strip
    cleaned = " ".join(lines).strip()

    # Remove any bare semicolons
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()

    # Check if there's anything left that looks like SQL
    return bool(cleaned) and cleaned not in (";", "")


def collect_sql_files_recursive(directory: Path) -> list[Path]:
    """
    Recursively collect all SQL files from a directory.

    No ordering is imposed - the retry executor handles dependencies.

    Args:
        directory: Root directory to search

    Returns:
        List of SQL file paths
    """
    if not directory.exists():
        return []

    sql_files: list[Path] = []

    for item in sorted(directory.iterdir()):
        if item.is_dir():
            sql_files.extend(collect_sql_files_recursive(item))
        elif item.is_file() and item.suffix == ".sql":
            sql_files.append(item)

    return sql_files
