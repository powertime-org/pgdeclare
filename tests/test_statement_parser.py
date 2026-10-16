"""
Tests for statement_parser.py - SQL parsing with pglast.

Covers:
- ParsedStatement duplicate key computation
- CREATE TABLE parsing
- CREATE FUNCTION parsing with argument normalization
- CREATE INDEX parsing
- CREATE TRIGGER parsing
- CREATE POLICY parsing
- SQL statement splitting
- File collection
"""

from pgdeclare.statement_parser import (
    ParsedStatement,
    parse_sql_content,
    parse_sql_file,
    collect_sql_files_recursive,
    parse_role_from_sql,
    _strip_leading_comments,
    _compute_line_offsets,
    _offset_to_line,
)


def _safe_dollar_tag(args: str) -> str:
    """Generate a dollar tag that doesn't conflict with the args content."""
    tag_base = "$pt$"
    tag = tag_base
    counter = 0
    while tag in args:
        counter += 1
        tag = f"$pt{counter}$"
    return tag


def get_signature_types(args: str) -> str:
    """
    Extract normalized argument types using the production parsing path.

    Creates a full CREATE FUNCTION statement with a safe dollar tag,
    parses it with parse_sql_content, and extracts the types from
    the resulting signature.

    Args:
        args: The argument string (e.g., "p_id uuid, p_name text")

    Returns:
        Comma-separated type string (e.g., "uuid, text")
    """
    if not args.strip():
        return ""
    tag = _safe_dollar_tag(args)
    sql = f"CREATE FUNCTION f({args}) RETURNS void AS {tag} {tag} LANGUAGE sql;"
    statements = parse_sql_content(sql)
    if not statements or not statements[0].signature:
        return ""
    sig = statements[0].signature
    # Extract just the types part from "public.f(types)"
    return sig.split("(", 1)[1].rstrip(")")


class TestParsedStatement:
    """Tests for ParsedStatement dataclass."""

    def test_duplicate_key_computed_from_object_type_and_name(self):
        """Duplicate key is auto-computed from object_type and object_name."""
        stmt = ParsedStatement(
            raw_sql="CREATE TABLE test(id int);",
            statement_type="CREATE TABLE",
            object_type="TABLE",
            object_name="public.test",
        )
        assert stmt.duplicate_key == "TABLE:public.test"

    def test_duplicate_key_uses_signature_for_functions(self):
        """Functions use signature for uniqueness (handles overloads)."""
        stmt = ParsedStatement(
            raw_sql="CREATE FUNCTION test(int) RETURNS void...",
            statement_type="CREATE FUNCTION",
            object_type="FUNCTION",
            object_name="public.test",
            signature="public.test(integer)",
        )
        assert stmt.duplicate_key == "FUNCTION:public.test(integer)"

    def test_duplicate_key_not_computed_without_object_info(self):
        """Untracked statements (GRANT, etc.) have no duplicate_key."""
        stmt = ParsedStatement(
            raw_sql="GRANT SELECT ON test TO app_user;",
            statement_type="GRANT",
            object_type=None,
            object_name=None,
        )
        assert stmt.duplicate_key is None

    def test_explicit_duplicate_key_not_overwritten(self):
        """Explicitly provided duplicate_key is preserved."""
        stmt = ParsedStatement(
            raw_sql="CREATE TABLE test(id int);",
            statement_type="CREATE TABLE",
            object_type="TABLE",
            object_name="public.test",
            duplicate_key="CUSTOM:key",
        )
        assert stmt.duplicate_key == "CUSTOM:key"


class TestParseCreateTable:
    """Tests for CREATE TABLE statement parsing."""

    def test_simple_table(self):
        """Parse simple CREATE TABLE statement."""
        sql = "CREATE TABLE test (id integer PRIMARY KEY);"
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        stmt = statements[0]
        assert stmt.statement_type == "CREATE TABLE"
        assert stmt.object_type == "TABLE"
        assert stmt.object_name == "public.test"
        assert stmt.is_idempotent is False

    def test_table_with_if_not_exists_is_idempotent(self):
        """CREATE TABLE IF NOT EXISTS is marked idempotent."""
        sql = "CREATE TABLE IF NOT EXISTS test (id integer);"
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        assert statements[0].is_idempotent is True

    def test_schema_qualified_table_name(self):
        """Schema-qualified table names are preserved."""
        sql = "CREATE TABLE myschema.test (id integer);"
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        assert statements[0].object_name == "myschema.test"

    def test_quoted_table_name(self):
        """Quoted identifiers are handled correctly."""
        sql = 'CREATE TABLE "MyTable" (id integer);'
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        assert statements[0].object_name == "public.MyTable"

    def test_table_with_multiple_columns(self):
        """Tables with multiple columns are parsed correctly."""
        sql = """
        CREATE TABLE account (
            id uuid PRIMARY KEY,
            name text NOT NULL,
            created_at timestamptz DEFAULT now()
        );
        """
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        stmt = statements[0]
        assert stmt.object_name == "public.account"
        # Raw SQL should be preserved
        assert "id uuid PRIMARY KEY" in stmt.raw_sql


class TestParseCreateFunction:
    """Tests for CREATE FUNCTION statement parsing."""

    def test_simple_function_with_args(self):
        """Parse function with typed arguments."""
        sql = """
        CREATE FUNCTION test_func(p_id uuid, p_name text) RETURNS void
        AS $$ BEGIN END; $$ LANGUAGE plpgsql;
        """
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        stmt = statements[0]
        assert stmt.statement_type == "CREATE FUNCTION"
        assert stmt.object_type == "FUNCTION"
        assert stmt.object_name == "public.test_func"
        # Signature should have normalized args (types only)
        assert stmt.signature == "public.test_func(uuid, text)"

    def test_function_or_replace_is_idempotent(self):
        """CREATE OR REPLACE FUNCTION is marked idempotent."""
        sql = "CREATE OR REPLACE FUNCTION test() RETURNS void AS $$ BEGIN END; $$ LANGUAGE plpgsql;"
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        assert statements[0].is_idempotent is True

    def test_function_args_normalized_to_types_only(self):
        """Function argument normalization removes parameter names."""
        # Test using the production parsing path
        assert get_signature_types("p_id uuid, p_name text") == "uuid, text"
        assert get_signature_types("") == ""
        assert get_signature_types("  ") == ""

    def test_function_args_with_in_out_mode(self):
        """IN/OUT/INOUT mode keywords are handled - OUT params are skipped."""
        # OUT parameters are excluded from function identity signature
        assert get_signature_types("IN p_id uuid, OUT p_result text") == "uuid"
        # INOUT parameters are included, integer normalizes to int4
        assert get_signature_types("INOUT p_count integer") == "int4"

    def test_function_args_with_default(self):
        """DEFAULT clauses are stripped from args."""
        # integer normalizes to int4
        assert (
            get_signature_types(
                "p_limit integer DEFAULT 10, p_offset integer DEFAULT 0"
            )
            == "int4, int4"
        )

    def test_function_args_array_type(self):
        """Array types are preserved."""
        assert get_signature_types("p_ids uuid[]") == "uuid[]"

    def test_dollar_quoted_body_handling(self):
        """Function with dollar-quoted body is parsed correctly."""
        sql = """
        CREATE FUNCTION complex_func() RETURNS text AS $$
        DECLARE
            result text;
        BEGIN
            result := 'Hello; World';
            RETURN result;
        END;
        $$ LANGUAGE plpgsql;
        """
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        stmt = statements[0]
        # Semicolons inside dollar quotes should not split the statement
        assert "Hello; World" in stmt.raw_sql

    def test_schema_qualified_function(self):
        """Schema-qualified function names are preserved."""
        sql = "CREATE FUNCTION myschema.test() RETURNS void AS $$ BEGIN END; $$ LANGUAGE plpgsql;"
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        assert statements[0].object_name == "myschema.test"


class TestMultiWordTypeNormalization:
    """Tests for multi-word PostgreSQL type normalization in function arguments.

    Note: pglast normalizes multi-word types to their canonical PostgreSQL names:
    - TIMESTAMP WITH TIME ZONE -> timestamptz
    - TIMESTAMP WITHOUT TIME ZONE -> timestamp
    - TIME WITH TIME ZONE -> timetz
    - DOUBLE PRECISION -> float8
    - CHARACTER VARYING -> varchar
    - BIT VARYING -> varbit
    """

    def test_timestamp_with_time_zone(self):
        """TIMESTAMP WITH TIME ZONE normalizes to timestamptz."""
        assert get_signature_types("p_ts TIMESTAMP WITH TIME ZONE") == "timestamptz"

    def test_timestamp_without_time_zone(self):
        """TIMESTAMP WITHOUT TIME ZONE normalizes to timestamp."""
        assert get_signature_types("p_ts TIMESTAMP WITHOUT TIME ZONE") == "timestamp"

    def test_time_with_time_zone(self):
        """TIME WITH TIME ZONE normalizes to timetz."""
        assert get_signature_types("p_time TIME WITH TIME ZONE") == "timetz"

    def test_double_precision(self):
        """DOUBLE PRECISION normalizes to float8."""
        assert get_signature_types("p_val DOUBLE PRECISION") == "float8"

    def test_character_varying(self):
        """CHARACTER VARYING normalizes to varchar."""
        assert get_signature_types("p_name CHARACTER VARYING") == "varchar"

    def test_bit_varying(self):
        """BIT VARYING normalizes to varbit."""
        assert get_signature_types("p_bits BIT VARYING") == "varbit"

    def test_multi_word_with_default(self):
        """Multi-word types with DEFAULT clause work correctly."""
        assert (
            get_signature_types(
                "p_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"
            )
            == "timestamptz"
        )

    def test_multi_word_with_in_mode(self):
        """Multi-word types with IN mode keyword work correctly."""
        assert get_signature_types("IN p_ts TIMESTAMP WITH TIME ZONE") == "timestamptz"

    def test_multi_word_array_type(self):
        """Multi-word array types are handled correctly."""
        assert get_signature_types("p_ts TIMESTAMP WITH TIME ZONE[]") == "timestamptz[]"

    def test_multi_word_array_with_space(self):
        """Multi-word array types with space before [] are handled."""
        assert (
            get_signature_types("p_ts TIMESTAMP WITH TIME ZONE []") == "timestamptz[]"
        )

    def test_mixed_simple_and_multi_word(self):
        """Mixed simple and multi-word types in same signature."""
        assert (
            get_signature_types("p_id uuid, p_ts TIMESTAMP WITH TIME ZONE, p_name text")
            == "uuid, timestamptz, text"
        )

    def test_multiple_multi_word_types(self):
        """Multiple multi-word types in same signature."""
        assert (
            get_signature_types("p_ts TIMESTAMP WITH TIME ZONE, p_val DOUBLE PRECISION")
            == "timestamptz, float8"
        )


class TestParseCreateIndex:
    """Tests for CREATE INDEX statement parsing."""

    def test_simple_index(self):
        """Parse simple CREATE INDEX statement."""
        sql = "CREATE INDEX idx_test ON test(id);"
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        stmt = statements[0]
        assert stmt.statement_type == "CREATE INDEX"
        assert stmt.object_type == "INDEX"
        assert stmt.object_name == "idx_test"

    def test_unique_index(self):
        """Parse CREATE UNIQUE INDEX statement."""
        sql = "CREATE UNIQUE INDEX idx_test_unique ON test(email);"
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        # Note: CREATE UNIQUE INDEX is handled by generic parsing
        # The parser extracts it as CREATE UNIQUE statement type
        assert statements[0].statement_type in ("CREATE INDEX", "CREATE UNIQUE")
        # Raw SQL is preserved
        assert "idx_test_unique" in statements[0].raw_sql

    def test_concurrent_index_if_not_exists(self):
        """CREATE INDEX CONCURRENTLY IF NOT EXISTS is idempotent."""
        sql = "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_test ON test(id);"
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        assert statements[0].is_idempotent is True


class TestParseCreateTrigger:
    """Tests for CREATE TRIGGER statement parsing."""

    def test_trigger_name_and_table_extracted(self):
        """Trigger name and table are extracted correctly."""
        sql = """
        CREATE TRIGGER audit_trigger
        AFTER INSERT OR UPDATE ON account
        FOR EACH ROW EXECUTE FUNCTION audit_log();
        """
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        stmt = statements[0]
        assert stmt.statement_type == "CREATE TRIGGER"
        assert stmt.object_type == "TRIGGER"
        # Object name should include table for uniqueness
        assert stmt.object_name is not None
        assert "account" in stmt.object_name
        assert "audit_trigger" in stmt.object_name

    def test_constraint_trigger(self):
        """CONSTRAINT TRIGGER is parsed correctly."""
        sql = """
        CREATE CONSTRAINT TRIGGER check_balance
        AFTER INSERT ON transaction
        DEFERRABLE INITIALLY DEFERRED
        FOR EACH ROW EXECUTE FUNCTION check_balance();
        """
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        # Note: CREATE CONSTRAINT TRIGGER is handled as generic CREATE CONSTRAINT
        # The parser recognizes the raw SQL contains the trigger definition
        assert "CREATE CONSTRAINT TRIGGER" in statements[0].raw_sql
        assert "check_balance" in statements[0].raw_sql

    def test_or_replace_trigger(self):
        """CREATE OR REPLACE TRIGGER is idempotent."""
        sql = """
        CREATE OR REPLACE TRIGGER update_trigger
        BEFORE UPDATE ON account
        FOR EACH ROW EXECUTE FUNCTION update_timestamp();
        """
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        assert statements[0].is_idempotent is True


class TestParseCreatePolicy:
    """Tests for CREATE POLICY statement parsing."""

    def test_policy_on_table(self):
        """Policy name and table are extracted correctly."""
        sql = """
        CREATE POLICY account_select ON account
        FOR SELECT USING (tenant_id = current_setting('app.tenant_id')::uuid);
        """
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        stmt = statements[0]
        assert stmt.statement_type == "CREATE POLICY"
        assert stmt.object_type == "POLICY"
        # Object name should be table.policy
        assert stmt.object_name == "public.account.account_select"

    def test_policy_with_schema(self):
        """Schema-qualified table names are preserved for policies."""
        sql = "CREATE POLICY test_policy ON myschema.mytable FOR ALL USING (true);"
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        assert statements[0].object_name == "myschema.mytable.test_policy"


class TestParseOtherStatements:
    """Tests for parsing other statement types."""

    def test_create_type_enum(self):
        """CREATE TYPE for enum is parsed correctly."""
        sql = "CREATE TYPE status AS ENUM ('active', 'inactive', 'pending');"
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        stmt = statements[0]
        assert stmt.statement_type == "CREATE TYPE"
        assert stmt.object_type == "TYPE"
        assert stmt.object_name == "public.status"

    def test_create_extension(self):
        """CREATE EXTENSION is parsed correctly."""
        sql = 'CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        stmt = statements[0]
        assert stmt.statement_type == "CREATE EXTENSION"
        assert stmt.object_type == "EXTENSION"
        assert stmt.is_idempotent is True

    def test_create_schema(self):
        """CREATE SCHEMA is parsed correctly."""
        sql = "CREATE SCHEMA IF NOT EXISTS api;"
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        stmt = statements[0]
        assert stmt.statement_type == "CREATE SCHEMA"
        assert stmt.object_type == "SCHEMA"
        assert stmt.object_name == "api"

    def test_create_sequence(self):
        """CREATE SEQUENCE is parsed correctly."""
        sql = "CREATE SEQUENCE IF NOT EXISTS order_id_seq;"
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        stmt = statements[0]
        assert stmt.statement_type == "CREATE SEQUENCE"
        assert stmt.object_type == "SEQUENCE"
        assert stmt.is_idempotent is True

    def test_create_role(self):
        """CREATE ROLE is parsed correctly."""
        sql = "CREATE ROLE app_user WITH LOGIN;"
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        stmt = statements[0]
        assert stmt.statement_type == "CREATE ROLE"
        assert stmt.object_type == "ROLE"
        assert stmt.object_name == "app_user"

    def test_create_view(self):
        """CREATE VIEW is parsed correctly."""
        sql = "CREATE OR REPLACE VIEW active_accounts AS SELECT * FROM account WHERE active = true;"
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        stmt = statements[0]
        assert stmt.statement_type == "CREATE VIEW"
        assert stmt.object_type == "VIEW"
        assert stmt.is_idempotent is True

    def test_grant_statement(self):
        """GRANT statements have no object tracking."""
        sql = "GRANT SELECT ON account TO app_user;"
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        stmt = statements[0]
        # Statement type includes the privilege for GRANT statements
        assert "GRANT" in stmt.statement_type
        assert stmt.object_type is None
        assert stmt.duplicate_key is None


class TestSQLSplitting:
    """Tests for SQL statement splitting."""

    def test_pglast_splits_multiple_statements(self):
        """pglast correctly splits multiple statements."""
        sql = """
        CREATE TABLE a (id int);
        CREATE TABLE b (id int);
        CREATE TABLE c (id int);
        """
        statements = parse_sql_content(sql)

        assert len(statements) == 3
        assert statements[0].object_name == "public.a"
        assert statements[1].object_name == "public.b"
        assert statements[2].object_name == "public.c"

    def test_dollar_quote_handling(self):
        """Dollar quotes containing semicolons don't split incorrectly."""
        sql = """
        CREATE FUNCTION test() RETURNS void AS $$
        BEGIN
            PERFORM 'semicolon; here';
        END;
        $$ LANGUAGE plpgsql;
        CREATE TABLE after_func (id int);
        """
        statements = parse_sql_content(sql)

        assert len(statements) == 2
        assert statements[0].statement_type == "CREATE FUNCTION"
        assert statements[1].statement_type == "CREATE TABLE"

    def test_comment_handling(self):
        """Comments are handled correctly and don't affect splitting."""
        sql = """
        -- This is a comment
        CREATE TABLE test (id int); -- inline comment
        -- Another comment
        CREATE TABLE test2 (id int);
        """
        statements = parse_sql_content(sql)

        assert len(statements) == 2

    def test_semicolon_in_string_not_split(self):
        """Semicolons inside strings don't cause splits."""
        sql = """
        INSERT INTO config (key, value) VALUES ('query', 'SELECT * FROM t;');
        """
        statements = parse_sql_content(sql)

        # Should be one statement, not split at the semicolon inside the string
        assert len(statements) == 1

    def test_empty_statements_filtered(self):
        """Empty statements and bare semicolons are filtered out."""
        sql = """
        CREATE TABLE test (id int);
        ;
        ;
        CREATE TABLE test2 (id int);
        """
        statements = parse_sql_content(sql)

        assert len(statements) == 2

    def test_source_file_preserved(self):
        """Source file is preserved in parsed statements."""
        sql = "CREATE TABLE test (id int);"
        statements = parse_sql_content(sql, source_file="schema/01_tables.sql")

        assert len(statements) == 1
        assert statements[0].source_file == "schema/01_tables.sql"


class TestCommentStripping:
    """Tests for comment stripping functionality."""

    def test_strip_leading_comments(self):
        """Leading comments and whitespace are stripped."""
        sql = """
        -- This is a comment
        -- Another comment
        CREATE TABLE test (id int);
        """
        stripped = _strip_leading_comments(sql)
        # The stripped result should start with the CREATE TABLE line
        # (may have leading whitespace on the line itself)
        assert "CREATE TABLE" in stripped
        assert stripped.strip().startswith("CREATE TABLE")

    def test_strip_preserves_sql_content(self):
        """SQL content after comments is preserved."""
        sql = """-- comment
CREATE TABLE test (
    id int,
    name text
);"""
        stripped = _strip_leading_comments(sql)
        assert "id int" in stripped
        assert "name text" in stripped

    def test_strip_handles_no_comments(self):
        """SQL without comments passes through unchanged."""
        sql = "CREATE TABLE test (id int);"
        stripped = _strip_leading_comments(sql)
        assert stripped == sql


class TestLineNumberComputation:
    """Tests for line number computation."""

    def test_compute_line_offsets(self):
        """Line offsets are computed correctly."""
        content = "line1\nline2\nline3"
        offsets = _compute_line_offsets(content)
        assert offsets == [0, 6, 12]

    def test_offset_to_line(self):
        """Character offset converts to correct line number."""
        content = "line1\nline2\nline3"
        offsets = _compute_line_offsets(content)

        assert _offset_to_line(0, offsets) == 1  # Start of line 1
        assert _offset_to_line(3, offsets) == 1  # Middle of line 1
        assert _offset_to_line(6, offsets) == 2  # Start of line 2
        assert _offset_to_line(12, offsets) == 3  # Start of line 3


class TestFileCollection:
    """Tests for SQL file collection."""

    def test_collect_recursive_from_nested_directories(self, tmp_path):
        """Files are collected recursively from nested directories."""
        # Create nested directory structure
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "b").mkdir()
        (tmp_path / "a" / "c").mkdir()

        # Create SQL files at various levels
        (tmp_path / "root.sql").write_text("SELECT 1;")
        (tmp_path / "a" / "level1.sql").write_text("SELECT 2;")
        (tmp_path / "a" / "b" / "level2.sql").write_text("SELECT 3;")
        (tmp_path / "a" / "c" / "level2b.sql").write_text("SELECT 4;")

        files = collect_sql_files_recursive(tmp_path)

        assert len(files) == 4
        names = [f.name for f in files]
        assert "root.sql" in names
        assert "level1.sql" in names
        assert "level2.sql" in names
        assert "level2b.sql" in names

    def test_files_sorted_alphabetically(self, tmp_path):
        """Files are sorted alphabetically within each directory."""
        (tmp_path / "03_third.sql").write_text("SELECT 3;")
        (tmp_path / "01_first.sql").write_text("SELECT 1;")
        (tmp_path / "02_second.sql").write_text("SELECT 2;")

        files = collect_sql_files_recursive(tmp_path)

        names = [f.name for f in files]
        assert names == ["01_first.sql", "02_second.sql", "03_third.sql"]

    def test_non_sql_files_excluded(self, tmp_path):
        """Non-.sql files are excluded from collection."""
        (tmp_path / "schema.sql").write_text("CREATE TABLE t(id int);")
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "config.json").write_text("{}")

        files = collect_sql_files_recursive(tmp_path)

        assert len(files) == 1
        assert files[0].name == "schema.sql"

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """Empty directory returns empty list."""
        files = collect_sql_files_recursive(tmp_path)
        assert files == []

    def test_nonexistent_directory_returns_empty_list(self, tmp_path):
        """Non-existent directory returns empty list."""
        files = collect_sql_files_recursive(tmp_path / "nonexistent")
        assert files == []


class TestParseFile:
    """Tests for parse_sql_file function."""

    def test_parse_sql_file(self, tmp_path):
        """parse_sql_file reads and parses a file."""
        sql_file = tmp_path / "test.sql"
        sql_file.write_text("""
        CREATE TABLE t1 (id int);
        CREATE TABLE t2 (id int);
        """)

        statements = parse_sql_file(sql_file, base_path=tmp_path)

        assert len(statements) == 2
        assert statements[0].source_file == "test.sql"

    def test_parse_sql_file_absolute_path(self, tmp_path):
        """parse_sql_file works without base_path."""
        sql_file = tmp_path / "test.sql"
        sql_file.write_text("CREATE TABLE t (id int);")

        statements = parse_sql_file(sql_file)

        assert len(statements) == 1
        # Without base_path, source_file is the full path
        assert statements[0].source_file == str(sql_file)


class TestEdgeCases:
    """Tests for edge cases and complex scenarios."""

    def test_statement_with_embedded_comments(self):
        """Statements with embedded comments are parsed correctly."""
        sql = """
        CREATE TABLE test (
            id integer, -- primary key
            name text   -- user's name
        );
        """
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        assert statements[0].object_name == "public.test"

    def test_multiline_function_with_complex_body(self):
        """Complex multi-line functions are parsed as single statements."""
        sql = """
        CREATE OR REPLACE FUNCTION complex_function(
            p_id uuid,
            p_name text DEFAULT 'default'
        )
        RETURNS TABLE (
            id uuid,
            name text,
            created_at timestamptz
        )
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = public
        AS $$
        DECLARE
            v_count integer;
        BEGIN
            SELECT COUNT(*) INTO v_count FROM users;
            RETURN QUERY
            SELECT u.id, u.name, u.created_at
            FROM users u
            WHERE u.id = p_id;
        END;
        $$;
        """
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        stmt = statements[0]
        assert stmt.statement_type == "CREATE FUNCTION"
        assert stmt.is_idempotent is True

    def test_statements_with_special_characters_in_strings(self):
        """Statements with special characters in strings are handled."""
        sql = """
        INSERT INTO log (message) VALUES ('Error: "invalid"; code=42');
        CREATE TABLE test (id int);
        """
        statements = parse_sql_content(sql)

        # Should be 2 statements
        assert len(statements) == 2

    def test_mixed_case_keywords(self):
        """Mixed-case SQL keywords are handled correctly."""
        sql = "Create Table TEST (id Integer);"
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        assert statements[0].statement_type == "CREATE TABLE"
        # Note: object name depends on SQL parsing (usually lowercase)

    def test_statement_at_file_start_with_comment_inside(self):
        """Statements with comments inside are parsed correctly."""
        sql = """CREATE TABLE test (
            -- This is a comment about id
            id integer PRIMARY KEY,
            -- This is about name
            name text
        );"""
        statements = parse_sql_content(sql)

        assert len(statements) == 1
        assert "id integer" in statements[0].raw_sql


class TestParseRoleFromSql:
    """Tests for parse_role_from_sql function - role password extraction."""

    def test_parse_role_with_password(self):
        """Extract role name and password from CREATE ROLE with PASSWORD."""
        sql = "CREATE ROLE app_user WITH LOGIN PASSWORD 'app_user';"
        result = parse_role_from_sql(sql)

        assert result is not None
        assert result["name"] == "app_user"
        assert result["password"] == "app_user"
        assert result["login"] is True

    def test_parse_role_no_password(self):
        """Handle CREATE ROLE without PASSWORD clause."""
        sql = "CREATE ROLE admin_role WITH SUPERUSER;"
        result = parse_role_from_sql(sql)

        assert result is not None
        assert result["name"] == "admin_role"
        assert "password" not in result

    def test_parse_role_password_with_escaped_quote(self):
        """Handle passwords containing escaped single quotes ('')."""
        sql = "CREATE ROLE test_user WITH LOGIN PASSWORD 'pass''word';"
        result = parse_role_from_sql(sql)

        assert result is not None
        assert result["name"] == "test_user"
        assert result["password"] == "pass'word"

    def test_parse_role_with_nologin(self):
        """Handle NOLOGIN attribute."""
        sql = "CREATE ROLE background_worker WITH NOLOGIN;"
        result = parse_role_from_sql(sql)

        assert result is not None
        assert result["name"] == "background_worker"
        assert result["login"] is False

    def test_parse_non_role_statement(self):
        """Return None for non-CREATE ROLE statements."""
        sql = "CREATE TABLE test (id int);"
        result = parse_role_from_sql(sql)

        assert result is None

    def test_parse_role_with_leading_comment(self):
        """Handle CREATE ROLE with leading comment."""
        sql = """-- Create API user for backend
        CREATE ROLE app_user WITH LOGIN PASSWORD 'secret';"""
        result = parse_role_from_sql(sql)

        assert result is not None
        assert result["name"] == "app_user"
        assert result["password"] == "secret"

    def test_parse_role_multiline(self):
        """Handle multi-line CREATE ROLE statement."""
        sql = """CREATE ROLE app_user
            WITH LOGIN
            PASSWORD 'app_user'
            CONNECTION LIMIT 100;"""
        result = parse_role_from_sql(sql)

        assert result is not None
        assert result["name"] == "app_user"
        assert result["password"] == "app_user"
        assert result["login"] is True


class TestParameterNameEdgeCases:
    """Tests for parameter names matching type keywords.

    To use reserved keywords as parameter names, they must be quoted with
    double quotes to produce valid SQL.
    """

    def test_parameter_named_time(self):
        """Parameter named 'time' (quoted) is not treated as type start."""
        assert get_signature_types('"time" text') == "text"

    def test_parameter_named_timestamp(self):
        """Parameter named 'timestamp' (quoted) is not treated as type start."""
        # integer normalizes to int4
        assert get_signature_types('"timestamp" integer') == "int4"

    def test_parameter_named_double(self):
        """Parameter named 'double' (quoted) is not treated as type start."""
        assert get_signature_types('"double" text') == "text"

    def test_parameter_named_interval(self):
        """Parameter named 'interval' (quoted) is not treated as type start."""
        assert get_signature_types('"interval" text') == "text"

    def test_parameter_named_national(self):
        """Parameter named 'national' (quoted) is not treated as type start."""
        assert get_signature_types('"national" varchar') == "varchar"

    def test_keyword_as_param_with_multi_word_type(self):
        """Parameter named like keyword followed by actual multi-word type."""
        # 'time' is param name (quoted), 'timestamp with time zone' is the type
        result = get_signature_types('"time" TIMESTAMP WITH TIME ZONE')
        assert result == "timestamptz"


class TestTypePrecisionNormalization:
    """Tests for type precision/length modifier stripping in function signatures."""

    def test_numeric_with_precision(self):
        """Numeric precision modifiers are stripped."""
        assert get_signature_types("p_val numeric(10,2)") == "numeric"

    def test_varchar_with_length(self):
        """Varchar length modifiers are stripped."""
        assert get_signature_types("p_name varchar(255)") == "varchar"

    def test_char_with_length(self):
        """Char length modifiers are stripped, normalized to bpchar."""
        # PostgreSQL internally represents char(n) as bpchar (blank-padded char)
        assert get_signature_types("p_code char(10)") == "bpchar"

    def test_decimal_with_precision(self):
        """Decimal precision modifiers are stripped, normalized to numeric."""
        # PostgreSQL internally represents decimal as numeric
        assert get_signature_types("p_price decimal(12,4)") == "numeric"

    def test_multiple_args_with_precision(self):
        """Multiple arguments with precision modifiers."""
        assert (
            get_signature_types("p_a numeric(10,2), p_b varchar(100)")
            == "numeric, varchar"
        )

    def test_precision_array_type(self):
        """Array types with precision keep array notation."""
        assert get_signature_types("p_vals numeric(10,2)[]") == "numeric[]"

    def test_mixed_precision_and_simple(self):
        """Mixed precision and simple types."""
        assert (
            get_signature_types("p_id uuid, p_val numeric(10,2), p_name text")
            == "uuid, numeric, text"
        )


class TestIntervalTypeNormalization:
    """Tests for INTERVAL type with field specifications.

    Note: pglast normalizes all INTERVAL types to just 'interval'.
    The field specification (DAY TO SECOND, etc.) is stored in typmods,
    not in the type name. This matches pg_get_function_identity_arguments().
    """

    def test_interval_day_to_second(self):
        """INTERVAL DAY TO SECOND normalizes to interval."""
        assert get_signature_types("p_dur INTERVAL DAY TO SECOND") == "interval"

    def test_interval_hour_to_minute(self):
        """INTERVAL HOUR TO MINUTE normalizes to interval."""
        assert get_signature_types("p_dur INTERVAL HOUR TO MINUTE") == "interval"

    def test_interval_year_to_month(self):
        """INTERVAL YEAR TO MONTH normalizes to interval."""
        assert get_signature_types("p_period INTERVAL YEAR TO MONTH") == "interval"

    def test_simple_interval(self):
        """Simple INTERVAL without field spec."""
        assert get_signature_types("p_dur INTERVAL") == "interval"


class TestNationalCharacterTypes:
    """Tests for NATIONAL CHARACTER type normalization.

    Note: pglast normalizes national character types:
    - NATIONAL CHARACTER -> bpchar
    - NATIONAL CHARACTER VARYING -> varchar
    - NATIONAL CHAR -> bpchar
    """

    def test_national_character(self):
        """NATIONAL CHARACTER normalizes to bpchar."""
        assert get_signature_types("p_name NATIONAL CHARACTER") == "bpchar"

    def test_national_character_varying(self):
        """NATIONAL CHARACTER VARYING normalizes to varchar."""
        assert get_signature_types("p_name NATIONAL CHARACTER VARYING") == "varchar"

    def test_national_char(self):
        """NATIONAL CHAR normalizes to bpchar."""
        assert get_signature_types("p_code NATIONAL CHAR") == "bpchar"


class TestCombinedEdgeCases:
    """Tests combining multiple edge cases.

    To use reserved keywords as parameter names, they must be quoted with
    double quotes to produce valid SQL.
    """

    def test_keyword_param_with_precision_type(self):
        """Parameter named like keyword (quoted) with precision type."""
        # Use quoted identifier for the reserved keyword 'time'
        result = get_signature_types('"time" numeric(10,2)')
        assert result == "numeric"

    def test_multi_word_type_normalized(self):
        """Multi-word types normalize to canonical names."""
        assert get_signature_types("p_ts TIMESTAMP WITH TIME ZONE") == "timestamptz"

    def test_complex_signature_valid_sql(self):
        """Complex signature with valid SQL (no reserved keyword params)."""
        args = (
            "p_name text, "  # normal param name
            "p_val numeric(10,2), "  # precision modifier
            "p_ts TIMESTAMP WITH TIME ZONE, "  # multi-word type
            "p_dur INTERVAL DAY TO SECOND"  # interval with fields
        )
        expected = "text, numeric, timestamptz, interval"
        assert get_signature_types(args) == expected

    def test_all_keyword_params_quoted(self):
        """All parameters named like type keywords - quoted for valid SQL."""
        # Use quoted identifiers for reserved keywords
        args = '"time" text, "timestamp" uuid, "double" integer, "interval" varchar'
        # integer normalizes to int4
        expected = "text, uuid, int4, varchar"
        assert get_signature_types(args) == expected


class TestUnnamedArgEdgeCases:
    """Tests for unnamed arguments - previously failing edge cases.

    Note: pglast normalizes types to canonical PostgreSQL names.
    Multi-word types become internal names (e.g., timestamptz, float8).
    """

    def test_unnamed_multi_word_arg(self):
        """Unnamed arguments with multi-word types normalize correctly."""
        assert get_signature_types("timestamp with time zone") == "timestamptz"
        assert get_signature_types("double precision") == "float8"

    def test_unnamed_simple_arg(self):
        """Unnamed arguments with simple types work correctly."""
        assert get_signature_types("uuid") == "uuid"
        assert get_signature_types("text") == "text"

    def test_unnamed_mixed_args(self):
        """Mix of unnamed and named arguments."""
        assert get_signature_types("uuid, p_name text") == "uuid, text"

    def test_multiple_unnamed_args(self):
        """Multiple unnamed arguments - integer normalizes to int4."""
        assert get_signature_types("uuid, text, integer") == "uuid, text, int4"

    def test_type_with_precision_and_qualifier(self):
        """timestamp(6) with time zone normalizes to timestamptz, strips precision."""
        result = get_signature_types("p_ts timestamp(6) with time zone")
        assert result == "timestamptz"

    def test_time_with_precision_and_qualifier(self):
        """time(6) with time zone normalizes to timetz, strips precision."""
        result = get_signature_types("p_t time(6) with time zone")
        assert result == "timetz"

    def test_default_with_comma_in_string(self):
        """Commas in DEFAULT string values should not split args."""
        result = get_signature_types("p_val text DEFAULT 'a,b', p_id uuid")
        assert result == "text, uuid"

    def test_default_with_complex_expression(self):
        """DEFAULT with complex expression containing parentheses."""
        result = get_signature_types("p_arr integer[] DEFAULT ARRAY[1,2,3], p_id uuid")
        assert result == "int4[], uuid"

    def test_variadic_argument(self):
        """VARIADIC arguments are handled correctly."""
        result = get_signature_types("VARIADIC p_ids uuid[]")
        assert result == "uuid[]"

    def test_unnamed_array_type(self):
        """Unnamed array type arguments."""
        assert get_signature_types("uuid[]") == "uuid[]"
        assert get_signature_types("text[]") == "text[]"


class TestPglastParsing:
    """Tests for pglast-based parsing of CREATE statements.

    These tests verify the AST-based parsing handles edge cases that broke
    with regex patterns, particularly types with parentheses like numeric(10,2).
    """

    def test_function_with_precision_type(self):
        """Function with precision type like numeric(10,2) parses correctly.

        This was the original bug: regex [^)]* stopped at first ) in numeric(10,2).
        """
        sql = "CREATE FUNCTION test(p_val numeric(10,2)) RETURNS void AS $$ $$ LANGUAGE sql;"
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        assert statements[0].signature == "public.test(numeric)"

    def test_function_with_multiple_precision_types(self):
        """Function with multiple precision types in signature."""
        sql = "CREATE FUNCTION test(p_a numeric(10,2), p_b varchar(255)) RETURNS void AS $$ $$ LANGUAGE sql;"
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        assert statements[0].signature == "public.test(numeric, varchar)"

    def test_function_with_array_default(self):
        """Function with array default value containing commas."""
        sql = "CREATE FUNCTION test(p_arr int[] DEFAULT ARRAY[1,2,3]) RETURNS void AS $$ $$ LANGUAGE sql;"
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        assert statements[0].signature == "public.test(int4[])"

    def test_table_ast_parsing(self):
        """CREATE TABLE uses AST parsing correctly."""
        sql = "CREATE TABLE IF NOT EXISTS test_table (id int, value numeric(10,2));"
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        assert statements[0].object_name == "public.test_table"
        assert statements[0].is_idempotent is True

    def test_view_ast_parsing(self):
        """CREATE VIEW uses AST parsing correctly."""
        sql = "CREATE OR REPLACE VIEW test_view AS SELECT 1;"
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        assert statements[0].object_name == "public.test_view"
        assert statements[0].is_idempotent is True

    def test_index_ast_parsing(self):
        """CREATE INDEX uses AST parsing correctly."""
        sql = "CREATE INDEX IF NOT EXISTS idx_test ON test_table(id);"
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        assert statements[0].object_name == "idx_test"
        assert statements[0].is_idempotent is True

    def test_trigger_ast_parsing(self):
        """CREATE TRIGGER uses AST parsing with table.trigger format."""
        sql = """
        CREATE TRIGGER audit_trigger
        AFTER INSERT ON account
        FOR EACH ROW EXECUTE FUNCTION audit_log();
        """
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        assert statements[0].object_name == "public.account.audit_trigger"

    def test_policy_ast_parsing(self):
        """CREATE POLICY uses AST parsing with table.policy format."""
        sql = (
            "CREATE POLICY read_own ON users FOR SELECT USING (id = current_user_id());"
        )
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        assert statements[0].object_name == "public.users.read_own"

    def test_enum_type_ast_parsing(self):
        """CREATE TYPE AS ENUM uses AST parsing correctly."""
        sql = "CREATE TYPE status AS ENUM ('active', 'inactive');"
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        assert statements[0].object_name == "public.status"

    def test_composite_type_ast_parsing(self):
        """CREATE TYPE AS composite uses AST parsing correctly."""
        sql = "CREATE TYPE address AS (street text, city text);"
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        assert statements[0].object_name == "public.address"

    def test_extension_ast_parsing(self):
        """CREATE EXTENSION uses AST parsing correctly."""
        sql = 'CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        assert statements[0].object_name == "uuid-ossp"
        assert statements[0].is_idempotent is True

    def test_schema_ast_parsing(self):
        """CREATE SCHEMA uses AST parsing correctly."""
        sql = "CREATE SCHEMA IF NOT EXISTS api;"
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        assert statements[0].object_name == "api"
        assert statements[0].is_idempotent is True

    def test_sequence_ast_parsing(self):
        """CREATE SEQUENCE uses AST parsing correctly."""
        sql = "CREATE SEQUENCE IF NOT EXISTS order_seq;"
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        assert statements[0].object_name == "public.order_seq"
        assert statements[0].is_idempotent is True

    def test_role_ast_parsing(self):
        """CREATE ROLE uses AST parsing correctly."""
        sql = "CREATE ROLE app_user WITH LOGIN;"
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        assert statements[0].object_name == "app_user"

    def test_function_schema_qualified(self):
        """Function with schema qualification is parsed correctly."""
        sql = "CREATE FUNCTION myschema.test(p_val numeric(10,2)) RETURNS void AS $$ $$ LANGUAGE sql;"
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        assert statements[0].object_name == "myschema.test"
        assert statements[0].signature == "myschema.test(numeric)"

    def test_custom_schema_type(self):
        """Custom schema-qualified type uses dot notation, not space."""
        sql = "CREATE FUNCTION test(p_val myschema.mytype) RETURNS void AS $$ $$ LANGUAGE sql;"
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        # Should be myschema.mytype, not "myschema mytype"
        assert statements[0].signature == "public.test(myschema.mytype)"

    def test_builtin_multiword_type_uses_space(self):
        """Built-in multi-word types (from pg_catalog) use space-joined names."""
        sql = "CREATE FUNCTION test(p_ts TIMESTAMP WITH TIME ZONE) RETURNS void AS $$ $$ LANGUAGE sql;"
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        # timestamptz is the canonical form
        assert statements[0].signature == "public.test(timestamptz)"

    def test_custom_schema_type_array(self):
        """Custom schema-qualified array type preserves dot notation."""
        sql = "CREATE FUNCTION test(p_vals myschema.mytype[]) RETURNS void AS $$ $$ LANGUAGE sql;"
        statements = parse_sql_content(sql)
        assert len(statements) == 1
        assert statements[0].signature == "public.test(myschema.mytype[])"

    def test_case_sensitive_public_type(self):
        """Case-sensitive type in public schema preserves case and schema."""
        sql = 'CREATE FUNCTION test(p_val public."MyType") RETURNS void AS $$ $$ LANGUAGE sql;'
        statements = parse_sql_content(sql)
        assert statements[0].signature == "public.test(public.MyType)"

    def test_lowercase_public_type_strips_schema(self):
        """Lowercase type in public schema strips the public prefix."""
        sql = "CREATE FUNCTION test(p_val public.mytype) RETURNS void AS $$ $$ LANGUAGE sql;"
        statements = parse_sql_content(sql)
        assert statements[0].signature == "public.test(mytype)"

    def test_case_sensitive_custom_schema_type(self):
        """Case-sensitive type in custom schema preserves case."""
        sql = 'CREATE FUNCTION test(p_val myschema."CamelCase") RETURNS void AS $$ $$ LANGUAGE sql;'
        statements = parse_sql_content(sql)
        assert statements[0].signature == "public.test(myschema.CamelCase)"
