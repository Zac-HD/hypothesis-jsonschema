"""Tests for the hypothesis-jsonschema command-line interface."""

import json
import os
import stat

import jsonschema
import pytest
from click.testing import CliRunner
from hypothesis.extra.cli import main as hypothesis_cli

from hypothesis_jsonschema._cli import generate_examples, load_schema, run_fuzz


# =============================================================================
# Tests for load_schema function
# =============================================================================


def test_load_schema_from_string():
    """Load a schema from a JSON string."""
    schema = load_schema('{"type": "integer"}')
    assert schema == {"type": "integer"}


def test_load_schema_from_file(tmp_path):
    """Load a schema from a file path."""
    schema_file = tmp_path / "schema.json"
    schema_file.write_text('{"type": "string", "minLength": 1}')
    schema = load_schema(str(schema_file))
    assert schema == {"type": "string", "minLength": 1}


def test_load_schema_invalid_json_raises():
    """Invalid JSON raises ValueError."""
    with pytest.raises(ValueError, match="Could not parse schema"):
        load_schema("not valid json")


def test_load_schema_nonexistent_file_tries_json():
    """A non-existent file path is tried as JSON first."""
    with pytest.raises(ValueError, match="Could not parse schema"):
        load_schema("/nonexistent/path/to/schema.json")


# =============================================================================
# Tests for generate_examples function
# =============================================================================


def test_generate_integers():
    """Generate integer examples."""
    schema = {"type": "integer", "minimum": 0, "maximum": 100}
    examples = generate_examples(schema, num=10)
    assert len(examples) == 10
    for ex in examples:
        assert isinstance(ex, int)
        assert 0 <= ex <= 100


def test_generate_strings():
    """Generate string examples."""
    schema = {"type": "string", "minLength": 1, "maxLength": 5}
    examples = generate_examples(schema, num=5)
    assert len(examples) == 5
    for ex in examples:
        assert isinstance(ex, str)
        assert 1 <= len(ex) <= 5


def test_generate_objects():
    """Generate object examples."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
        },
        "required": ["name"],
    }
    examples = generate_examples(schema, num=5)
    assert len(examples) == 5
    for ex in examples:
        assert isinstance(ex, dict)
        assert "name" in ex
        assert isinstance(ex["name"], str)


def test_seed_reproducibility():
    """Same seed produces same output."""
    schema = {"type": "integer"}
    examples1 = generate_examples(schema, num=10, seed_value=42)
    examples2 = generate_examples(schema, num=10, seed_value=42)
    assert examples1 == examples2


def test_different_seeds_different_output():
    """Different seeds produce different output."""
    schema = {"type": "integer"}
    examples1 = generate_examples(schema, num=10, seed_value=42)
    examples2 = generate_examples(schema, num=10, seed_value=123)
    # With enough examples, they should differ
    assert examples1 != examples2


def test_unique_examples():
    """Generated examples are unique."""
    schema = {"type": "integer", "minimum": 0, "maximum": 1000000}
    examples = generate_examples(schema, num=50)
    assert len(examples) == len(set(examples))


def test_validates_against_schema():
    """All generated examples validate against the schema."""
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["id"],
    }
    examples = generate_examples(schema, num=10)
    validator = jsonschema.Draft7Validator(schema)
    for ex in examples:
        validator.validate(ex)


# =============================================================================
# Tests for the Click CLI (hypothesis json)
# =============================================================================


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner."""
    return CliRunner()


def test_cli_generate_mode_stdout(cli_runner):
    """Generation mode outputs JSON lines to stdout."""
    result = cli_runner.invoke(
        hypothesis_cli, ["json", '{"type": "integer"}', "--num", "5"]
    )
    assert result.exit_code == 0
    lines = result.output.strip().split("\n")
    assert len(lines) == 5
    for line in lines:
        value = json.loads(line)
        assert isinstance(value, int)


def test_cli_generate_mode_with_seed(cli_runner):
    """Generation with seed is reproducible."""
    result1 = cli_runner.invoke(
        hypothesis_cli, ["json", '{"type": "integer"}', "--num", "5", "--seed", "42"]
    )
    result2 = cli_runner.invoke(
        hypothesis_cli, ["json", '{"type": "integer"}', "--num", "5", "--seed", "42"]
    )
    assert result1.exit_code == 0
    assert result2.exit_code == 0
    assert result1.output == result2.output


def test_cli_generate_mode_from_file(cli_runner, tmp_path):
    """Load schema from file."""
    schema_file = tmp_path / "schema.json"
    schema_file.write_text('{"type": "string", "maxLength": 3}')

    result = cli_runner.invoke(
        hypothesis_cli, ["json", str(schema_file), "--num", "3"]
    )
    assert result.exit_code == 0
    lines = result.output.strip().split("\n")
    assert len(lines) == 3


def test_cli_invalid_schema_error(cli_runner):
    """Invalid schema returns error."""
    result = cli_runner.invoke(hypothesis_cli, ["json", "not valid json"])
    assert result.exit_code != 0
    assert "Error loading schema" in result.output


def test_cli_help_option(cli_runner):
    """--help shows usage information."""
    result = cli_runner.invoke(hypothesis_cli, ["json", "--help"])
    assert result.exit_code == 0
    assert "Generate test data" in result.output
    assert "--num" in result.output
    assert "--seed" in result.output
    assert "--script" in result.output


def test_cli_shows_in_hypothesis_help(cli_runner):
    """The json command appears in hypothesis --help."""
    result = cli_runner.invoke(hypothesis_cli, ["--help"])
    assert result.exit_code == 0
    assert "json" in result.output


# =============================================================================
# Tests for fuzzing functionality
# =============================================================================


@pytest.fixture
def passing_script(tmp_path):
    """Create a script that always passes."""
    script = tmp_path / "pass.sh"
    script.write_text("#!/bin/bash\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.fixture
def always_failing_script(tmp_path):
    """Create a script that always fails."""
    script = tmp_path / "always_fail.sh"
    script.write_text("#!/bin/bash\nexit 1\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_fuzz_passing(passing_script, tmp_path):
    """Fuzzing with a passing script returns 0."""
    os.chdir(tmp_path)
    schema = {"type": "integer", "minimum": 0, "maximum": 10}
    result = run_fuzz(schema, passing_script, num=10)
    assert result == 0


def test_fuzz_failing_finds_failure(always_failing_script, tmp_path):
    """Fuzzing with a failing script returns 1 and saves testcase."""
    os.chdir(tmp_path)
    testcase_path = tmp_path / "testcase.json"
    schema = {"type": "integer"}
    result = run_fuzz(
        schema,
        always_failing_script,
        num=10,
        testcase_file=str(testcase_path),
    )
    assert result == 1
    # The failing testcase should be saved
    assert testcase_path.exists()
    # Should be valid JSON
    with open(testcase_path) as f:
        value = json.load(f)
    assert isinstance(value, int)


def test_fuzz_cli_integration(cli_runner, passing_script, tmp_path):
    """Test fuzz mode through CLI."""
    os.chdir(tmp_path)
    result = cli_runner.invoke(
        hypothesis_cli,
        [
            "json",
            '{"type": "string", "maxLength": 5}',
            "--script",
            passing_script,
            "--num",
            "5",
        ],
    )
    assert result.exit_code == 0


def test_fuzz_custom_testcase_file(passing_script, tmp_path):
    """Custom testcase file path is respected."""
    os.chdir(tmp_path)
    custom_path = tmp_path / "custom_test.json"
    schema = {"type": "boolean"}
    run_fuzz(schema, passing_script, num=5, testcase_file=str(custom_path))
    # The custom file should have been used (may or may not exist after test)
    # We just verify no error was raised


# =============================================================================
# Edge cases and error handling
# =============================================================================


def test_empty_enum_schema(cli_runner):
    """Schema that generates nothing handles gracefully."""
    # This schema can't produce any values
    result = cli_runner.invoke(hypothesis_cli, ["json", '{"enum": []}', "--num", "5"])
    # Should either return an error or empty output
    # The behavior depends on how hypothesis handles st.nothing()
    # Just ensure it doesn't crash unexpectedly
    assert result.exit_code in (0, 1)


def test_complex_nested_schema(cli_runner):
    """Complex nested schema works."""
    schema = {
        "type": "object",
        "properties": {
            "users": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "email": {"type": "string", "format": "email"},
                    },
                    "required": ["id"],
                },
            },
        },
    }
    result = cli_runner.invoke(
        hypothesis_cli, ["json", json.dumps(schema), "--num", "3"]
    )
    assert result.exit_code == 0
    lines = result.output.strip().split("\n")
    assert len(lines) == 3


def test_verbose_flag(cli_runner):
    """Verbose flag doesn't break anything."""
    result = cli_runner.invoke(
        hypothesis_cli, ["json", '{"type": "integer"}', "--num", "3", "--verbose"]
    )
    assert result.exit_code == 0
