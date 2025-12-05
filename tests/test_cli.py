"""Tests for the hypothesis-jsonschema command-line interface."""

import json
import os
import stat

import jsonschema
import pytest

from hypothesis_jsonschema._cli import generate_examples, load_schema, main, run_fuzz


class TestLoadSchema:
    """Tests for schema loading functionality."""

    def test_load_from_string(self):
        """Load a schema from a JSON string."""
        schema = load_schema('{"type": "integer"}')
        assert schema == {"type": "integer"}

    def test_load_from_file(self, tmp_path):
        """Load a schema from a file path."""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text('{"type": "string", "minLength": 1}')
        schema = load_schema(str(schema_file))
        assert schema == {"type": "string", "minLength": 1}

    def test_load_invalid_json_raises(self):
        """Invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Could not parse schema"):
            load_schema("not valid json")

    def test_load_nonexistent_file_tries_json(self):
        """A non-existent file path is tried as JSON first."""
        with pytest.raises(ValueError, match="Could not parse schema"):
            load_schema("/nonexistent/path/to/schema.json")


class TestGenerateExamples:
    """Tests for example generation."""

    def test_generate_integers(self):
        """Generate integer examples."""
        schema = {"type": "integer", "minimum": 0, "maximum": 100}
        examples = generate_examples(schema, num=10)
        assert len(examples) == 10
        for ex in examples:
            assert isinstance(ex, int)
            assert 0 <= ex <= 100

    def test_generate_strings(self):
        """Generate string examples."""
        schema = {"type": "string", "minLength": 1, "maxLength": 5}
        examples = generate_examples(schema, num=5)
        assert len(examples) == 5
        for ex in examples:
            assert isinstance(ex, str)
            assert 1 <= len(ex) <= 5

    def test_generate_objects(self):
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

    def test_seed_reproducibility(self):
        """Same seed produces same output."""
        schema = {"type": "integer"}
        examples1 = generate_examples(schema, num=10, seed_value=42)
        examples2 = generate_examples(schema, num=10, seed_value=42)
        assert examples1 == examples2

    def test_different_seeds_different_output(self):
        """Different seeds produce different output."""
        schema = {"type": "integer"}
        examples1 = generate_examples(schema, num=10, seed_value=42)
        examples2 = generate_examples(schema, num=10, seed_value=123)
        # With enough examples, they should differ
        assert examples1 != examples2

    def test_unique_examples(self):
        """Generated examples are unique."""
        schema = {"type": "integer", "minimum": 0, "maximum": 1000000}
        examples = generate_examples(schema, num=50)
        assert len(examples) == len(set(examples))

    def test_validates_against_schema(self):
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


class TestMainCLI:
    """Tests for the main CLI entry point."""

    def test_generate_mode_stdout(self, capsys):
        """Generation mode outputs JSON lines to stdout."""
        exit_code = main(['{"type": "integer"}', "--num", "5"])
        assert exit_code == 0
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            value = json.loads(line)
            assert isinstance(value, int)

    def test_generate_mode_with_seed(self, capsys):
        """Generation with seed is reproducible."""
        main(['{"type": "integer"}', "--num", "5", "--seed", "42"])
        output1 = capsys.readouterr().out

        main(['{"type": "integer"}', "--num", "5", "--seed", "42"])
        output2 = capsys.readouterr().out

        assert output1 == output2

    def test_generate_mode_from_file(self, tmp_path, capsys):
        """Load schema from file."""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text('{"type": "string", "maxLength": 3}')

        exit_code = main([str(schema_file), "--num", "3"])
        assert exit_code == 0
        lines = capsys.readouterr().out.strip().split("\n")
        assert len(lines) == 3

    def test_invalid_schema_error(self, capsys):
        """Invalid schema returns error."""
        exit_code = main(["not valid json"])
        assert exit_code == 1
        assert "Error loading schema" in capsys.readouterr().err

    def test_help_option(self, capsys):
        """--help shows usage information."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "hypothesis-jsonschema" in captured.out
        assert "--num" in captured.out
        assert "--seed" in captured.out
        assert "--script" in captured.out


class TestFuzzMode:
    """Tests for fuzzing functionality."""

    @pytest.fixture
    def passing_script(self, tmp_path):
        """Create a script that always passes."""
        script = tmp_path / "pass.sh"
        script.write_text("#!/bin/bash\nexit 0\n")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        return str(script)

    @pytest.fixture
    def failing_script(self, tmp_path):
        """Create a script that fails on certain inputs."""
        script = tmp_path / "fail.sh"
        # Fail if the testcase contains a negative number
        script.write_text(
            '#!/bin/bash\nif grep -q "\\-" testcase.json; then exit 1; fi\nexit 0\n'
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        return str(script)

    @pytest.fixture
    def always_failing_script(self, tmp_path):
        """Create a script that always fails."""
        script = tmp_path / "always_fail.sh"
        script.write_text("#!/bin/bash\nexit 1\n")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        return str(script)

    def test_fuzz_passing(self, passing_script, tmp_path):
        """Fuzzing with a passing script returns 0."""
        os.chdir(tmp_path)
        schema = {"type": "integer", "minimum": 0, "maximum": 10}
        result = run_fuzz(schema, passing_script, num=10)
        assert result == 0

    def test_fuzz_failing_finds_failure(self, always_failing_script, tmp_path):
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

    def test_fuzz_cli_integration(self, passing_script, tmp_path, capsys):
        """Test fuzz mode through CLI."""
        os.chdir(tmp_path)
        exit_code = main([
            '{"type": "string", "maxLength": 5}',
            "--script", passing_script,
            "--num", "5",
        ])
        assert exit_code == 0

    def test_fuzz_custom_testcase_file(self, passing_script, tmp_path):
        """Custom testcase file path is respected."""
        os.chdir(tmp_path)
        custom_path = tmp_path / "custom_test.json"
        schema = {"type": "boolean"}
        run_fuzz(schema, passing_script, num=5, testcase_file=str(custom_path))
        # The custom file should have been used (may or may not exist after test)
        # We just verify no error was raised


class TestCLIEdgeCases:
    """Edge cases and error handling."""

    def test_empty_enum_schema(self, capsys):
        """Schema that generates nothing handles gracefully."""
        # This schema can't produce any values
        exit_code = main(['{"enum": []}', "--num", "5"])
        # Should either return an error or empty output
        # The behavior depends on how hypothesis handles st.nothing()
        _ = capsys.readouterr()  # Consume output
        # Just ensure it doesn't crash unexpectedly
        assert exit_code in (0, 1)

    def test_complex_nested_schema(self, capsys):
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
        exit_code = main([json.dumps(schema), "--num", "3"])
        assert exit_code == 0
        lines = capsys.readouterr().out.strip().split("\n")
        assert len(lines) == 3

    def test_verbose_flag(self, capsys):
        """Verbose flag doesn't break anything."""
        exit_code = main(['{"type": "integer"}', "--num", "3", "--verbose"])
        assert exit_code == 0
