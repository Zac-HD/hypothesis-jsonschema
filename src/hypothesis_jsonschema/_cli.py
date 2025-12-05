"""Command-line interface for hypothesis-jsonschema.

This module integrates with the Hypothesis CLI to provide JSON schema-based
test data generation and fuzzing capabilities.

Usage::

    hypothesis json schema.json --num 100
    hypothesis json '{"type": "integer"}' --num 10 --seed 42
    hypothesis json schema.json --script ./test.sh --num 1000

"""

import json as json_module
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Set

from hypothesis import (
    HealthCheck,
    Phase,
    Verbosity,
    given,
    settings,
)
from hypothesis.database import DirectoryBasedExampleDatabase

from . import from_schema
from ._encode import encode_canonical_json

# Lazy import click to match hypothesis.extra.cli pattern
try:
    import click
except ImportError:
    click = None  # type: ignore


def load_schema(schema_arg: str) -> dict:
    """Load a JSON schema from a string or file path.

    Args:
        schema_arg: Either a JSON string or a path to a JSON file.

    Returns:
        The parsed JSON schema as a dictionary.

    Raises:
        ValueError: If the schema cannot be parsed.
        FileNotFoundError: If a file path is given but doesn't exist.
    """
    # Check if it's a file path
    if os.path.isfile(schema_arg):
        with open(schema_arg, encoding="utf-8") as f:
            return json_module.load(f)
    # Otherwise, parse as JSON string
    try:
        return json_module.loads(schema_arg)
    except json_module.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse schema as JSON: {e}\n"
            f"If this is a file path, ensure the file exists."
        ) from e


def generate_examples(
    schema: dict,
    num: int,
    seed_value: Optional[int] = None,
) -> List[Any]:
    """Generate unique JSON examples conforming to a schema.

    This runs a Hypothesis test internally to generate examples. When a seed
    is provided, the output is deterministic.

    Args:
        schema: A JSON schema dictionary.
        num: The number of unique examples to generate.
        seed_value: Optional seed for reproducible output.

    Returns:
        A list of JSON-serializable values matching the schema.
    """
    results: List[Any] = []
    seen: Set[str] = set()
    strategy = from_schema(schema)

    # We need to collect enough unique examples
    # Use more attempts than requested to account for duplicates
    max_attempts = max(num * 10, 1000)

    # Build settings for generation-only mode
    test_settings = settings(
        max_examples=max_attempts,
        database=None,  # Don't persist anything
        phases=[Phase.generate],  # Only generate, no shrinking/replay
        suppress_health_check=list(HealthCheck),  # Allow slow/filter-heavy strategies
        deadline=None,  # No time limits
        verbosity=Verbosity.quiet,
    )

    class StopCollection(Exception):
        """Signal to stop collecting examples."""

    @test_settings
    @given(strategy)
    def collector(value: Any) -> None:
        canonical = encode_canonical_json(value)
        if canonical not in seen:
            seen.add(canonical)
            results.append(value)
        if len(results) >= num:
            raise StopCollection

    # Apply seed if provided
    if seed_value is not None:
        from hypothesis import seed

        collector = seed(seed_value)(collector)

    try:
        collector()
    except StopCollection:
        pass  # We collected enough examples
    except Exception as e:
        # Re-raise unexpected errors, but not the "stop" signal
        if not isinstance(e.__cause__, StopCollection):
            raise

    return results[:num]


def run_fuzz(
    schema: dict,
    script: str,
    num: int,
    testcase_file: str = "testcase.json",
    *,
    verbose: bool = False,
) -> int:
    """Fuzz an external program with schema-conforming inputs.

    This creates a Hypothesis test that:
    1. Generates a value conforming to the schema
    2. Writes it to a JSON file
    3. Executes the provided script
    4. If the script fails (non-zero exit), Hypothesis will shrink to find
       a minimal failing input.

    The database is keyed by the schema (not the script), so running with
    the same schema will replay interesting examples.

    Args:
        schema: A JSON schema dictionary.
        script: Path to or command for the test script.
        num: Number of examples to test (max_examples).
        testcase_file: Path where test cases are written (default: testcase.json).
        verbose: Whether to print progress information.

    Returns:
        0 if all tests pass, 1 if a failure was found.
    """
    strategy = from_schema(schema)

    # Create a database in a temp directory
    # This ensures we replay interesting examples across runs
    db_dir = Path(tempfile.gettempdir()) / "hypothesis-jsonschema-fuzz"
    db_dir.mkdir(exist_ok=True)

    test_settings = settings(
        max_examples=num,
        database=DirectoryBasedExampleDatabase(str(db_dir)),
        suppress_health_check=list(HealthCheck),
        deadline=None,  # External scripts may be slow
        verbosity=Verbosity.verbose if verbose else Verbosity.normal,
    )

    failure_found = False
    failure_example: Optional[Any] = None

    @test_settings
    @given(strategy)
    def fuzz_test(value: Any) -> None:
        nonlocal failure_found, failure_example

        # Write the test case to file
        with open(testcase_file, "w", encoding="utf-8") as f:
            json_module.dump(value, f, indent=2)

        # Run the external script
        result = subprocess.run(
            script,
            shell=True,
            capture_output=not verbose,
        )

        if result.returncode != 0:
            failure_found = True
            failure_example = value
            # Raise to trigger Hypothesis shrinking
            raise AssertionError(
                f"Script exited with code {result.returncode}\n"
                f"Test case written to: {testcase_file}"
            )

    try:
        fuzz_test()
    except AssertionError as e:
        # This is expected when a failure is found
        print(f"\n{'=' * 60}", file=sys.stderr)
        print("FAILURE FOUND!", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)
        print(f"Minimal failing example saved to: {testcase_file}", file=sys.stderr)
        if failure_example is not None:
            print(
                f"\nFailing input:\n{json_module.dumps(failure_example, indent=2)}",
                file=sys.stderr,
            )
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    if verbose:
        print(f"\nAll {num} test cases passed.", file=sys.stderr)
    return 0


# Only define CLI if click is available
if click is not None:
    import hypothesis.extra.cli

    @hypothesis.extra.cli.main.command()  # type: ignore[union-attr]
    @click.argument("schema", type=str)
    @click.option(
        "--num",
        "-n",
        type=int,
        default=100,
        help="Number of examples to generate (default: 100)",
    )
    @click.option(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducible output (generation mode only)",
    )
    @click.option(
        "--script",
        type=str,
        default=None,
        help="Script to run against each example (enables fuzz mode)",
    )
    @click.option(
        "--testcase",
        type=str,
        default="testcase.json",
        help="File path for test cases in fuzz mode (default: testcase.json)",
    )
    @click.option(
        "--verbose",
        "-v",
        is_flag=True,
        help="Enable verbose output",
    )
    def json(
        schema: str,
        num: int,
        seed: Optional[int],
        script: Optional[str],
        testcase: str,
        verbose: bool,  # noqa: FBT001
    ) -> None:
        """Generate test data from JSON schemata.

        SCHEMA can be a JSON string or a path to a JSON file.

        \b
        Examples:
            hypothesis json schema.json --num 100
            hypothesis json '{"type": "integer", "minimum": 0}' --num 10 --seed 42
            hypothesis json schema.json --script ./test.sh --num 1000
        """
        # Load the schema
        try:
            schema_dict = load_schema(schema)
        except (ValueError, FileNotFoundError, json_module.JSONDecodeError) as e:
            raise click.ClickException(f"Error loading schema: {e}") from e

        # Dispatch to the appropriate mode
        if script is not None:
            # Fuzz mode
            if seed is not None:
                click.echo(
                    "Warning: --seed is ignored in fuzz mode "
                    "(database handles reproducibility)",
                    err=True,
                )
            exit_code = run_fuzz(
                schema=schema_dict,
                script=script,
                num=num,
                testcase_file=testcase,
                verbose=verbose,
            )
            sys.exit(exit_code)
        else:
            # Generation mode
            try:
                examples = generate_examples(
                    schema=schema_dict,
                    num=num,
                    seed_value=seed,
                )
            except Exception as e:
                raise click.ClickException(f"Error generating examples: {e}") from e

            # Output one example per line
            for example in examples:
                click.echo(json_module.dumps(example))

            if verbose and len(examples) < num:
                click.echo(
                    f"Warning: Only generated {len(examples)} unique examples "
                    f"(requested {num})",
                    err=True,
                )
