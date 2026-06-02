# ruff: noqa: E731
import struct
import re
import pytest
import sqlite3
import inspect
from contextlib import contextmanager

EXT_PATH = "./dist/lembed0"
MODEL1_PATH = "./dist/.models/all-MiniLM-L6-v2.e4ce9877.q8_0.gguf"


def connect(ext, path=":memory:", extra_entrypoint=None):
    db = sqlite3.connect(path)

    db.execute(
        "create temp table base_functions as select name from pragma_function_list"
    )
    db.execute("create temp table base_modules as select name from pragma_module_list")

    db.enable_load_extension(True)
    db.load_extension(ext)

    if extra_entrypoint:
        db.execute("select load_extension(?, ?)", [ext, extra_entrypoint])

    db.execute(
        "create temp table loaded_functions as select name from pragma_function_list where name not in (select name from base_functions) order by name"
    )
    db.execute(
        "create temp table loaded_modules as select name from pragma_module_list where name not in (select name from base_modules) order by name"
    )

    db.row_factory = sqlite3.Row
    return db



db = connect(EXT_PATH)


def explain_query_plan(sql):
    return db.execute("explain query plan " + sql).fetchone()["detail"]


def execute_all(cursor, sql, args=None):
    if args is None:
        args = []
    results = cursor.execute(sql, args).fetchall()
    return list(map(lambda x: dict(x), results))


def spread_args(args):
    return ",".join(["?"] * len(args))


@contextmanager
def _raises(message, error=sqlite3.OperationalError):
    with pytest.raises(error, match=re.escape(message)):
        yield


FUNCTIONS = [
    "_lembed_api",
    "lembed",
    "lembed",
    "lembed_context_options",
    "lembed_debug",
    "lembed_model_from_file",
    "lembed_model_options",
    "lembed_model_size",
    "lembed_token_score",
    "lembed_token_to_piece",
    "lembed_tokenize_json",
    "lembed_version",
]
MODULES = [
    "lembed_chunks",
    "lembed_models",
]


def test_funcs():
    funcs = list(
        map(
            lambda a: a[0],
            db.execute("select name from loaded_functions").fetchall(),
        )
    )
    assert funcs == FUNCTIONS


def test_modules():
    modules = list(
        map(lambda a: a[0], db.execute("select name from loaded_modules").fetchall())
    )
    assert modules == MODULES


def test_lembed_version():
    lembed_version = lambda *args: db.execute(
        "select lembed_version()", args
    ).fetchone()[0]
    assert lembed_version()[0] == "v"


def test_lembed_debug():
    lembed_debug = lambda *args: db.execute("select lembed_debug()", args).fetchone()[0]
    d = lembed_debug().split("\n")
    assert len(d) == 4


def test_lembed():
    db.execute(
        "insert into temp.lembed_models(name, model) values (?, lembed_model_from_file(?))",
        ["aaa", MODEL1_PATH],
    )
    lembed = lambda *args: db.execute(
        "select lembed({})".format(spread_args(args)), args
    ).fetchone()[0]
    a = lembed("aaa", "alex garcia")
    assert len(a) == (384 * 4)
    assert struct.unpack("1f", a[0:4])[0] == pytest.approx(
        0.002983473241329193, rel=1e-2
    )

    with _raises(
        "Unknown model name 'aaaaaaaaa'. Was it registered with lembed_models?"
    ):
        lembed("aaaaaaaaa", "alex garcia")

    with _raises("No default model has been registered yet with lembed_models"):
        lembed("alex garcia")

    db.execute(
        "insert into temp.lembed_models(name, model) values (?, lembed_model_from_file(?))",
        ["default", MODEL1_PATH],
    )
    a = lembed("alex garcia")
    assert len(a) == (384 * 4)
    assert struct.unpack("1f", a[0:4])[0] == pytest.approx(
        0.002983473241329193, rel=1e-2
    )


def test__lembed_api():
    # _lembed_api is an internal noop entrypoint; ensure it exists and is callable
    _lembed_api = lambda *args: db.execute("select _lembed_api()", args).fetchone()[0]
    # calling should not raise
    _lembed_api()


def test_lembed_context_options():
    # Ensure the context options factory returns a pointer-like object when called
    r = db.execute("select lembed_context_options('n_ctx', 512)").fetchone()[0]
    assert r is not None


def test_lembed_model_size():
    db.execute(
        "insert into temp.lembed_models(name, model) values (?, lembed_model_from_file(?))",
        ["size_test", MODEL1_PATH],
    )
    size = db.execute(
        "select lembed_model_size(model) from temp.lembed_models where name = ?",
        ["size_test"],
    ).fetchone()[0]
    assert isinstance(size, int)
    assert size > 0


def test_lembed_model_from_file():
    p = db.execute("select lembed_model_from_file(?)", [MODEL1_PATH]).fetchone()[0]
    assert p is not None


def test_lembed_model_options():
    r = db.execute("select lembed_model_options('n_gpu_layers', 0)").fetchone()[0]
    assert r is not None


def test_lembed_tokenize_json():
    db.execute(
        "insert into temp.lembed_models(name, model) values (?, lembed_model_from_file(?))",
        ["tok", MODEL1_PATH],
    )
    s = db.execute("select lembed_tokenize_json(?, ?)", ["tok", "hello world"]).fetchone()[0]
    import json
    tokens = json.loads(s)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)


def test_lembed_token_score():
    db.execute(
        "insert into temp.lembed_models(name, model) values (?, lembed_model_from_file(?))",
        ["score", MODEL1_PATH],
    )
    s = db.execute("select lembed_tokenize_json(?, ?)", ["score", "hello"]).fetchone()[0]
    import json
    tokens = json.loads(s)
    token = tokens[0]
    score = db.execute("select lembed_token_score(?, ?)", ["score", token]).fetchone()[0]
    assert isinstance(score, float)


def test_lembed_token_to_piece():
    db.execute(
        "insert into temp.lembed_models(name, model) values (?, lembed_model_from_file(?))",
        ["piece", MODEL1_PATH],
    )
    s = db.execute("select lembed_tokenize_json(?, ?)", ["piece", "hello"]).fetchone()[0]
    import json
    tokens = json.loads(s)
    token = tokens[0]
    piece = db.execute("select lembed_token_to_piece(?, ?)", ["piece", token]).fetchone()[0]
    assert isinstance(piece, str)
    assert len(piece) > 0


def test_lembed_chunks():
    db.execute(
        "insert into temp.lembed_models(name, model) values (?, lembed_model_from_file(?))",
        ["chunky", MODEL1_PATH],
    )
    # Query the virtual table; module expects (model_name, text)
    rows = db.execute("select contents from lembed_chunks(?, ?)", ["chunky", "The quick brown fox jumps over the lazy dog"]).fetchall()
    assert len(rows) > 0
    assert all(' ' in r[0] or len(r[0]) > 0 for r in rows)


def test_lembed_models():
    # Ensure lembed_models virtual table reflects inserted models
    db.execute(
        "insert into temp.lembed_models(name, model) values (?, lembed_model_from_file(?))",
        ["m1", MODEL1_PATH],
    )
    rows = db.execute("select name from temp.lembed_models").fetchall()
    names = [r[0] for r in rows]
    assert 'm1' in names


def test_coverage():
    current_module = inspect.getmodule(inspect.currentframe())
    test_methods = [
        member[0]
        for member in inspect.getmembers(current_module)
        if member[0].startswith("test_")
    ]
    funcs_with_tests = set([x.replace("test_", "") for x in test_methods])
    for func in [*FUNCTIONS, *MODULES]:
        assert func in funcs_with_tests, f"{func} is not tested"
